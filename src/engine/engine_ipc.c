// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "engine/engine_ipc.h"

#include <math.h>
#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_forward.h"      // mj_Euler (fallback)
#include "engine/engine_util_blas.h"    // mju_dot3
#include "engine/engine_util_errmem.h"  // mju_malloc, mju_free
#include "engine/engine_util_solve.h"   // mju_cholFactor, mju_cholSolve

// point-triangle: distance, closest point cp, barycentric weights w of cp (for the barrier gradient)
static mjtNum ipc_ptTri(const mjtNum* p, const mjtNum* a, const mjtNum* b, const mjtNum* c,
                        mjtNum* cp, mjtNum* w) {
  mjtNum ab[3], ac[3], ap[3];
  for (int k=0; k < 3; k++) { ab[k]=b[k]-a[k]; ac[k]=c[k]-a[k]; ap[k]=p[k]-a[k]; }
  mjtNum d1 = mju_dot3(ab, ap), d2 = mju_dot3(ac, ap);
  if (d1 <= 0 && d2 <= 0) { w[0]=1; w[1]=0; w[2]=0; }
  else {
    mjtNum bp[3]; for (int k=0; k < 3; k++) bp[k] = p[k]-b[k];
    mjtNum d3 = mju_dot3(ab, bp), d4 = mju_dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3) { w[0]=0; w[1]=1; w[2]=0; }
    else {
      mjtNum vc = d1*d4 - d3*d2;
      if (vc <= 0 && d1 >= 0 && d3 <= 0) { mjtNum t=d1/(d1-d3); w[0]=1-t; w[1]=t; w[2]=0; }
      else {
        mjtNum cq[3]; for (int k=0; k < 3; k++) cq[k] = p[k]-c[k];
        mjtNum d5 = mju_dot3(ab, cq), d6 = mju_dot3(ac, cq);
        if (d6 >= 0 && d5 <= d6) { w[0]=0; w[1]=0; w[2]=1; }
        else {
          mjtNum vb = d5*d2 - d1*d6;
          if (vb <= 0 && d2 >= 0 && d6 <= 0) { mjtNum t=d2/(d2-d6); w[0]=1-t; w[1]=0; w[2]=t; }
          else {
            mjtNum va = d3*d6 - d5*d4;
            if (va <= 0 && (d4-d3) >= 0 && (d5-d6) >= 0) {
              mjtNum t=(d4-d3)/((d4-d3)+(d5-d6)); w[0]=0; w[1]=1-t; w[2]=t;
            } else {
              mjtNum den = 1.0/(va+vb+vc), t = vb*den, u = vc*den; w[0]=1-t-u; w[1]=t; w[2]=u;
            }
          }
        }
      }
    }
  }
  for (int k=0; k < 3; k++) cp[k] = w[0]*a[k] + w[1]*b[k] + w[2]*c[k];
  mjtNum dd[3]; for (int k=0; k < 3; k++) dd[k] = p[k]-cp[k];
  return sqrt(mju_dot3(dd, dd));
}

// closest distance between segment p1p2 and segment q1q2; closest points cp1, cp2 and the segment
// parameters st = {s, t} (cp1 = p1+s*(p2-p1), cp2 = q1+t*(q2-q1)). No parallel-edge mollifier yet:
// the degenerate (near-parallel) branch falls back to s=0, which is fine for non-parallel crossings.
static mjtNum ipc_segSeg(const mjtNum* p1, const mjtNum* p2, const mjtNum* q1, const mjtNum* q2,
                         mjtNum* cp1, mjtNum* cp2, mjtNum* st) {
  mjtNum d1[3], d2[3], rr[3];
  for (int k=0; k < 3; k++) { d1[k]=p2[k]-p1[k]; d2[k]=q2[k]-q1[k]; rr[k]=p1[k]-q1[k]; }
  mjtNum a = mju_dot3(d1, d1), e = mju_dot3(d2, d2), fq = mju_dot3(d2, rr);
  mjtNum s, t;
  if (a <= 1e-12 && e <= 1e-12) { s = 0; t = 0; }
  else if (a <= 1e-12) { s = 0; t = fq/e; }
  else {
    mjtNum c = mju_dot3(d1, rr);
    if (e <= 1e-12) { t = 0; s = -c/a; }
    else {
      mjtNum b = mju_dot3(d1, d2), den = a*e - b*b;
      s = (den > 1e-12) ? (b*fq - c*e)/den : 0.0;   // parallel: s=0 (mollifier territory, deferred)
      s = (s < 0) ? 0 : (s > 1 ? 1 : s);
      t = (b*s + fq)/e;
      if (t < 0)      { t = 0; s = -c/a; }
      else if (t > 1) { t = 1; s = (b - c)/a; }
    }
  }
  s = (s < 0) ? 0 : (s > 1 ? 1 : s);
  t = (t < 0) ? 0 : (t > 1 ? 1 : t);
  st[0] = s; st[1] = t;
  for (int k=0; k < 3; k++) { cp1[k]=p1[k]+s*d1[k]; cp2[k]=q1[k]+t*d2[k]; }
  mjtNum dd[3]; for (int k=0; k < 3; k++) dd[k] = cp1[k]-cp2[k];
  return sqrt(mju_dot3(dd, dd));
}

// signed distance from a static geom's surface to world point x (positive outside) + outward unit
// normal n. Closed-form for plane/sphere/capsule/box; returns +large (no contact) for other types.
static mjtNum ipc_geomDist(int type, const mjtNum* size, const mjtNum* gpos, const mjtNum* gmat,
                           const mjtNum* x, mjtNum* n) {
  mjtNum dx[3]; for (int k=0; k < 3; k++) dx[k] = x[k]-gpos[k];
  if (type == mjGEOM_PLANE) {
    n[0]=gmat[2]; n[1]=gmat[5]; n[2]=gmat[8];   // local +z axis in world coords
    return mju_dot3(n, dx);
  }
  if (type == mjGEOM_SPHERE) {
    mjtNum L = sqrt(mju_dot3(dx, dx));
    if (L < 1e-12) { n[0]=0; n[1]=0; n[2]=1; return -size[0]; }
    for (int k=0; k < 3; k++) n[k]=dx[k]/L;
    return L - size[0];
  }
  mjtNum p[3]; mju_mulMatTVec3(p, gmat, dx);   // world -> geom-local
  mjtNum nl[3] = {0,0,0}, dist;
  if (type == mjGEOM_CAPSULE) {
    mjtNum hz = size[1], zc = p[2] > hz ? hz : (p[2] < -hz ? -hz : p[2]);
    mjtNum q[3] = {p[0], p[1], p[2]-zc};       // vector from nearest axis point
    mjtNum L = sqrt(mju_dot3(q, q));
    if (L < 1e-12) { nl[0]=1; } else { for (int k=0; k < 3; k++) nl[k]=q[k]/L; }
    dist = L - size[0];
  } else if (type == mjGEOM_BOX) {
    mjtNum q[3]; int outside = 0;
    for (int k=0; k < 3; k++) {
      mjtNum c = p[k] > size[k] ? size[k] : (p[k] < -size[k] ? -size[k] : p[k]);
      q[k] = p[k]-c; if (q[k] != 0) outside = 1;
    }
    if (outside) {
      mjtNum L = sqrt(mju_dot3(q, q)); dist = L;
      for (int k=0; k < 3; k++) nl[k]=q[k]/L;
    } else {                                    // inside: least-penetration face
      int ax = 0; mjtNum best = 1e30;
      for (int k=0; k < 3; k++) { mjtNum pen = size[k]-(p[k] < 0 ? -p[k] : p[k]); if (pen < best) { best = pen; ax = k; } }
      dist = -best; nl[ax] = p[ax] > 0 ? 1 : -1;
    }
  } else {
    n[0]=0; n[1]=0; n[2]=1; return 1e30;        // unsupported geom -> no barrier
  }
  mju_mulMatVec3(n, gmat, nl);                   // geom-local normal -> world
  return dist;
}

// world-space sharp corner vertices of a static geom (features that can poke through a flex face):
// box -> 8 corners; smooth/infinite geoms have none. Fills out (up to 8*3), returns the count.
static int ipc_geomVerts(int type, const mjtNum* size, const mjtNum* gpos, const mjtNum* gmat,
                         mjtNum* out) {
  if (type != mjGEOM_BOX) return 0;
  int n = 0;
  for (int sx=-1; sx <= 1; sx+=2) for (int sy=-1; sy <= 1; sy+=2) for (int sz=-1; sz <= 1; sz+=2) {
    mjtNum loc[3] = {sx*size[0], sy*size[1], sz*size[2]}, wc[3];
    mju_mulMatVec3(wc, gmat, loc);
    for (int k=0; k < 3; k++) out[3*n+k] = gpos[k] + wc[k];
    n++;
  }
  return n;
}

// world-space sharp edges of a static geom (a geom edge can slice through a flex edge between
// vertices): box -> 12 edges, each two endpoints (6 mjtNum); others none. Returns the edge count.
static int ipc_geomEdges(int type, const mjtNum* size, const mjtNum* gpos, const mjtNum* gmat,
                         mjtNum* out) {
  if (type != mjGEOM_BOX) return 0;
  int n = 0;
  for (int axis=0; axis < 3; axis++) {
    int a1 = (axis+1)%3, a2 = (axis+2)%3;
    for (int s1=-1; s1 <= 1; s1+=2) for (int s2=-1; s2 <= 1; s2+=2) {
      mjtNum lo[3], hi[3], w1[3], w2[3];
      lo[axis] = -size[axis]; hi[axis] = size[axis];
      lo[a1] = hi[a1] = s1*size[a1];
      lo[a2] = hi[a2] = s2*size[a2];
      mju_mulMatVec3(w1, gmat, lo);
      mju_mulMatVec3(w2, gmat, hi);
      for (int k=0; k < 3; k++) { out[6*n+k] = gpos[k]+w1[k]; out[6*n+3+k] = gpos[k]+w2[k]; }
      n++;
    }
  }
  return n;   // 12
}

// IPC log-barrier on a surface gap g (C-IPC offset) and its 1st/2nd derivatives, for 0 < g < gh
static mjtNum ipc_Bg (mjtNum g, mjtNum gh) { return (g > 0 && g < gh) ? -(g-gh)*(g-gh)*log(g/gh) : 0.0; }
static mjtNum ipc_Bd (mjtNum g, mjtNum gh) { mjtNum u=g-gh; return -2*u*log(g/gh) - u*u/g; }
static mjtNum ipc_Bdd(mjtNum g, mjtNum gh) { mjtNum u=g-gh; return -2*log(g/gh) - 4*u/g + u*u/(g*g); }

// IPC incremental-potential energy: inertia + edge-stretch + vertex-triangle barrier over the
// per-iteration active pair list (apair: 4 vertex ids per pair)
static mjtNum ipc_energy(const mjModel* m, const mjData* d, int nfv, int va, int ea, int en,
                         int ne, const int* el, const mjtNum* x, const mjtNum* xtil,
                         const int* fidx, const mjtNum* mass, const mjtNum* kE, mjtNum h, mjtNum r,
                         mjtNum ghat, mjtNum kappa, int napair, const int* apair,
                         int naedge, const int* aedge) {
  mjtNum E = 0, ih2 = 1.0/(h*h);
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum mh = mass[v]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  for (int e=0; e < en; e++) {
    int a = m->flex_edge[2*(ea+e)], b = m->flex_edge[2*(ea+e)+1];
    mjtNum dv[3] = {x[3*a]-x[3*b], x[3*a+1]-x[3*b+1], x[3*a+2]-x[3*b+2]};
    mjtNum L = sqrt(mju_dot3(dv, dv)), L0 = m->flexedge_length0[ea+e];
    E += 0.5*kE[e]*(L-L0)*(L-L0);
  }
  for (int p=0; p < napair; p++) {
    const int* pr = apair + 4*p; mjtNum cp[3], w[3];
    mjtNum g = ipc_ptTri(&x[3*pr[0]], &x[3*pr[1]], &x[3*pr[2]], &x[3*pr[3]], cp, w) - 2*r;
    if (g > 0 && g < ghat) E += kappa*ipc_Bg(g, ghat);
  }
  for (int p=0; p < naedge; p++) {
    const int* pr = aedge + 4*p; mjtNum cp1[3], cp2[3], stp[2];
    mjtNum g = ipc_segSeg(&x[3*pr[0]], &x[3*pr[1]], &x[3*pr[2]], &x[3*pr[3]], cp1, cp2, stp) - 2*r;
    if (g > 0 && g < ghat) E += kappa*ipc_Bg(g, ghat);
  }
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    for (int gi=0; gi < m->ngeom; gi++) {
      mjtNum nrm[3];
      mjtNum g = ipc_geomDist(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi,
                              d->geom_xmat+9*gi, &x[3*v], nrm) - r;
      if (g > 0 && g < ghat) E += kappa*ipc_Bg(g, ghat);
    }
  }
  for (int gi=0; gi < m->ngeom; gi++) {
    mjtNum gv[24]; int ngv = ipc_geomVerts(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi,
                                           d->geom_xmat+9*gi, gv);
    for (int c=0; c < ngv; c++) for (int e=0; e < ne; e++) {
      mjtNum cp[3], w[3];
      mjtNum g = ipc_ptTri(&gv[3*c], &x[3*el[3*e]], &x[3*el[3*e+1]], &x[3*el[3*e+2]], cp, w) - r;
      if (g > 0 && g < ghat) E += kappa*ipc_Bg(g, ghat);
    }
  }
  for (int gi=0; gi < m->ngeom; gi++) {
    mjtNum ge[72]; int nge = ipc_geomEdges(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi,
                                           d->geom_xmat+9*gi, ge);
    for (int c=0; c < nge; c++) for (int e=0; e < en; e++) {
      int a = m->flex_edge[2*(ea+e)], b = m->flex_edge[2*(ea+e)+1];
      mjtNum cp1[3], cp2[3], stp[2];
      mjtNum g = ipc_segSeg(&ge[6*c], &ge[6*c+3], &x[3*a], &x[3*b], cp1, cp2, stp) - r;
      if (g > 0 && g < ghat) E += kappa*ipc_Bg(g, ghat);
    }
  }
  return E;
}


// IPC-style variational integrator (PROTOTYPE, phases 0a/0b): a single 2D flex -- inertia + edge-
// stretch penalty + vertex-triangle SELF-CONTACT log-barrier -- by projected-Newton with a step-
// capped (crude-CCD) line search, starting each step from the last collision-free state.
// Intersection-free and freeze-free in validation, but PROTOTYPE quality: dense solve + per-step
// malloc (small flex only), vertex-triangle only (no edge-edge), step-cap is not a rigorous CCD,
// kappa hardcoded. Falls back to Euler if there is no 2D flex.
void mj_IPC(const mjModel* m, mjData* d) {
  mjtNum h = m->opt.timestep, tc = 0.02, kscale = 1000.0, kappa = 100.0;
  int f = -1;
  for (int i=0; i < m->nflex; i++) if (m->flex_dim[i] == 2) { f = i; break; }
  if (f < 0) { mj_Euler(m, d); return; }
  int nfv = m->flex_vertnum[f], va = m->flex_vertadr[f];
  int ea = m->flex_edgeadr[f], en = m->flex_edgenum[f];
  int ne = m->flex_elemnum[f];
  const int* el = m->flex_elem + m->flex_elemdataadr[f];
  mjtNum r = m->flex_radius[f], ghat = r;   // contact activates within ghat of the 2r surface gap

  int* dofadr = (int*) mju_malloc(nfv*sizeof(int));
  int* fidx   = (int*) mju_malloc(nfv*sizeof(int));
  mjtNum* mass = (mjtNum*) mju_malloc(nfv*sizeof(mjtNum));
  int nfree = 0;
  for (int v=0; v < nfv; v++) {
    int bid = m->flex_vertbodyid[va+v];
    dofadr[v] = -1; fidx[v] = -1; mass[v] = 0;
    if (m->body_dofnum[bid] == 3) {
      int da = m->body_dofadr[bid];
      dofadr[v] = da; fidx[v] = nfree++;
      mass[v] = d->qM[m->M_rowadr[da] + m->M_rownnz[da] - 1];   // diagonal (point mass)
    }
  }
  int N = 3*nfree, Na = (N > 0 ? N : 1);
  mjtNum* kE = (mjtNum*) mju_malloc(en*sizeof(mjtNum));
  for (int e=0; e < en; e++) {
    mjtNum iw = m->flexedge_invweight0[ea+e];
    kE[e] = kscale / ((iw > 1e-12 ? iw : 1e-9) * tc * tc);
  }
  int amax = 4*(nfv*16 + 1);                 // active vertex-triangle pairs, 4 vertex ids each
  int* apair = (int*) mju_malloc(amax*sizeof(int));
  int aemax = 4*(nfv*32 + 1);                 // active edge-edge pairs, 4 vertex ids each
  int* aedge = (int*) mju_malloc(aemax*sizeof(int));
  mjtNum* x    = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* xtil = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* xold = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* xn   = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* grad = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* dx   = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* rhs  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* H    = (mjtNum*) mju_malloc((size_t)Na*Na*sizeof(mjtNum));
  mjtNum* Hf   = (mjtNum*) mju_malloc((size_t)Na*Na*sizeof(mjtNum));

  const mjtNum* vx = d->flexvert_xpos;
  for (int v=0; v < nfv; v++) {
    for (int c=0; c < 3; c++) xold[3*v+c] = vx[3*(va+v)+c];
    if (fidx[v] >= 0) {
      int da = dofadr[v];
      for (int c=0; c < 3; c++)
        xtil[3*v+c] = vx[3*(va+v)+c] + h*d->qvel[da+c] + h*h*d->qacc_smooth[da+c];
    } else {
      for (int c=0; c < 3; c++) xtil[3*v+c] = vx[3*(va+v)+c];   // pinned: fixed
    }
  }
  for (int i=0; i < 3*nfv; i++) x[i] = xold[i];   // start from last collision-free state (feasibility)

  mjtNum ih2 = 1.0/(h*h);
  for (int it=0; it < 12 && N > 0; it++) {
    for (int i=0; i < N; i++) grad[i] = 0;
    for (int i=0; i < N*N; i++) H[i] = 0;
    for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
      int fi = fidx[v]; mjtNum mh = mass[v]*ih2;
      for (int c=0; c < 3; c++) { grad[3*fi+c] += mh*(x[3*v+c]-xtil[3*v+c]); H[(3*fi+c)*N+(3*fi+c)] += mh; }
    }
    for (int e=0; e < en; e++) {
      int a = m->flex_edge[2*(ea+e)], b = m->flex_edge[2*(ea+e)+1];
      mjtNum dv[3] = {x[3*a]-x[3*b], x[3*a+1]-x[3*b+1], x[3*a+2]-x[3*b+2]};
      mjtNum L = sqrt(mju_dot3(dv, dv));
      if (L < 1e-12) continue;
      mjtNum L0 = m->flexedge_length0[ea+e], k = kE[e];
      mjtNum dh[3] = {dv[0]/L, dv[1]/L, dv[2]/L};
      mjtNum g = k*(L-L0), t = (L > L0) ? k*(1.0 - L0/L) : 0.0;   // PSD-project tangential term
      mjtNum he[9];
      for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
        he[3*i+j] = k*dh[i]*dh[j] + t*((i==j?1.0:0.0) - dh[i]*dh[j]);
      int fa = fidx[a], fb = fidx[b];
      if (fa >= 0) for (int c=0; c < 3; c++) grad[3*fa+c] += g*dh[c];
      if (fb >= 0) for (int c=0; c < 3; c++) grad[3*fb+c] -= g*dh[c];
      for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {
        mjtNum val = he[3*i+j];
        if (fa >= 0) H[(3*fa+i)*N+(3*fa+j)] += val;
        if (fb >= 0) H[(3*fb+i)*N+(3*fb+j)] += val;
        if (fa >= 0 && fb >= 0) { H[(3*fa+i)*N+(3*fb+j)] -= val; H[(3*fb+i)*N+(3*fa+j)] -= val; }
      }
    }
    // vertex-triangle self-contact barrier: build the active set at x, assemble grad + GN Hessian
    int napair = 0;
    for (int v=0; v < nfv; v++) for (int e=0; e < ne; e++) {
      int A = el[3*e], B = el[3*e+1], C = el[3*e+2];
      if (v == A || v == B || v == C) continue;
      mjtNum cp[3], w[3];
      mjtNum dd = ipc_ptTri(&x[3*v], &x[3*A], &x[3*B], &x[3*C], cp, w);
      mjtNum g = dd - 2*r;
      if (g <= 0 || g >= ghat) continue;
      if (4*napair+3 < amax) { int* pr = apair+4*napair; pr[0]=v; pr[1]=A; pr[2]=B; pr[3]=C; napair++; }
      mjtNum nrm[3]; for (int c=0; c < 3; c++) nrm[c] = (x[3*v+c]-cp[c])/dd;   // dg/dp = n
      int idv[4] = {v, A, B, C}; mjtNum cw[4] = {1.0, -w[0], -w[1], -w[2]};   // dg/d(point) = cw*n
      mjtNum bd = kappa*ipc_Bd(g, ghat), bdd = kappa*ipc_Bdd(g, ghat);
      for (int p=0; p < 4; p++) { int fp = fidx[idv[p]]; if (fp < 0) continue;
        for (int c=0; c < 3; c++) grad[3*fp+c] += bd*cw[p]*nrm[c]; }
      for (int p=0; p < 4; p++) for (int q=0; q < 4; q++) {
        int fp = fidx[idv[p]], fq = fidx[idv[q]]; if (fp < 0 || fq < 0) continue;
        for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
          H[(3*fp+i)*N+(3*fq+j)] += bdd*cw[p]*cw[q]*nrm[i]*nrm[j];   // Gauss-Newton (PSD)
      }
    }
    // edge-edge self-contact barrier: same barrier/Hessian machinery, closest points via ipc_segSeg
    int naedge = 0;
    for (int e1=0; e1 < en; e1++) {
      int a1 = m->flex_edge[2*(ea+e1)], b1 = m->flex_edge[2*(ea+e1)+1];
      for (int e2=e1+1; e2 < en; e2++) {
        int a2 = m->flex_edge[2*(ea+e2)], b2 = m->flex_edge[2*(ea+e2)+1];
        if (a1==a2 || a1==b2 || b1==a2 || b1==b2) continue;   // skip edges sharing a vertex
        mjtNum cp1[3], cp2[3], stp[2];
        mjtNum dd = ipc_segSeg(&x[3*a1], &x[3*b1], &x[3*a2], &x[3*b2], cp1, cp2, stp);
        mjtNum g = dd - 2*r;
        if (g <= 0 || g >= ghat) continue;
        if (4*naedge+3 < aemax) { int* pr = aedge+4*naedge; pr[0]=a1; pr[1]=b1; pr[2]=a2; pr[3]=b2; naedge++; }
        mjtNum nrm[3]; for (int c=0; c < 3; c++) nrm[c] = (cp1[c]-cp2[c])/dd;   // dg/dcp1 = n
        int idv[4] = {a1, b1, a2, b2};
        mjtNum cw[4] = {1.0-stp[0], stp[0], -(1.0-stp[1]), -stp[1]};   // dg/d(endpoint) = cw*n
        mjtNum bd = kappa*ipc_Bd(g, ghat), bdd = kappa*ipc_Bdd(g, ghat);
        for (int p=0; p < 4; p++) { int fp = fidx[idv[p]]; if (fp < 0) continue;
          for (int c=0; c < 3; c++) grad[3*fp+c] += bd*cw[p]*nrm[c]; }
        for (int p=0; p < 4; p++) for (int q=0; q < 4; q++) {
          int fp = fidx[idv[p]], fq = fidx[idv[q]]; if (fp < 0 || fq < 0) continue;
          for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
            H[(3*fp+i)*N+(3*fq+j)] += bdd*cw[p]*cw[q]*nrm[i]*nrm[j];   // Gauss-Newton (PSD)
        }
      }
    }
    // flex-vs-static-geom barrier: each free vertex against each (fixed) geom, closed-form distance
    for (int v=0; v < nfv; v++) { int fv = fidx[v]; if (fv < 0) continue;
      for (int gi=0; gi < m->ngeom; gi++) {
        mjtNum nrm[3];
        mjtNum dgg = ipc_geomDist(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi,
                                  d->geom_xmat+9*gi, &x[3*v], nrm);
        mjtNum g = dgg - r;   // vertex inflated by flex radius; geom is solid (dg/dx = n)
        if (g <= 0 || g >= ghat) continue;
        mjtNum bd = kappa*ipc_Bd(g, ghat), bdd = kappa*ipc_Bdd(g, ghat);
        for (int c=0; c < 3; c++) grad[3*fv+c] += bd*nrm[c];
        for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
          H[(3*fv+i)*N+(3*fv+j)] += bdd*nrm[i]*nrm[j];   // Gauss-Newton (PSD), vertex-only block
      }
    }
    // static-geom-corner vs flex-triangle barrier: a sharp geom corner poking through a flex face
    // (the dual of vertex-vs-geom; corner is fixed, gradient lands on the 3 triangle vertices)
    for (int gi=0; gi < m->ngeom; gi++) {
      mjtNum gv[24]; int ngv = ipc_geomVerts(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi,
                                             d->geom_xmat+9*gi, gv);
      for (int c=0; c < ngv; c++) for (int e=0; e < ne; e++) {
        int A = el[3*e], B = el[3*e+1], C = el[3*e+2];
        mjtNum cp[3], w[3];
        mjtNum dd = ipc_ptTri(&gv[3*c], &x[3*A], &x[3*B], &x[3*C], cp, w);
        mjtNum g = dd - r;
        if (g <= 0 || g >= ghat) continue;
        mjtNum nrm[3]; for (int k=0; k < 3; k++) nrm[k] = (gv[3*c+k]-cp[k])/dd;   // dg/d(corner)=n
        int idv[3] = {A, B, C}; mjtNum cw[3] = {-w[0], -w[1], -w[2]};   // corner fixed; tri gets -w*n
        mjtNum bd = kappa*ipc_Bd(g, ghat), bdd = kappa*ipc_Bdd(g, ghat);
        for (int p=0; p < 3; p++) { int fp = fidx[idv[p]]; if (fp < 0) continue;
          for (int k=0; k < 3; k++) grad[3*fp+k] += bd*cw[p]*nrm[k]; }
        for (int p=0; p < 3; p++) for (int q=0; q < 3; q++) {
          int fp = fidx[idv[p]], fq = fidx[idv[q]]; if (fp < 0 || fq < 0) continue;
          for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
            H[(3*fp+i)*N+(3*fq+j)] += bdd*cw[p]*cw[q]*nrm[i]*nrm[j];   // Gauss-Newton (PSD)
        }
      }
    }
    // static-geom-edge vs flex-edge barrier: a sharp geom edge slicing through a flex edge between
    // vertices (the third feature pair; geom edge fixed, gradient lands on the 2 flex endpoints)
    for (int gi=0; gi < m->ngeom; gi++) {
      mjtNum ge[72]; int nge = ipc_geomEdges(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi,
                                             d->geom_xmat+9*gi, ge);
      for (int c=0; c < nge; c++) for (int e=0; e < en; e++) {
        int a = m->flex_edge[2*(ea+e)], b = m->flex_edge[2*(ea+e)+1];
        mjtNum cp1[3], cp2[3], stp[2];
        mjtNum dd = ipc_segSeg(&ge[6*c], &ge[6*c+3], &x[3*a], &x[3*b], cp1, cp2, stp);
        mjtNum g = dd - r;
        if (g <= 0 || g >= ghat) continue;
        mjtNum nrm[3]; for (int k=0; k < 3; k++) nrm[k] = (cp1[k]-cp2[k])/dd;   // dg/dcp1 = n
        int idv[2] = {a, b}; mjtNum cw[2] = {-(1.0-stp[1]), -stp[1]};   // geom edge fixed; flex -w*n
        mjtNum bd = kappa*ipc_Bd(g, ghat), bdd = kappa*ipc_Bdd(g, ghat);
        for (int p=0; p < 2; p++) { int fp = fidx[idv[p]]; if (fp < 0) continue;
          for (int k=0; k < 3; k++) grad[3*fp+k] += bd*cw[p]*nrm[k]; }
        for (int p=0; p < 2; p++) for (int q=0; q < 2; q++) {
          int fp = fidx[idv[p]], fq = fidx[idv[q]]; if (fp < 0 || fq < 0) continue;
          for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
            H[(3*fp+i)*N+(3*fq+j)] += bdd*cw[p]*cw[q]*nrm[i]*nrm[j];   // Gauss-Newton (PSD)
        }
      }
    }
    mjtNum gn = 0; for (int i=0; i < N; i++) gn += grad[i]*grad[i];
    if (sqrt(gn) < 1e-8) break;
    for (int i=0; i < N*N; i++) Hf[i] = H[i];
    mju_cholFactor(Hf, N, 1e-12);
    for (int i=0; i < N; i++) rhs[i] = -grad[i];
    mju_cholSolve(dx, Hf, rhs, N);
    // step-capped (crude-CCD) backtracking line search: cap the max vertex move at 0.4*ghat so no
    // pair tunnels in one step, then backtrack on the energy (which spikes as a contact closes)
    mjtNum mx = 0; for (int i=0; i < N; i++) { mjtNum a = dx[i] < 0 ? -dx[i] : dx[i]; if (a > mx) mx = a; }
    mjtNum cap = (mx > 0.4*ghat) ? (0.4*ghat/mx) : 1.0;
    mjtNum E0 = ipc_energy(m, d, nfv, va, ea, en, ne, el, x, xtil, fidx, mass, kE, h, r, ghat, kappa,
                           napair, apair, naedge, aedge);
    mjtNum gdx = 0; for (int i=0; i < N; i++) gdx += grad[i]*dx[i];
    mjtNum alpha = cap;
    for (int ls=0; ls < 25; ls++) {
      for (int i=0; i < 3*nfv; i++) xn[i] = x[i];
      for (int v=0; v < nfv; v++) if (fidx[v] >= 0) { int fi = fidx[v];
        for (int c=0; c < 3; c++) xn[3*v+c] = x[3*v+c] + alpha*dx[3*fi+c]; }
      if (ipc_energy(m, d, nfv, va, ea, en, ne, el, xn, xtil, fidx, mass, kE, h, r, ghat, kappa,
                     napair, apair, naedge, aedge) <= E0 + 1e-4*alpha*gdx) break;
      alpha *= 0.5;
    }
    for (int i=0; i < 3*nfv; i++) x[i] = xn[i];
  }
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    int da = dofadr[v];
    for (int c=0; c < 3; c++) { mjtNum dp = x[3*v+c]-xold[3*v+c]; d->qvel[da+c] = dp/h; d->qpos[da+c] += dp; }
  }
  d->time += h;

  mju_free(dofadr); mju_free(fidx); mju_free(mass); mju_free(kE); mju_free(apair); mju_free(aedge);
  mju_free(x); mju_free(xtil); mju_free(xold); mju_free(xn);
  mju_free(grad); mju_free(dx); mju_free(rhs); mju_free(H); mju_free(Hf);
}
