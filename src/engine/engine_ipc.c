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

// one active contact. type: 0 vertex-triangle self, 1 edge-edge self, 2 flex-vertex vs geom
// surface, 3 geom-corner vs flex-triangle, 4 geom-edge vs flex-edge. idx/gi meaning per type
// (see ipc_conGap). The geom side is static, so its features (gv/ge) are precomputed once per step.
typedef struct { int type; int idx[4]; int gi; } ipcCon;

// gap g of a contact at configuration x, plus the barrier gradient direction n, the involved flex
// vertices idv[*nidx] and their weights cw (dg/d(vertex_p) = cw[p]*n). gv/ge are the precomputed
// world-space static-geom corners/edges. Single source of the per-type contact geometry.
static mjtNum ipc_conGap(const ipcCon* con, const mjModel* m, const mjData* d, const mjtNum* x,
                         const mjtNum* gv, const mjtNum* ge, mjtNum r,
                         mjtNum* n, int* idv, mjtNum* cw, int* nidx) {
  switch (con->type) {
  case 0: {   // vertex-triangle self-contact (v against triangle A,B,C)
    int v=con->idx[0], A=con->idx[1], B=con->idx[2], C=con->idx[3];
    mjtNum cp[3], w[3], dd = ipc_ptTri(&x[3*v], &x[3*A], &x[3*B], &x[3*C], cp, w);
    for (int k=0; k < 3; k++) n[k] = (x[3*v+k]-cp[k])/dd;
    idv[0]=v; idv[1]=A; idv[2]=B; idv[3]=C;
    cw[0]=1; cw[1]=-w[0]; cw[2]=-w[1]; cw[3]=-w[2]; *nidx=4;
    return dd - 2*r;
  }
  case 1: {   // edge-edge self-contact (edge a1b1 against edge a2b2)
    int a1=con->idx[0], b1=con->idx[1], a2=con->idx[2], b2=con->idx[3];
    mjtNum cp1[3], cp2[3], st[2], dd = ipc_segSeg(&x[3*a1], &x[3*b1], &x[3*a2], &x[3*b2], cp1, cp2, st);
    for (int k=0; k < 3; k++) n[k] = (cp1[k]-cp2[k])/dd;
    idv[0]=a1; idv[1]=b1; idv[2]=a2; idv[3]=b2;
    cw[0]=1-st[0]; cw[1]=st[0]; cw[2]=-(1-st[1]); cw[3]=-st[1]; *nidx=4;
    return dd - 2*r;
  }
  case 2: {   // flex-vertex v vs static geom gi surface
    int v=con->idx[0], gi=con->gi;
    mjtNum dd = ipc_geomDist(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi,
                             d->geom_xmat+9*gi, &x[3*v], n);
    idv[0]=v; cw[0]=1; *nidx=1;
    return dd - r;
  }
  case 3: {   // static geom corner gv[idx0] vs flex triangle A,B,C
    const mjtNum* corner = &gv[3*con->idx[0]];
    int A=con->idx[1], B=con->idx[2], C=con->idx[3];
    mjtNum cp[3], w[3], dd = ipc_ptTri(corner, &x[3*A], &x[3*B], &x[3*C], cp, w);
    for (int k=0; k < 3; k++) n[k] = (corner[k]-cp[k])/dd;
    idv[0]=A; idv[1]=B; idv[2]=C; cw[0]=-w[0]; cw[1]=-w[1]; cw[2]=-w[2]; *nidx=3;
    return dd - r;
  }
  default: {  // case 4: static geom edge ge[idx0] vs flex edge a,b
    const mjtNum* eg = &ge[6*con->idx[0]];
    int a=con->idx[1], b=con->idx[2];
    mjtNum cp1[3], cp2[3], st[2], dd = ipc_segSeg(eg, eg+3, &x[3*a], &x[3*b], cp1, cp2, st);
    for (int k=0; k < 3; k++) n[k] = (cp1[k]-cp2[k])/dd;
    idv[0]=a; idv[1]=b; cw[0]=-(1-st[1]); cw[1]=-st[1]; *nidx=2;
    return dd - r;
  }
  }
}

// cached per-contact data for the matrix-free Hessian apply: GN block = bdd * (cw[p]*n)(cw[q]*n)^T
// over the involved free-dof indices f[0..nidx) (f<0 = pinned, skipped).
typedef struct { mjtNum n[3], cw[4], bdd; int f[4], nidx; } ipcCC;

// evaluate a candidate contact at x; if active (0<g<ghat) accumulate its barrier gradient and the
// Hessian diagonal (for the Jacobi preconditioner), and cache the GN block (ccache) + contact (acon).
static void ipc_try(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x,
                    const mjtNum* gv, const mjtNum* ge, mjtNum r, mjtNum ghat, mjtNum kappa,
                    const int* fidx, mjtNum* grad, mjtNum* diagH,
                    ipcCon* acon, ipcCC* ccache, int* nacon, int amax) {
  if (*nacon >= amax) return;
  mjtNum n[3], cw[4]; int idv[4], nidx;
  mjtNum g = ipc_conGap(&con, m, d, x, gv, ge, r, n, idv, cw, &nidx);
  if (g <= 0 || g >= ghat) return;
  mjtNum bd = kappa*ipc_Bd(g, ghat), bdd = kappa*ipc_Bdd(g, ghat);
  ipcCC* cc = &ccache[*nacon];
  cc->bdd = bdd; cc->nidx = nidx;
  for (int k=0; k < 3; k++) cc->n[k] = n[k];
  for (int p=0; p < nidx; p++) {
    int fp = fidx[idv[p]];
    cc->cw[p] = cw[p]; cc->f[p] = fp;
    if (fp < 0) continue;
    for (int k=0; k < 3; k++) {
      grad[3*fp+k]  += bd*cw[p]*n[k];
      diagH[3*fp+k] += bdd*cw[p]*cw[p]*n[k]*n[k];
    }
  }
  acon[*nacon] = con;
  (*nacon)++;
}

// matrix-free Hessian-vector product Hp = H*p in the free-dof space (size N): inertia (mdiag) +
// stretch (per-edge 3x3 block he between free dofs efa,efb) + contact Gauss-Newton blocks (ccache).
static void ipc_applyH(const mjtNum* p, mjtNum* Hp, int N, const mjtNum* mdiag,
                       int en, const mjtNum* he, const int* efa, const int* efb,
                       const ipcCC* ccache, int nacon) {
  for (int i=0; i < N; i++) Hp[i] = mdiag[i]*p[i];
  for (int e=0; e < en; e++) {
    int fa = efa[e], fb = efb[e]; const mjtNum* h = he+9*e;
    mjtNum dv[3];
    for (int k=0; k < 3; k++) dv[k] = (fa>=0 ? p[3*fa+k]:0) - (fb>=0 ? p[3*fb+k]:0);
    mjtNum hd[3];
    for (int i=0; i < 3; i++) hd[i] = h[3*i]*dv[0]+h[3*i+1]*dv[1]+h[3*i+2]*dv[2];
    if (fa >= 0) for (int k=0; k < 3; k++) Hp[3*fa+k] += hd[k];
    if (fb >= 0) for (int k=0; k < 3; k++) Hp[3*fb+k] -= hd[k];
  }
  for (int c=0; c < nacon; c++) {
    const ipcCC* cc = &ccache[c];
    mjtNum s = 0;
    for (int q=0; q < cc->nidx; q++) { int fq = cc->f[q]; if (fq < 0) continue;
      s += cc->cw[q]*(cc->n[0]*p[3*fq]+cc->n[1]*p[3*fq+1]+cc->n[2]*p[3*fq+2]); }
    for (int q=0; q < cc->nidx; q++) { int fp = cc->f[q]; if (fp < 0) continue;
      mjtNum a = cc->bdd*cc->cw[q]*s;
      for (int k=0; k < 3; k++) Hp[3*fp+k] += a*cc->n[k]; }
  }
}

// matrix-free preconditioned CG: solve H dx = -grad with a Jacobi (diagH) preconditioner
static void ipc_pcg(mjtNum* dx, const mjtNum* grad, int N, const mjtNum* diagH, const mjtNum* mdiag,
                    int en, const mjtNum* he, const int* efa, const int* efb,
                    const ipcCC* ccache, int nacon, mjtNum* rcg, mjtNum* zcg, mjtNum* pcg, mjtNum* Hp) {
  mjtNum rz = 0, r0 = 0;
  for (int i=0; i < N; i++) {
    dx[i] = 0; rcg[i] = -grad[i]; zcg[i] = rcg[i]/diagH[i];
    pcg[i] = zcg[i]; rz += rcg[i]*zcg[i]; r0 += rcg[i]*rcg[i];
  }
  if (r0 < 1e-30) return;
  for (int it=0; it < 200; it++) {
    ipc_applyH(pcg, Hp, N, mdiag, en, he, efa, efb, ccache, nacon);
    mjtNum pHp = 0; for (int i=0; i < N; i++) pHp += pcg[i]*Hp[i];
    if (pHp <= 1e-30) break;
    mjtNum alpha = rz/pHp;
    mjtNum rr = 0;
    for (int i=0; i < N; i++) { dx[i] += alpha*pcg[i]; rcg[i] -= alpha*Hp[i]; rr += rcg[i]*rcg[i]; }
    if (rr < 1e-8*r0) break;
    mjtNum rznew = 0;
    for (int i=0; i < N; i++) { zcg[i] = rcg[i]/diagH[i]; rznew += rcg[i]*zcg[i]; }
    mjtNum beta = rznew/rz; rz = rznew;
    for (int i=0; i < N; i++) pcg[i] = zcg[i] + beta*pcg[i];
  }
}

// IPC incremental-potential energy: inertia + edge-stretch penalty + contact barriers over the
// per-iteration active contact list acon (cached so the line search doesn't re-enumerate all pairs)
static mjtNum ipc_energy(const mjModel* m, const mjData* d, int nfv, int ea, int en,
                         const mjtNum* x, const mjtNum* xtil, const int* fidx, const mjtNum* mass,
                         const mjtNum* kE, mjtNum h, mjtNum r, mjtNum ghat, mjtNum kappa,
                         const mjtNum* gv, const mjtNum* ge, const ipcCon* acon, int nacon) {
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
  for (int c=0; c < nacon; c++) {
    mjtNum n[3], cw[4]; int idv[4], nidx;
    mjtNum g = ipc_conGap(&acon[c], m, d, x, gv, ge, r, n, idv, cw, &nidx);
    if (g > 0 && g < ghat) E += kappa*ipc_Bg(g, ghat);
  }
  return E;
}


// append a candidate contact if its gap at x is below the (margin-inflated) detection threshold
static void ipc_addCand(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x,
                        const mjtNum* gv, const mjtNum* ge, mjtNum r, mjtNum thresh,
                        ipcCon* cand, int* nc, int candmax) {
  if (*nc >= candmax) return;
  mjtNum n[3], cw[4]; int idv[4], nidx;
  mjtNum g = ipc_conGap(&con, m, d, x, gv, ge, r, n, idv, cw, &nidx);
  if (g > 0 && g < thresh) cand[(*nc)++] = con;
}

// build the candidate-contact list once per step (brute enumeration at x, gated by a velocity-aware
// threshold so any pair that could close within the step is captured). The Newton loop then only
// re-tests these candidates each iteration instead of re-enumerating all O(n^2) pairs.
static int ipc_candidates(const mjModel* m, const mjData* d, const mjtNum* x, const mjtNum* gv,
                          const mjtNum* ge, int ngv, int nge, mjtNum r, mjtNum thresh,
                          int nfv, int ne, const int* el, int en, int ea, const int* fidx,
                          ipcCon* cand, int candmax) {
  int nc = 0;
  for (int v=0; v < nfv; v++) for (int e=0; e < ne; e++) {                  // vertex-triangle self
    int A = el[3*e], B = el[3*e+1], C = el[3*e+2];
    if (v == A || v == B || v == C) continue;
    ipcCon con = {0, {v, A, B, C}, -1};
    ipc_addCand(con, m, d, x, gv, ge, r, thresh, cand, &nc, candmax);
  }
  for (int e1=0; e1 < en; e1++) {                                          // edge-edge self
    int a1 = m->flex_edge[2*(ea+e1)], b1 = m->flex_edge[2*(ea+e1)+1];
    for (int e2=e1+1; e2 < en; e2++) {
      int a2 = m->flex_edge[2*(ea+e2)], b2 = m->flex_edge[2*(ea+e2)+1];
      if (a1==a2 || a1==b2 || b1==a2 || b1==b2) continue;
      ipcCon con = {1, {a1, b1, a2, b2}, -1};
      ipc_addCand(con, m, d, x, gv, ge, r, thresh, cand, &nc, candmax);
    }
  }
  for (int gi=0; gi < m->ngeom; gi++) for (int v=0; v < nfv; v++) {          // flex-vertex vs geom
    if (fidx[v] < 0) continue;
    ipcCon con = {2, {v, 0, 0, 0}, gi};
    ipc_addCand(con, m, d, x, gv, ge, r, thresh, cand, &nc, candmax);
  }
  for (int c=0; c < ngv; c++) for (int e=0; e < ne; e++) {                  // geom-corner vs triangle
    ipcCon con = {3, {c, el[3*e], el[3*e+1], el[3*e+2]}, -1};
    ipc_addCand(con, m, d, x, gv, ge, r, thresh, cand, &nc, candmax);
  }
  for (int c=0; c < nge; c++) for (int e=0; e < en; e++) {                  // geom-edge vs flex-edge
    ipcCon con = {4, {c, m->flex_edge[2*(ea+e)], m->flex_edge[2*(ea+e)+1], 0}, -1};
    ipc_addCand(con, m, d, x, gv, ge, r, thresh, cand, &nc, candmax);
  }
  return nc;
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
  int amax = nfv*64 + 1024;                   // capacity of the active-contact list
  ipcCon* acon = (ipcCon*) mju_malloc(amax*sizeof(ipcCon));
  ipcCC* ccache = (ipcCC*) mju_malloc(amax*sizeof(ipcCC));
  int candmax = nfv*128 + 4096;               // capacity of the per-step candidate list
  ipcCon* cand = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon));
  // precompute static-geom sharp features (corners/edges) once per step (geoms are fixed here)
  int gvcap = 8*m->ngeom + 1, gecap = 12*m->ngeom + 1, ngv = 0, nge = 0;
  mjtNum* gv = (mjtNum*) mju_malloc(3*gvcap*sizeof(mjtNum));
  mjtNum* ge = (mjtNum*) mju_malloc(6*gecap*sizeof(mjtNum));
  for (int gi=0; gi < m->ngeom; gi++) {
    ngv += ipc_geomVerts(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, gv+3*ngv);
    nge += ipc_geomEdges(m->geom_type[gi], m->geom_size+3*gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, ge+6*nge);
  }
  mjtNum* x    = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* xtil = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* xold = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* xn   = (mjtNum*) mju_malloc(3*nfv*sizeof(mjtNum));
  mjtNum* grad = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* dx   = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* mdiag= (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // inertia diagonal (Jacobi precond base)
  mjtNum* diagH= (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // full Hessian diagonal (preconditioner)
  mjtNum* rcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // CG residual / search-dir / Hp buffers
  mjtNum* zcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* pcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* Hpv  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* he   = (mjtNum*) mju_malloc((en > 0 ? 9*en : 1)*sizeof(mjtNum));  // per-edge stretch block
  int* efa     = (int*) mju_malloc((en > 0 ? en : 1)*sizeof(int));         // edge free-dof indices
  int* efb     = (int*) mju_malloc((en > 0 ? en : 1)*sizeof(int));

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
  for (int i=0; i < N; i++) mdiag[i] = 0;                          // inertia diagonal (fixed per step)
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    int fi = fidx[v]; for (int c=0; c < 3; c++) mdiag[3*fi+c] = mass[v]*ih2;
  }
  for (int e=0; e < en; e++) {                                     // edge endpoints -> free-dof indices
    efa[e] = fidx[m->flex_edge[2*(ea+e)]]; efb[e] = fidx[m->flex_edge[2*(ea+e)+1]];
  }
  // build the candidate-contact list once per step: detection threshold inflated by the predictor
  // displacement so any pair that could close during the step is captured (verified by gap checks)
  mjtNum maxdisp = 0;
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum dd[3]; for (int c=0; c < 3; c++) dd[c] = xtil[3*v+c]-xold[3*v+c];
    mjtNum L = sqrt(mju_dot3(dd, dd)); if (L > maxdisp) maxdisp = L;
  }
  mjtNum thresh = 3*ghat + 4*maxdisp;
  int ncand = ipc_candidates(m, d, x, gv, ge, ngv, nge, r, thresh, nfv, ne, el, en, ea, fidx,
                             cand, candmax);
  for (int it=0; it < 12 && N > 0; it++) {
    for (int i=0; i < N; i++) { grad[i] = 0; diagH[i] = mdiag[i]; }
    for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
      int fi = fidx[v]; mjtNum mh = mass[v]*ih2;
      for (int c=0; c < 3; c++) grad[3*fi+c] += mh*(x[3*v+c]-xtil[3*v+c]);
    }
    for (int e=0; e < en; e++) {
      int a = m->flex_edge[2*(ea+e)], b = m->flex_edge[2*(ea+e)+1], fa = efa[e], fb = efb[e];
      mjtNum* heb = he + 9*e;
      mjtNum dv[3] = {x[3*a]-x[3*b], x[3*a+1]-x[3*b+1], x[3*a+2]-x[3*b+2]};
      mjtNum L = sqrt(mju_dot3(dv, dv));
      if (L < 1e-12) { for (int i=0; i < 9; i++) heb[i] = 0; continue; }
      mjtNum L0 = m->flexedge_length0[ea+e], k = kE[e];
      mjtNum dh[3] = {dv[0]/L, dv[1]/L, dv[2]/L};
      mjtNum gg = k*(L-L0), t = (L > L0) ? k*(1.0 - L0/L) : 0.0;   // PSD-project tangential term
      for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
        heb[3*i+j] = k*dh[i]*dh[j] + t*((i==j?1.0:0.0) - dh[i]*dh[j]);   // 3x3 block (off-diag in applyH)
      if (fa >= 0) for (int c=0; c < 3; c++) { grad[3*fa+c] += gg*dh[c]; diagH[3*fa+c] += heb[3*c+c]; }
      if (fb >= 0) for (int c=0; c < 3; c++) { grad[3*fb+c] -= gg*dh[c]; diagH[3*fb+c] += heb[3*c+c]; }
    }
    // assemble active contacts: re-test the per-step candidate list at the current x (gate 0<g<ghat),
    // accumulating grad + Hessian diagonal and caching the GN blocks (acon/ccache) for the solve + LS
    int nacon = 0;
    for (int c=0; c < ncand; c++)
      ipc_try(cand[c], m, d, x, gv, ge, r, ghat, kappa, fidx, grad, diagH, acon, ccache, &nacon, amax);
    mjtNum gn = 0; for (int i=0; i < N; i++) gn += grad[i]*grad[i];
    if (sqrt(gn) < 1e-8) break;
    ipc_pcg(dx, grad, N, diagH, mdiag, en, he, efa, efb, ccache, nacon, rcg, zcg, pcg, Hpv);
    // step-capped (crude-CCD) backtracking line search: cap the max vertex move at 0.4*ghat so no
    // pair tunnels in one step, then backtrack on the energy (which spikes as a contact closes)
    mjtNum mx = 0; for (int i=0; i < N; i++) { mjtNum a = dx[i] < 0 ? -dx[i] : dx[i]; if (a > mx) mx = a; }
    mjtNum cap = (mx > 0.4*ghat) ? (0.4*ghat/mx) : 1.0;
    mjtNum E0 = ipc_energy(m, d, nfv, ea, en, x, xtil, fidx, mass, kE, h, r, ghat, kappa,
                           gv, ge, acon, nacon);
    mjtNum gdx = 0; for (int i=0; i < N; i++) gdx += grad[i]*dx[i];
    mjtNum alpha = cap;
    for (int ls=0; ls < 25; ls++) {
      for (int i=0; i < 3*nfv; i++) xn[i] = x[i];
      for (int v=0; v < nfv; v++) if (fidx[v] >= 0) { int fi = fidx[v];
        for (int c=0; c < 3; c++) xn[3*v+c] = x[3*v+c] + alpha*dx[3*fi+c]; }
      if (ipc_energy(m, d, nfv, ea, en, xn, xtil, fidx, mass, kE, h, r, ghat, kappa,
                     gv, ge, acon, nacon) <= E0 + 1e-4*alpha*gdx) break;
      alpha *= 0.5;
    }
    for (int i=0; i < 3*nfv; i++) x[i] = xn[i];
  }
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    int da = dofadr[v];
    for (int c=0; c < 3; c++) { mjtNum dp = x[3*v+c]-xold[3*v+c]; d->qvel[da+c] = dp/h; d->qpos[da+c] += dp; }
  }
  d->time += h;

  mju_free(dofadr); mju_free(fidx); mju_free(mass); mju_free(kE);
  mju_free(acon); mju_free(ccache); mju_free(cand);
  mju_free(gv); mju_free(ge); mju_free(he); mju_free(efa); mju_free(efb);
  mju_free(x); mju_free(xtil); mju_free(xold); mju_free(xn);
  mju_free(grad); mju_free(dx); mju_free(mdiag); mju_free(diagH);
  mju_free(rcg); mju_free(zcg); mju_free(pcg); mju_free(Hpv);
}
