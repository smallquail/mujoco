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
#include <stdlib.h>   // qsort (sparse Hessian pattern build)

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_forward.h"      // mj_Euler (fallback)
#include "engine/engine_util_blas.h"    // mju_dot3
#include "engine/engine_util_solve.h"   // mju_cholFactor, mju_cholSolve (joint null-space projector)
#include "engine/engine_util_errmem.h"  // mju_malloc, mju_free
#include "engine/engine_util_spatial.h" // mju_mulQuat, mju_mat2Rot, mju_rotVecQuat (affine bodies)
#include "engine/engine_core_smooth.h"  // mj_kinematics (affine predicted poses)
#include "engine/engine_support.h"      // mj_integratePos (affine predicted poses)

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

// closest point (out) on a convex polygon face to point p: project onto the face plane; if the
// projection is inside the polygon use it, else clamp to the nearest boundary edge. pv = the face's
// mesh-local vertex indices (nv of them) into the vertex array vbase; nrm = the face's plane normal.
static void ipc_closestOnPoly(const mjtNum* p, const float* vbase, const int* pv, int nv,
                              const mjtNum* nrm, mjtNum* out) {
  const float* a0 = vbase + 3*pv[0];
  mjtNum dpl = nrm[0]*(p[0]-a0[0]) + nrm[1]*(p[1]-a0[1]) + nrm[2]*(p[2]-a0[2]);
  mjtNum pp[3]; for (int k=0; k < 3; k++) pp[k] = p[k] - dpl*nrm[k];   // projection onto face plane
  int npos = 0, nneg = 0;
  for (int i=0; i < nv; i++) {
    const float* a = vbase+3*pv[i]; const float* b = vbase+3*pv[(i+1)%nv];
    mjtNum e[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, w[3]={pp[0]-a[0],pp[1]-a[1],pp[2]-a[2]};
    mjtNum cr = (e[1]*w[2]-e[2]*w[1])*nrm[0] + (e[2]*w[0]-e[0]*w[2])*nrm[1] + (e[0]*w[1]-e[1]*w[0])*nrm[2];
    if (cr > 0) npos++; else nneg++;
  }
  if (npos == 0 || nneg == 0) { for (int k=0; k < 3; k++) out[k] = pp[k]; return; }  // inside polygon
  mjtNum best = 1e30;                                                                // clamp to edges
  for (int i=0; i < nv; i++) {
    const float* a = vbase+3*pv[i]; const float* b = vbase+3*pv[(i+1)%nv];
    mjtNum e[3]={b[0]-a[0],b[1]-a[1],b[2]-a[2]}, w[3]={p[0]-a[0],p[1]-a[1],p[2]-a[2]};
    mjtNum t = (e[0]*w[0]+e[1]*w[1]+e[2]*w[2]) / (e[0]*e[0]+e[1]*e[1]+e[2]*e[2]+1e-18);
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    mjtNum c[3]={a[0]+t*e[0], a[1]+t*e[1], a[2]+t*e[2]};
    mjtNum d2=(p[0]-c[0])*(p[0]-c[0])+(p[1]-c[1])*(p[1]-c[1])+(p[2]-c[2])*(p[2]-c[2]);
    if (d2 < best) { best = d2; for (int k=0; k < 3; k++) out[k]=c[k]; }
  }
}

// signed distance from a static geom's surface to world point x (positive outside) + outward unit
// normal n. Closed-form for plane/sphere/capsule/box; returns +large (no contact) for other types.
static mjtNum ipc_geomDist(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat,
                           const mjtNum* x, mjtNum* n, mjtNum cutoff) {
  int type = m->geom_type[gi]; const mjtNum* size = m->geom_size + 3*gi;
  mjtNum dx[3]; for (int k=0; k < 3; k++) dx[k] = x[k]-gpos[k];
  if (type == mjGEOM_MESH) {
    // point vs CONVEX-hull mesh, using MuJoCo's precomputed convex polygon faces. First the max
    // signed face-plane distance (cheap; gives the inside test and a far/cheap reject). If the point
    // is INSIDE (maxd<=0) return maxd with the max-face normal. If clearly FAR (maxd large) return
    // maxd (barrier is off there, normal direction irrelevant). Only NEAR the surface do the proper
    // point-to-convex-hull CLOSEST POINT (min over faces of the clamped projection) -> correct
    // distance AND a normal that points at the actual nearest surface (no max-plane normal chatter).
    int meshid = m->geom_dataid[gi];
    const float* vbase = m->mesh_vert + 3*m->mesh_vertadr[meshid];
    int polyadr = m->mesh_polyadr[meshid], pn = m->mesh_polynum[meshid];
    if (pn <= 0) { n[0]=0; n[1]=0; n[2]=1; return 1e30; }
    mjtNum pl[3]; mju_mulMatTVec3(pl, gmat, dx);   // world -> mesh-local
    mjtNum maxd = -1e30; const mjtNum* bestn = m->mesh_polynormal + 3*polyadr;
    for (int p=0; p < pn; p++) {
      const mjtNum* pnl = m->mesh_polynormal + 3*(polyadr+p);
      const float* v0 = vbase + 3*m->mesh_polyvert[m->mesh_polyvertadr[polyadr+p]];
      mjtNum c = pnl[0]*v0[0]+pnl[1]*v0[1]+pnl[2]*v0[2];
      mjtNum dd = pnl[0]*pl[0]+pnl[1]*pl[1]+pnl[2]*pl[2] - c;
      if (dd > maxd) { maxd = dd; bestn = pnl; }
    }
    if (maxd <= 0) { mju_mulMatVec3(n, gmat, bestn); return maxd; }   // inside: penetration
    // far reject: maxd (max signed plane distance) is a LOWER bound on the true distance for a convex
    // hull, so if it already exceeds the cutoff the closest-point search can't bring it into range --
    // skip the O(faces) loop (this is what makes the per-vertex broadphase against a mesh affordable).
    if (maxd > cutoff) { mju_mulMatVec3(n, gmat, bestn); return maxd; }
    mjtNum best = 1e30, bc[3] = {0,0,0};                              // outside: true closest point
    for (int p=0; p < pn; p++) {
      const mjtNum* pnl = m->mesh_polynormal + 3*(polyadr+p);
      const int* pv = m->mesh_polyvert + m->mesh_polyvertadr[polyadr+p];
      int nv = m->mesh_polyvertnum[polyadr+p];
      mjtNum cc[3]; ipc_closestOnPoly(pl, vbase, pv, nv, pnl, cc);
      mjtNum d2 = (pl[0]-cc[0])*(pl[0]-cc[0]) + (pl[1]-cc[1])*(pl[1]-cc[1]) + (pl[2]-cc[2])*(pl[2]-cc[2]);
      if (d2 < best) { best = d2; bc[0]=cc[0]; bc[1]=cc[1]; bc[2]=cc[2]; }
    }
    mjtNum dist = sqrt(best);
    mjtNum nl[3]; for (int k=0; k < 3; k++) nl[k] = (dist > 1e-12) ? (pl[k]-bc[k])/dist : bestn[k];
    mju_mulMatVec3(n, gmat, nl);                   // outward normal (point - closest) -> world
    return dist;
  }
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

// world-space VERTICES of a static geom (sharp features that can poke through a flex triangle):
// box -> 8 corners; mesh -> all its vertices; smooth/infinite geoms none. Returns the count.
static int ipc_geomVerts(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat,
                         mjtNum* out) {
  int type = m->geom_type[gi];
  if (type == mjGEOM_BOX) {
    const mjtNum* size = m->geom_size + 3*gi;
    int n = 0;
    for (int sx=-1; sx <= 1; sx+=2) for (int sy=-1; sy <= 1; sy+=2) for (int sz=-1; sz <= 1; sz+=2) {
      mjtNum loc[3] = {sx*size[0], sy*size[1], sz*size[2]}, wc[3];
      mju_mulMatVec3(wc, gmat, loc);
      for (int k=0; k < 3; k++) out[3*n+k] = gpos[k] + wc[k];
      n++;
    }
    return n;
  }
  if (type == mjGEOM_MESH) {
    int mid = m->geom_dataid[gi], nv = m->mesh_vertnum[mid];
    const float* vb = m->mesh_vert + 3*m->mesh_vertadr[mid];
    for (int i=0; i < nv; i++) {
      mjtNum lv[3] = {vb[3*i], vb[3*i+1], vb[3*i+2]}, wv[3];
      mju_mulMatVec3(wv, gmat, lv);
      for (int k=0; k < 3; k++) out[3*i+k] = gpos[k] + wv[k];
    }
    return nv;
  }
  return 0;
}

// world-space EDGES of a static geom (a geom edge can slice through a flex triangle between flex
// vertices): box -> 12 edges; mesh -> its convex-polygon edges (deduped: each shared edge emitted
// once, by the polygon traversing it low->high index). Each edge = two endpoints. Returns count.
static int ipc_geomEdges(const mjModel* m, int gi, const mjtNum* gpos, const mjtNum* gmat,
                         mjtNum* out) {
  int type = m->geom_type[gi];
  if (type == mjGEOM_BOX) {
    const mjtNum* size = m->geom_size + 3*gi;
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
    return n;
  }
  if (type == mjGEOM_MESH) {
    int mid = m->geom_dataid[gi];
    const float* vb = m->mesh_vert + 3*m->mesh_vertadr[mid];
    int pa = m->mesh_polyadr[mid], pn = m->mesh_polynum[mid], n = 0;
    for (int p=0; p < pn; p++) {
      int adr = m->mesh_polyvertadr[pa+p], nvp = m->mesh_polyvertnum[pa+p];
      for (int j=0; j < nvp; j++) {
        int a = m->mesh_polyvert[adr+j], b = m->mesh_polyvert[adr+(j+1)%nvp];
        if (a >= b) continue;   // dedup: emit each hull edge once (the low->high traversal)
        mjtNum la[3]={vb[3*a],vb[3*a+1],vb[3*a+2]}, lb[3]={vb[3*b],vb[3*b+1],vb[3*b+2]}, wa[3], wb[3];
        mju_mulMatVec3(wa, gmat, la); mju_mulMatVec3(wb, gmat, lb);
        for (int k=0; k < 3; k++) { out[6*n+k] = gpos[k]+wa[k]; out[6*n+3+k] = gpos[k]+wb[k]; }
        n++;
      }
    }
    return n;
  }
  return 0;
}

// IPC log-barrier on a surface gap g (C-IPC offset) and its 1st/2nd derivatives, for 0 < g < gh
static mjtNum ipc_Bg (mjtNum g, mjtNum gh) { return (g > 0 && g < gh) ? -(g-gh)*(g-gh)*log(g/gh) : 0.0; }
static mjtNum ipc_Bd (mjtNum g, mjtNum gh) { mjtNum u=g-gh; return -2*u*log(g/gh) - u*u/g; }
static mjtNum ipc_Bdd(mjtNum g, mjtNum gh) { mjtNum u=g-gh; return -2*log(g/gh) - 4*u/g + u*u/(g*g); }

// affine-body contact barrier (per-kappa value + 1st/2nd derivative). STRICT offset log-barrier for g>0
// (force -> inf as g->0, so a loaded feature equilibrates at g>0 and the CCD never has to cap the step to
// zero -- no freeze), plus a QUADRATIC recovery for g<=0 (stiffness krec per kappa) that pushes a feature
// back out if the lossy qpos round-trip reconstructs a penetrating pose. The g=0 force jump is only ever
// hit by such a discrete round-trip jump, not by a smooth approach (the strict barrier prevents that).
static void ipc_barrierC(mjtNum g, mjtNum gh, mjtNum krec, mjtNum* v, mjtNum* d1, mjtNum* d2) {
  if (g >= gh) { *v = 0; *d1 = 0; *d2 = 0; return; }
  if (g > 0)   { *v = ipc_Bg(g, gh); *d1 = ipc_Bd(g, gh); *d2 = ipc_Bdd(g, gh); return; }
  *d2 = krec; *d1 = krec*g; *v = 0.5*krec*g*g;             // g<=0: quadratic push-out
}

// one active contact. type: 0 vertex-triangle self, 1 edge-edge self, 2 flex-vertex vs geom
// surface, 3 geom-corner vs flex-triangle, 4 geom-edge vs flex-edge. idx/gi meaning per type
// (see ipc_conGap). The geom side is static, so its features (gv/ge) are precomputed once per step.
typedef struct { int type; int idx[4]; int gi; } ipcCon;

// gap g of a contact at configuration x, plus the barrier gradient direction n, the involved flex
// vertices idv[*nidx] and their weights cw (dg/d(vertex_p) = cw[p]*n). gv/ge are the precomputed
// world-space static-geom corners/edges. Single source of the per-type contact geometry.
static mjtNum ipc_conGap(const ipcCon* con, const mjModel* m, const mjData* d, const mjtNum* x,
                         const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad,
                         mjtNum* n, int* idv, mjtNum* cw, int* nidx, mjtNum cutoff) {
  switch (con->type) {
  case 0: {   // vertex-triangle: flex self-contact, OR a rigid point (sphere) vs a flex triangle A,B,C
    int v=con->idx[0], A=con->idx[1], B=con->idx[2], C=con->idx[3];
    mjtNum cp[3], w[3], dd = ipc_ptTri(&x[3*v], &x[3*A], &x[3*B], &x[3*C], cp, w);
    for (int k=0; k < 3; k++) n[k] = (x[3*v+k]-cp[k])/dd;
    idv[0]=v; idv[1]=A; idv[2]=B; idv[3]=C;
    cw[0]=1; cw[1]=-w[0]; cw[2]=-w[1]; cw[3]=-w[2]; *nidx=4;
    return dd - (rad[v] + rad[A]);   // point radius + triangle (flex) radius; per-flex for cross-flex contact
  }
  case 1: {   // edge-edge contact (edge a1b1 against edge a2b2): flex self OR cross-flex
    int a1=con->idx[0], b1=con->idx[1], a2=con->idx[2], b2=con->idx[3];
    mjtNum cp1[3], cp2[3], st[2], dd = ipc_segSeg(&x[3*a1], &x[3*b1], &x[3*a2], &x[3*b2], cp1, cp2, st);
    for (int k=0; k < 3; k++) n[k] = (cp1[k]-cp2[k])/dd;
    idv[0]=a1; idv[1]=b1; idv[2]=a2; idv[3]=b2;
    cw[0]=1-st[0]; cw[1]=st[0]; cw[2]=-(1-st[1]); cw[3]=-st[1]; *nidx=4;
    return dd - (rad[a1] + rad[a2]);   // both edges' (flex) radii
  }
  case 2: {   // free point v (flex vertex or rigid sphere) vs static geom gi surface
    int v=con->idx[0], gi=con->gi;
    mjtNum dd = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, &x[3*v], n, cutoff + rad[v]);
    idv[0]=v; cw[0]=1; *nidx=1;
    return dd - rad[v];
  }
  case 3: {   // static geom corner gv[idx0] vs flex triangle A,B,C
    const mjtNum* corner = &gv[3*con->idx[0]];
    int A=con->idx[1], B=con->idx[2], C=con->idx[3];
    mjtNum cp[3], w[3], dd = ipc_ptTri(corner, &x[3*A], &x[3*B], &x[3*C], cp, w);
    for (int k=0; k < 3; k++) n[k] = (corner[k]-cp[k])/dd;
    idv[0]=A; idv[1]=B; idv[2]=C; cw[0]=-w[0]; cw[1]=-w[1]; cw[2]=-w[2]; *nidx=3;
    return dd - rad[A];   // flex triangle radius
  }
  case 5: {   // rigid point p1 vs rigid point p2 (sphere-sphere)
    int p1=con->idx[0], p2=con->idx[1];
    mjtNum dv[3], dd=0;
    for (int k=0; k < 3; k++) { dv[k] = x[3*p1+k]-x[3*p2+k]; dd += dv[k]*dv[k]; } dd = mju_sqrt(dd);
    for (int k=0; k < 3; k++) n[k] = dv[k]/dd;
    idv[0]=p1; idv[1]=p2; cw[0]=1; cw[1]=-1; *nidx=2;
    return dd - (rad[p1]+rad[p2]);
  }
  default: {  // case 4: static geom edge ge[idx0] vs flex edge a,b
    const mjtNum* eg = &ge[6*con->idx[0]];
    int a=con->idx[1], b=con->idx[2];
    mjtNum cp1[3], cp2[3], st[2], dd = ipc_segSeg(eg, eg+3, &x[3*a], &x[3*b], cp1, cp2, st);
    for (int k=0; k < 3; k++) n[k] = (cp1[k]-cp2[k])/dd;
    idv[0]=a; idv[1]=b; cw[0]=-(1-st[1]); cw[1]=-st[1]; *nidx=2;
    return dd - rad[a];   // flex edge radius
  }
  }
}

// the flex vertices a contact involves (geom features are fixed and excluded): fills v[*nv]
static void ipc_conVerts(const ipcCon* con, int* v, int* nv) {
  switch (con->type) {
  case 0: case 1: v[0]=con->idx[0]; v[1]=con->idx[1]; v[2]=con->idx[2]; v[3]=con->idx[3]; *nv=4; break;
  case 2: v[0]=con->idx[0]; *nv=1; break;
  case 3: v[0]=con->idx[1]; v[1]=con->idx[2]; v[2]=con->idx[3]; *nv=3; break;   // flex triangle A,B,C
  case 5: v[0]=con->idx[0]; v[1]=con->idx[1]; *nv=2; break;                     // sphere-sphere (both points)
  default: v[0]=con->idx[1]; v[1]=con->idx[2]; *nv=2; break;                    // flex edge a,b
  }
}

// per-contact barrier activation distance d_hat. Never exceeds the global ghat, but shrinks to the
// thinnest participating radius: a thin flex (e.g. a drawstring sitting in a thick bag's sleeve) then
// gets a proportionally thin barrier zone instead of resting deep inside the thick neighbour's zone
// (which keeps it permanently active + ratchets kappa -> ill-conditioned). Geom features carry no
// radius (excluded by ipc_conVerts), so geom/flex and sphere/flex contacts keep the global ghat.
static mjtNum ipc_conGhat(const ipcCon* con, const mjtNum* rad, mjtNum ghat) {
  int vv[4], nvv; ipc_conVerts(con, vv, &nvv);
  mjtNum g = ghat;
  for (int q = 0; q < nvv; q++) if (rad[vv[q]] < g) g = rad[vv[q]];
  return g;
}

// surface gap of a contact with its flex vertices advanced by t*dxw (geom features fixed); gap-only,
// for the CCD conservative advancement (recomputes the closest feature at the advanced configuration).
static mjtNum ipc_conGapAdv(const ipcCon* con, const mjModel* m, const mjData* d, const mjtNum* x,
                            const mjtNum* dxw, mjtNum t, const mjtNum* gv, const mjtNum* ge,
                            mjtNum r, const mjtNum* rad, const int* fidx) {
  int v[4], nv; mjtNum P[4][3];
  ipc_conVerts(con, v, &nv);
  for (int q=0; q < nv; q++) { int fq = fidx[v[q]];
    for (int k=0; k < 3; k++) P[q][k] = x[3*v[q]+k] + (fq >= 0 ? t*dxw[3*fq+k] : 0.0); }
  mjtNum cp[3], w[3], c1[3], c2[3], st[2];
  switch (con->type) {
  case 0:  return ipc_ptTri(P[0], P[1], P[2], P[3], cp, w) - (rad[con->idx[0]] + rad[con->idx[1]]);
  case 1:  return ipc_segSeg(P[0], P[1], P[2], P[3], c1, c2, st) - (rad[con->idx[0]] + rad[con->idx[2]]);
  case 2:  { mjtNum nn[3];
             return ipc_geomDist(m, con->gi, d->geom_xpos+3*con->gi, d->geom_xmat+9*con->gi, P[0], nn, 1e30) - rad[con->idx[0]]; }
  case 3:  return ipc_ptTri(&gv[3*con->idx[0]], P[0], P[1], P[2], cp, w) - rad[con->idx[1]];
  case 5:  { mjtNum dd=0; for (int k=0; k < 3; k++){ mjtNum t=P[0][k]-P[1][k]; dd+=t*t; }
             return mju_sqrt(dd) - (rad[con->idx[0]]+rad[con->idx[1]]); }
  default: { const mjtNum* eg = &ge[6*con->idx[0]]; return ipc_segSeg(eg, eg+3, P[0], P[1], c1, c2, st) - rad[con->idx[1]]; }
  }
}

// rigorous additive CCD (Li et al.): largest alpha in [0,1] s.t. advancing x by alpha*dxw keeps every
// candidate's surface gap above 20% of its current value -- conservative advancement, no tunneling.
// For self pairs the common (mean) displacement is removed so coherent motion (free fall) isn't
// throttled; geom features are fixed so only the flex side's speed bounds the gap-shrink rate.
// IMPORTANT: mean-removal is only valid when the pair can genuinely move together. A sphere-vs-flex
// contact (type 0 with the "vertex" a rigid point, idx[0]>=nfv) is NOT such a pair -- the flex side may
// be blocked (e.g. the bag bottom pinned against the bin floor) so it cannot follow the sphere. Removing
// the mean there underestimates the gap-shrink rate and lets the sphere punch through, so it is excluded.
static mjtNum ipc_ccd(const mjModel* m, const mjData* d, const mjtNum* x, const mjtNum* dxw,
                      const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad, int nfv,
                      const int* fidx, const ipcCon* cand, int ncand, const mjtNum* cgap) {
  mjtNum alpha = 1.0;
  for (int c=0; c < ncand; c++) {
    const ipcCon* con = &cand[c];
    int v[4], nv, self = (con->type <= 1) && (con->idx[0] < nfv);   // genuine flex-flex self-contact only
    ipc_conVerts(con, v, &nv);
    mjtNum dp[4][3], mean[3] = {0,0,0};
    for (int q=0; q < nv; q++) { int fq = fidx[v[q]];
      for (int k=0; k < 3; k++) dp[q][k] = (fq >= 0 ? dxw[3*fq+k] : 0.0); }
    if (self) {
      for (int q=0; q < nv; q++) for (int k=0; k < 3; k++) mean[k] += dp[q][k];
      for (int k=0; k < 3; k++) mean[k] /= nv;
      for (int q=0; q < nv; q++) for (int k=0; k < 3; k++) dp[q][k] -= mean[k];
    }
    mjtNum l;   // bound on the gap-shrink rate per unit alpha
    if (con->type == 0) { mjtNum a0=sqrt(mju_dot3(dp[0],dp[0])), b0=0;
      for (int q=1;q<4;q++){ mjtNum s=sqrt(mju_dot3(dp[q],dp[q])); if (s>b0) b0=s; } l=a0+b0; }
    else if (con->type == 1) { mjtNum a0=0,b0=0;
      for (int q=0;q<2;q++){ mjtNum s=sqrt(mju_dot3(dp[q],dp[q])); if (s>a0) a0=s; }
      for (int q=2;q<4;q++){ mjtNum s=sqrt(mju_dot3(dp[q],dp[q])); if (s>b0) b0=s; } l=a0+b0; }
    else if (con->type == 5) { l = sqrt(mju_dot3(dp[0],dp[0])) + sqrt(mju_dot3(dp[1],dp[1])); }  // both move
    else { l=0; for (int q=0;q<nv;q++){ mjtNum s=sqrt(mju_dot3(dp[q],dp[q])); if (s>l) l=s; } }
    if (l < 1e-12) continue;
    mjtNum g0 = cgap[c];   // gap at x, cached by ipc_try this Newton iter (== ipc_conGapAdv at t=0)
    if (g0 <= 0) continue;   // already touching/penetrating: the barrier + energy own it
    if (l <= 0.8*g0) continue;   // full alpha=1 step shrinks gap by <= l, stays above the 20% floor
    mjtNum gtarget = 0.2*g0, t = 0;
    for (int it=0; it < 32; it++) {
      mjtNum g = ipc_conGapAdv(con, m, d, x, dxw, t, gv, ge, r, rad, fidx);
      mjtNum room = g - gtarget;
      if (room <= 1e-9*g0) break;
      t += room / l;
      if (t >= alpha) { t = alpha; break; }
    }
    if (t < alpha) alpha = t;
  }
  return alpha;
}

// cached per-contact data for the matrix-free Hessian apply: GN block = bdd * (cw[p]*n)(cw[q]*n)^T
// over the involved free-dof indices f[0..nidx) (f<0 = pinned, skipped).
typedef struct { mjtNum n[3], cw[8], bdd; int f[8], nidx; } ipcCC;   // up to 8 cps (affine-affine: 4 per body)

// One membrane element (2D triangle): the 3 flex vertices (global vg, free-dof fv, -1 if pinned),
// the FEM stiffness metric M (symmetric 3x3 over the element's 3 edges, read from flex_stiffness),
// and the rest squared edge lengths Lr2. We consume MuJoCo's P1 FEM membrane energy (engine_passive,
// Kharevych et al.): E = 1/4 e^T M e with e[edge] = L^2 - L0^2 (frame-invariant). The force is the
// gradient (matches mj_flexPassiveStretch exactly) and the Hessian is its Gauss-Newton part.
typedef struct { int vg[3], fv[3]; mjtNum M[6]; mjtNum Lr2[3];
                 mjtNum kD;        // Rayleigh damping coefficient = flex_damping/h (constant)
                 mjtNum ep[3];     // elongations e_a at the previous step (xold), refreshed each step
               } ipcElem;
static const int ipc_eedge[3][2] = {{1, 2}, {2, 0}, {0, 1}};   // local edges (== engine_passive's)
static inline mjtNum ipc_Mab(const mjtNum* M6, int a, int b) {
  static const int id[3][3] = {{0,1,2},{1,3,4},{2,4,5}};   // 3x3 symmetric from 6 upper-tri entries
  return M6[id[a][b]];
}
// edge vectors d[a] = x[v_a0]-x[v_a1] and elongations e[a] = |d[a]|^2 - Lr2[a] for an element at x
static void ipc_elemEval(const ipcElem* el, const mjtNum* x, mjtNum d[3][3], mjtNum e[3]) {
  for (int a=0; a < 3; a++) {
    int p = el->vg[ipc_eedge[a][0]], q = el->vg[ipc_eedge[a][1]];
    mjtNum L2 = 0;
    for (int k=0; k < 3; k++) { d[a][k] = x[3*p+k]-x[3*q+k]; L2 += d[a][k]*d[a][k]; }
    e[a] = L2 - el->Lr2[a];
  }
}


// evaluate a candidate contact at x; if active (0<g<ghat) accumulate its barrier gradient and the
// Hessian diagonal (for the Jacobi preconditioner), and cache the GN block (ccache) + contact (acon).
static void ipc_try(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x,
                    const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad, mjtNum ghat, mjtNum kappa,
                    const int* fidx, mjtNum* grad,
                    ipcCon* acon, ipcCC* ccache, int* nacon, int amax, mjtNum* gout) {
  mjtNum n[3], cw[4]; int idv[4], nidx;
  mjtNum ghc = ipc_conGhat(&con, rad, ghat);   // per-contact barrier width (thin for thin participants)
  mjtNum g = ipc_conGap(&con, m, d, x, gv, ge, r, rad, n, idv, cw, &nidx, ghat);   // detection cutoff stays global
  *gout = g;   // cache the gap at x so CCD (g0) and the E0 energy can reuse it without recomputing
  if (g <= 0 || g >= ghc) return;
  if (*nacon >= amax) return;
  mjtNum bd = kappa*ipc_Bd(g, ghc), bdd = kappa*ipc_Bdd(g, ghc);
  ipcCC* cc = &ccache[*nacon];
  cc->bdd = bdd; cc->nidx = nidx;
  for (int k=0; k < 3; k++) cc->n[k] = n[k];
  for (int p=0; p < nidx; p++) {
    int fp = fidx[idv[p]];
    cc->cw[p] = cw[p]; cc->f[p] = fp;
    if (fp < 0) continue;
    for (int i=0; i < 3; i++) grad[3*fp+i] += bd*cw[p]*n[i];   // barrier gradient (force into grad)
  }
  acon[*nacon] = con;                                          // GN Hessian block (bdd, n, cw) cached
  (*nacon)++;
}

// matrix-free Hessian-vector product Hp = H*p in the free-dof space (size N): inertia (mdiag) + the
// full FEM membrane element Hessian + contact Gauss-Newton blocks (ccache). The element Hessian is
// the FULL analytic second derivative of E = 1/4 e^T M e:
//   H = 2 sum_ab M[a,b] g_a g_b^T          (the "Gauss-Newton" / material part, g_a = edge vector)
//     + sum_a Me_a (edge-Laplacian_a (x) I) (the geometric/stress part, Me_a = sum_b M[a,b] e_b)
// applied in factored form per element (no projection): estr caches g_a (9) and Me_a (3) this Newton
// iter. The inertia term mdiag dominates so the total H is normally positive-definite for CG.
// 3x3 inverse (row-major); returns 0 columns if singular
static void ipc_mat3inv(mjtNum* inv, const mjtNum* m) {
  mjtNum det = m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6]);
  mjtNum id = (mju_abs(det) > mjMINVAL) ? 1.0/det : 0;
  inv[0]=(m[4]*m[8]-m[5]*m[7])*id; inv[1]=(m[2]*m[7]-m[1]*m[8])*id; inv[2]=(m[1]*m[5]-m[2]*m[4])*id;
  inv[3]=(m[5]*m[6]-m[3]*m[8])*id; inv[4]=(m[0]*m[8]-m[2]*m[6])*id; inv[5]=(m[2]*m[3]-m[0]*m[5])*id;
  inv[6]=(m[3]*m[7]-m[4]*m[6])*id; inv[7]=(m[1]*m[6]-m[0]*m[7])*id; inv[8]=(m[0]*m[4]-m[1]*m[3])*id;
}

// 4x4 inverse (row-major) by Gauss-Jordan with partial pivoting
static void ipc_mat4inv(mjtNum* inv, const mjtNum* min) {
  mjtNum a[4][8];
  for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) { a[i][j] = min[4*i+j]; a[i][4+j] = (i==j) ? 1.0 : 0.0; }
  for (int col=0; col < 4; col++) {
    int piv = col;
    for (int r=col+1; r < 4; r++) if (mju_abs(a[r][col]) > mju_abs(a[piv][col])) piv = r;
    if (piv != col) for (int j=0; j < 8; j++) { mjtNum t = a[col][j]; a[col][j] = a[piv][j]; a[piv][j] = t; }
    mjtNum dv = a[col][col]; if (mju_abs(dv) < mjMINVAL) dv = (dv < 0 ? -mjMINVAL : mjMINVAL);
    for (int j=0; j < 8; j++) a[col][j] /= dv;
    for (int r=0; r < 4; r++) if (r != col) { mjtNum f = a[r][col]; for (int j=0; j < 8; j++) a[r][j] -= f*a[col][j]; }
  }
  for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) inv[4*i+j] = a[i][4+j];
}

// ----------------------------------- affine (ABD) rigid bodies ------------------------------------
// A rigid body is stepped via 4 control points forming a (general) tetra fixed in the body frame (rest
// positions Xr). The control points are free points in the solver; the rigid pose maps to them by FK
// (cp = pos + R*Xr) and is recovered by fitting the affine frame F = D_curr*Dinv and extracting its
// rotation (ipc_affineCP/ipc_affineReadback). Rigidity = orthogonality energy + the affine inertia block
// M_cp4 (= B^-T M_aff B^-1, tiled as M_cp4[i][j]*I3); free motion is carried exactly by the rigid predictor.
// A free body uses a radii-of-gyration tetra (well-scaled); jointed bodies place control points at the
// joint anchors (Stage 1).
typedef struct {
  int body, qadr, vadr, ja;  // mjModel body id; qpos/dof/joint addresses (free: pos+quat/lin+ang; hinge: angle/dof)
  int isHinge;              // 0 = free-joint body; 1 = hinge joint (1-3 hinges sharing an anchor)
  int jstatic;              // hinge to a STATIC parent (axis points pinned to world); else coupled to jparent
  int jparent;              // affine index of the parent hinge body (-1 if static-hinge or free)
  int njnt, njpt;           // # hinges on this body (1-3); # joint-constraint points (<=2)
  mjtNum jaxis[3][3], hpt[3]; // the njnt hinge world axes (readback projection); shared anchor (world, setup)
  int cp[4];                // the 4 control points' indices into the free-point array (this body owns all 4)
  int sdof[4];              // solver-dof base (3*fidx) of each control point
  mjtNum Xr[4][3];          // rest positions of the 4 control points in the body frame (radii-of-gyration tetra)
  mjtNum Dinv[9];           // ([Xr1-Xr0, Xr2-Xr0, Xr3-Xr0])^-1 -- maps current edges to F (orthogonality)
  mjtNum Binv[16];          // inverse of [1, Xr_i^T] rows: world point -> affine weights in this body's tetra
  mjtNum Mcp4[4][4];        // affine inertia in control-point coords (each entry scales an I3 block)
  mjtNum kappaO;            // orthogonality (rigidity) stiffness
  mjtNum jw[2][4];          // joint: njpt constraint points as affine combos of THIS body's control points
  mjtNum jwp[2][4];         // hinge child: same njpt points as affine combos of the PARENT's control points
  mjtNum janch[2][3];       // hinge-to-static: the njpt points' fixed world target positions
  int pfull[2];             // per point: 1 = full coincidence (3 axis-aligned rows); 0 = single row along pn
  mjtNum pn[2][3];          // per point: the single-row normal (the locked 3rd-axis direction, k=2)
  mjtNum ghatC, cdampH;     // contact barrier support distance; normal-dashpot Hessian coefficient (c/h)
  int nfeat;                // number of collision features (box corners / capsule ends / sphere centers)
  mjtNum cfeat[24][4];      // each feature's affine weights in the control-point tetra (constant)
  mjtNum cfrad[24];         // each feature's radius offset (0 for a box corner, geom radius for capsule/sphere)
  int ncap, caps[12][2], capgeom[12];   // capsules (for affine-affine self-collision): the 2 end-feature indices; geom id
  // ---- reduced coordinates (genuine "don't add the DOFs"): the body's 12 control-point DOFs are NOT solver
  // unknowns; the solver works in the nfree FREE DOFs z. cp-increment = Mb*(parent cp-increment) + Nb*dz.
  int nfree, zadr;          // # free DOFs after this body's joint; offset into the reduced vector z
  mjtNum Mb[144], Nb[144];  // Mb: parent cp-DOFs -> this body's constrained part (12x12); Nb: free basis (12 x nfree)
} ipcAffine;

// affine body inertia (M_cp star block) + orthogonality (rigidity) energy over the 4 control points.
// Returns the energy and, optionally (g/H non-NULL), the 12-gradient and 12x12 Gauss-Newton (PSD) Hessian
// in control-point-local order (point k, coord c -> 3k+c). F is linear in the control points so the energy
// is polynomial -- no polar decomposition in the solve. M_cp is the "star" form (cp0 = com): translation
// mode mass = body mass; off-diagonals couple cp0 to each axis point.
static mjtNum ipc_affineGH(const ipcAffine* af, const mjtNum* x, const mjtNum* xtil, mjtNum ih2,
                           mjtNum g[12], mjtNum H[12][12]) {
  mjtNum y[4][3], yt[4][3];
  for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) {
    y[k][c] = x[3*af->cp[k]+c]; yt[k][c] = xtil[3*af->cp[k]+c];
  }
  if (g) for (int i=0; i < 12; i++) { g[i] = 0; if (H) for (int j=0; j < 12; j++) H[i][j] = 0; }
  mjtNum E = 0;
  for (int a=0; a < 4; a++) for (int b=0; b < 4; b++) {
    mjtNum s = af->Mcp4[a][b]*ih2;
    for (int c=0; c < 3; c++) {
      E += 0.5*s*(y[a][c]-yt[a][c])*(y[b][c]-yt[b][c]);
      if (g) g[3*a+c] += s*(y[b][c]-yt[b][c]);
      if (H) H[3*a+c][3*b+c] += s;
    }
  }
  // orthogonality: F[c][col] = sum_m (y_{m+1}[c]-y0[c]) * Dinv[m][col]; C = F^T F - I (6 indep);
  // V = kappaO * sum_comp w_comp C^2. dF[c][col]/dy_{m+1}[c] = Dinv[m][col], dF[c][col]/dy0[c] = -sum_m Dinv[m][col]
  mjtNum F[3][3];
  for (int c=0; c < 3; c++) for (int col=0; col < 3; col++) {
    mjtNum s = 0; for (int mm=0; mm < 3; mm++) s += (y[mm+1][c]-y[0][c])*af->Dinv[3*mm+col];
    F[c][col] = s;
  }
  const int ca[6] = {0,1,2,0,0,1}, cb[6] = {0,1,2,1,2,2}; const mjtNum w[6] = {1,1,1,2,2,2};
  mjtNum C[6], Jc[6][12];
  for (int q=0; q < 6; q++) {
    int a = ca[q], b = cb[q];
    mjtNum dot = 0; for (int c=0; c < 3; c++) dot += F[c][a]*F[c][b];
    C[q] = dot - (a == b ? 1.0 : 0.0);
    mjtNum csa = af->Dinv[a] + af->Dinv[3+a] + af->Dinv[6+a];     // sum_m Dinv[m][a]
    mjtNum csb = af->Dinv[b] + af->Dinv[3+b] + af->Dinv[6+b];     // sum_m Dinv[m][b]
    for (int i=0; i < 12; i++) Jc[q][i] = 0;
    for (int c=0; c < 3; c++) {                                   // dC_ab/dcp
      for (int mm=0; mm < 3; mm++)
        Jc[q][3*(mm+1)+c] += af->Dinv[3*mm+a]*F[c][b] + af->Dinv[3*mm+b]*F[c][a];
      Jc[q][3*0+c] -= csa*F[c][b] + csb*F[c][a];
    }
  }
  for (int q=0; q < 6; q++) {
    E += af->kappaO*w[q]*C[q]*C[q];
    if (g) {
      mjtNum gc = 2.0*af->kappaO*w[q]*C[q], hc = 2.0*af->kappaO*w[q];
      for (int i=0; i < 12; i++) {
        g[i] += gc*Jc[q][i];
        if (H) for (int j=0; j < 12; j++) H[i][j] += hc*Jc[q][i]*Jc[q][j];
      }
    }
  }
  return E;
}

// a geom is a STATIC collision obstacle only if its body is rigidly ANCHORED to the world: 0-dof AND no
// moving ancestor. A 0-dof body welded to a moving/affine body (the humanoid's hands, head) moves WITH
// that body and is lumped into its inertia -- treating it as a static geom makes the affine body collide
// with its own welded part (the forearm vs its attached hand -> a huge spurious barrier force, off-target).
static int ipc_bodyAnchored(const mjModel* m, int b) {
  while (b > 0) { if (m->body_dofnum[b] > 0) return 0; b = m->body_parentid[b]; }
  return 1;
}

// affine rigid-body contacts vs static geoms (anchored to world): each collision feature (box corner, capsule
// end, sphere center) is an affine combination of the 4 control points (world pos = sum_k cw_k * cp_k), so
// the barrier gradient distributes to the control points via cw and the Hessian reuses the existing ipcCC
// machinery. gap = signed-distance-to-geom - feature radius. Appends each active contact to ccache.
static void ipc_affineContact(const mjModel* m, const mjData* d, const ipcAffine* aff, int naff,
                              const mjtNum* x, const mjtNum* xold, mjtNum kappa, mjtNum* grad,
                              ipcCC* ccache, int* nacon, int amax, mjtNum* affHc) {
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a];
    mjtNum ghc = af->ghatC, cdH = af->cdampH;
    for (int ft=0; ft < af->nfeat; ft++) {
      mjtNum pw[3] = {0,0,0}, dpw[3] = {0,0,0};   // feature world pos and its displacement this step
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) {
        pw[c] += af->cfeat[ft][k]*x[3*af->cp[k]+c];
        dpw[c] += af->cfeat[ft][k]*(x[3*af->cp[k]+c] - xold[3*af->cp[k]+c]);
      }
      for (int gi=0; gi < m->ngeom; gi++) {
        if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;
        if (!ipc_bodyAnchored(m, m->geom_bodyid[gi])) continue;            // static geoms only
        mjtNum n[3], g = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, pw, n, ghc+af->cfrad[ft])
                         - af->cfrad[ft];
        if (g >= ghc || *nacon >= amax) continue;             // (g<=0 is NOT skipped: clamp recovers it)
        // clamped barrier (force, recovering through g<=0) + one-sided normal dashpot (dissipation)
        mjtNum bv, bd1, bd2; ipc_barrierC(g, ghc, 100.0, &bv, &bd1, &bd2);
        mjtNum dn = dpw[0]*n[0] + dpw[1]*n[1] + dpw[2]*n[2];
        mjtNum cde = (dn < 0) ? cdH : 0;                       // ONE-SIDED dashpot: damp approach (compression)
        mjtNum bd = kappa*bd1, bdd = kappa*bd2 + cde;          // only, NOT separation (else glue)
        for (int k=0; k < 4; k++) {
          if (af->sdof[k] < 0) continue;                       // pinned control point: no solver DOF
          for (int c=0; c < 3; c++) grad[af->sdof[k]+c] += (bd + cde*dn)*af->cfeat[ft][k]*n[c];
        }
        if (affHc) {                                           // per-body GN contact Hessian for the reduced solve
          mjtNum u[12]; for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) u[3*k+c] = af->cfeat[ft][k]*n[c];
          mjtNum* Hc = affHc + 144*a;
          for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) Hc[12*i+j] += bdd*u[i]*u[j];
        }
        ipcCC* cc = &ccache[(*nacon)++];
        for (int c=0; c < 3; c++) cc->n[c] = n[c];
        for (int k=0; k < 4; k++) { cc->cw[k] = af->cfeat[ft][k]; cc->f[k] = (af->sdof[k] >= 0) ? af->sdof[k]/3 : -1; }
        cc->nidx = 4; cc->bdd = bdd;
      }
    }
  }
}

static inline mjtNum ipc_fmin(mjtNum a, mjtNum b);   // defined below; used by the affine-affine helpers

// ===== affine-AFFINE self-collision (capsule-capsule): two MOVING affine bodies =======================
// each capsule is a segment whose two ends are affine combos of a body's control points; ipc_segSeg gives
// the closest-point params (s,t). The barrier gradient/Hessian distribute to BOTH bodies' control points.

// MuJoCo's collision filter for an affine-affine geom pair: geom bitmask + weld self/parent + <exclude>.
static int ipc_affineAACanCollide(const mjModel* m, int bA, int gA, int bB, int gB) {
  if (!((m->geom_contype[gA] & m->geom_conaffinity[gB]) || (m->geom_contype[gB] & m->geom_conaffinity[gA]))) return 0;
  int w1 = m->body_weldid[bA], w2 = m->body_weldid[bB];
  if (w1 == w2) return 0;                                          // same weld
  if (!(m->opt.disableflags & mjDSBL_FILTERPARENT)) {             // weld-parent filter (default ON)
    if (w1 == m->body_weldid[m->body_parentid[w2]] || w2 == m->body_weldid[m->body_parentid[w1]]) return 0;
  }
  for (int i=0; i < m->nexclude; i++)                             // <contact><exclude>
    if (m->exclude_signature[i] == ((bA<<16)+bB) || m->exclude_signature[i] == ((bB<<16)+bA)) return 0;
  return 1;
}

// gap + closest-point data for one capsule pair (capA of body a, capB of body b). Fills the per-cp weights
// cwA/cwB (dg/d cp = cw*n), the normal n, the two cp world positions, and the segment params. Returns gap.
static mjtNum ipc_affineAAGap(const ipcAffine* afA, int capA, const ipcAffine* afB, int capB,
                              const mjtNum* x, mjtNum* n, mjtNum* cwA, mjtNum* cwB) {
  const mjtNum *wA0 = afA->cfeat[afA->caps[capA][0]], *wA1 = afA->cfeat[afA->caps[capA][1]];
  const mjtNum *wB0 = afB->cfeat[afB->caps[capB][0]], *wB1 = afB->cfeat[afB->caps[capB][1]];
  mjtNum PA0[3]={0,0,0}, PA1[3]={0,0,0}, PB0[3]={0,0,0}, PB1[3]={0,0,0};
  for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) {
    PA0[c] += wA0[k]*x[3*afA->cp[k]+c]; PA1[c] += wA1[k]*x[3*afA->cp[k]+c];
    PB0[c] += wB0[k]*x[3*afB->cp[k]+c]; PB1[c] += wB1[k]*x[3*afB->cp[k]+c]; }
  mjtNum cp1[3], cp2[3], st[2], dd = ipc_segSeg(PA0, PA1, PB0, PB1, cp1, cp2, st);
  mjtNum radsum = afA->cfrad[afA->caps[capA][0]] + afB->cfrad[afB->caps[capB][0]];
  if (dd < 1e-9) dd = 1e-9;
  for (int c=0; c < 3; c++) n[c] = (cp1[c]-cp2[c])/dd;            // outward for A
  mjtNum s = st[0], t = st[1];
  for (int k=0; k < 4; k++) { cwA[k] = (1-s)*wA0[k] + s*wA1[k]; cwB[k] = -((1-t)*wB0[k] + t*wB1[k]); }
  return dd - radsum;
}

// affine-affine barrier: gradient -> grad (both bodies), GN block -> ccache (8 cps), per-body diagonal -> affHc.
static void ipc_affineAffineContact(const mjModel* m, const ipcAffine* aff, int naff, const mjtNum* x,
                                    const mjtNum* xold, mjtNum kappa, mjtNum* grad, ipcCC* ccache, int* nacon,
                                    int amax, mjtNum* affHc) {
  for (int a=0; a < naff; a++) for (int b=a+1; b < naff; b++) {
    const ipcAffine *afA = &aff[a], *afB = &aff[b];
    for (int ca=0; ca < afA->ncap; ca++) for (int cb=0; cb < afB->ncap; cb++) {
      if (!ipc_affineAACanCollide(m, afA->body, afA->capgeom[ca], afB->body, afB->capgeom[cb])) continue;
      mjtNum n[3], cwA[4], cwB[4];
      mjtNum ghc = ipc_fmin(afA->ghatC, afB->ghatC), g = ipc_affineAAGap(afA, ca, afB, cb, x, n, cwA, cwB);
      if (g >= ghc || *nacon >= amax) continue;
      mjtNum bv, bd1, bd2; ipc_barrierC(g, ghc, 100.0, &bv, &bd1, &bd2);
      mjtNum dn = 0;                                              // relative approach speed along n
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++)
        dn += (cwA[k]*(x[3*afA->cp[k]+c]-xold[3*afA->cp[k]+c]) + cwB[k]*(x[3*afB->cp[k]+c]-xold[3*afB->cp[k]+c]))*n[c];
      mjtNum cde = (dn < 0) ? ipc_fmin(afA->cdampH, afB->cdampH) : 0;
      mjtNum bd = kappa*bd1, bdd = kappa*bd2 + cde;
      for (int k=0; k < 4; k++) {                                 // gradient to both bodies
        if (afA->sdof[k] >= 0) for (int c=0; c < 3; c++) grad[afA->sdof[k]+c] += (bd + cde*dn)*cwA[k]*n[c];
        if (afB->sdof[k] >= 0) for (int c=0; c < 3; c++) grad[afB->sdof[k]+c] += (bd + cde*dn)*cwB[k]*n[c];
      }
      if (affHc) {                                               // per-body GN diagonal blocks (CRB)
        mjtNum uA[12], uB[12];
        for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) { uA[3*k+c]=cwA[k]*n[c]; uB[3*k+c]=cwB[k]*n[c]; }
        mjtNum* HA = affHc + 144*a; mjtNum* HB = affHc + 144*b;
        for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) { HA[12*i+j] += bdd*uA[i]*uA[j]; HB[12*i+j] += bdd*uB[i]*uB[j]; }
      }
      ipcCC* cc = &ccache[(*nacon)++];                           // full 8-cp block for the matrix-free PCG path
      for (int c=0; c < 3; c++) cc->n[c] = n[c];
      for (int k=0; k < 4; k++) { cc->cw[k]=cwA[k]; cc->f[k]=(afA->sdof[k]>=0)?afA->sdof[k]/3:-1;
                                  cc->cw[4+k]=cwB[k]; cc->f[4+k]=(afB->sdof[k]>=0)?afB->sdof[k]/3:-1; }
      cc->nidx = 8; cc->bdd = bdd;
    }
  }
}

// affine-affine contact energy (line search): barrier + one-sided dashpot.
static mjtNum ipc_affineAffineEnergy(const mjModel* m, const ipcAffine* aff, int naff,
                                     const mjtNum* x, const mjtNum* xold, mjtNum kappa) {
  mjtNum E = 0;
  for (int a=0; a < naff; a++) for (int b=a+1; b < naff; b++) {
    const ipcAffine *afA = &aff[a], *afB = &aff[b];
    for (int ca=0; ca < afA->ncap; ca++) for (int cb=0; cb < afB->ncap; cb++) {
      if (!ipc_affineAACanCollide(m, afA->body, afA->capgeom[ca], afB->body, afB->capgeom[cb])) continue;
      mjtNum n[3], cwA[4], cwB[4];
      mjtNum ghc = ipc_fmin(afA->ghatC, afB->ghatC), g = ipc_affineAAGap(afA, ca, afB, cb, x, n, cwA, cwB);
      if (g >= ghc) continue;
      mjtNum bv, bd1, bd2; ipc_barrierC(g, ghc, 100.0, &bv, &bd1, &bd2);
      mjtNum dn = 0;
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++)
        dn += (cwA[k]*(x[3*afA->cp[k]+c]-xold[3*afA->cp[k]+c]) + cwB[k]*(x[3*afB->cp[k]+c]-xold[3*afB->cp[k]+c]))*n[c];
      mjtNum cde = (dn < 0) ? ipc_fmin(afA->cdampH, afB->cdampH) : 0;
      E += kappa*bv + (dn < 0 ? 0.5*cde*dn*dn : 0);
    }
  }
  return E;
}

// conservative additive CCD for affine-affine pairs: cap alpha so no pair closes more than ~90% of its gap.
static mjtNum ipc_affineAffineCCD(const mjModel* m, const ipcAffine* aff, int naff, const mjtNum* x, const mjtNum* dx) {
  mjtNum cap = 1.0;
  for (int a=0; a < naff; a++) for (int b=a+1; b < naff; b++) {
    const ipcAffine *afA = &aff[a], *afB = &aff[b];
    for (int ca=0; ca < afA->ncap; ca++) for (int cb=0; cb < afB->ncap; cb++) {
      if (!ipc_affineAACanCollide(m, afA->body, afA->capgeom[ca], afB->body, afB->capgeom[cb])) continue;
      mjtNum n[3], cwA[4], cwB[4], g = ipc_affineAAGap(afA, ca, afB, cb, x, n, cwA, cwB);
      if (g <= 0) continue;                                       // already touching: clamp/barrier owns it
      // gap shrink rate <= max endpoint speed of A + of B (per-contact, affine-mapped -- NOT the flat dgap)
      mjtNum vmax = 0;
      for (int e=0; e < 2; e++) { const mjtNum* w = afA->cfeat[afA->caps[ca][e]]; mjtNum v[3]={0,0,0};
        for (int k=0; k < 4; k++) if (afA->sdof[k]>=0) for (int c=0; c<3; c++) v[c]+=w[k]*dx[afA->sdof[k]+c];
        mjtNum L=sqrt(mju_dot3(v,v)); if (L>vmax) vmax=L; }
      mjtNum vb = 0;
      for (int e=0; e < 2; e++) { const mjtNum* w = afB->cfeat[afB->caps[cb][e]]; mjtNum v[3]={0,0,0};
        for (int k=0; k < 4; k++) if (afB->sdof[k]>=0) for (int c=0; c<3; c++) v[c]+=w[k]*dx[afB->sdof[k]+c];
        mjtNum L=sqrt(mju_dot3(v,v)); if (L>vb) vb=L; }
      mjtNum l = vmax + vb;
      if (l > 1e-12) { mjtNum amx = 0.9*g/l; if (amx < cap) cap = amx; }
    }
  }
  return cap;
}

// total affine contact energy at x: barrier + normal-dashpot dissipation (line search / E0)
static mjtNum ipc_affineContactEnergy(const mjModel* m, const mjData* d, const ipcAffine* aff, int naff,
                                      const mjtNum* x, const mjtNum* xold, mjtNum kappa) {
  mjtNum E = 0;
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a];
    mjtNum ghc = af->ghatC, cdH = af->cdampH;
    for (int ft=0; ft < af->nfeat; ft++) {
      mjtNum pw[3] = {0,0,0}, dpw[3] = {0,0,0};
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) {
        pw[c] += af->cfeat[ft][k]*x[3*af->cp[k]+c];
        dpw[c] += af->cfeat[ft][k]*(x[3*af->cp[k]+c] - xold[3*af->cp[k]+c]);
      }
      for (int gi=0; gi < m->ngeom; gi++) {
        if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;
        if (!ipc_bodyAnchored(m, m->geom_bodyid[gi])) continue;
        mjtNum n[3], g = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, pw, n, ghc+af->cfrad[ft])
                         - af->cfrad[ft];
        if (g < ghc) {
          mjtNum bv, bd1, bd2; ipc_barrierC(g, ghc, 100.0, &bv, &bd1, &bd2);
          mjtNum dn = dpw[0]*n[0] + dpw[1]*n[1] + dpw[2]*n[2];
          E += kappa*bv + (dn < 0 ? 0.5*cdH*dn*dn : 0);              // clamped barrier + one-sided dashpot
        }
      }
    }
  }
  return E;
}

// CCD step cap for affine contact features (linear: exact for planes, conservative via the current normal)
static mjtNum ipc_affineContactCCD(const mjModel* m, const mjData* d, const ipcAffine* aff, int naff,
                                   const mjtNum* x, const mjtNum* dx) {
  mjtNum cap = 1.0;
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a];
    for (int ft=0; ft < af->nfeat; ft++) {
      mjtNum pw[3] = {0,0,0}, dw[3] = {0,0,0};
      for (int k=0; k < 4; k++) {
        for (int c=0; c < 3; c++) pw[c] += af->cfeat[ft][k]*x[3*af->cp[k]+c];
        if (af->sdof[k] >= 0) for (int c=0; c < 3; c++) dw[c] += af->cfeat[ft][k]*dx[af->sdof[k]+c];  // pinned: dx=0
      }
      for (int gi=0; gi < m->ngeom; gi++) {
        if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;
        if (!ipc_bodyAnchored(m, m->geom_bodyid[gi])) continue;
        mjtNum n[3], g = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, pw, n, 1e30) - af->cfrad[ft];
        if (g <= 0) continue;     // already penetrating: the recovery push (not the CCD) owns it
        mjtNum rate = dw[0]*n[0] + dw[1]*n[1] + dw[2]*n[2];       // d(gap)/d(alpha)
        if (rate < 0) { mjtNum amx = 0.9*g/(-rate); if (amx < cap) cap = amx; }
      }
    }
  }
  return cap;
}

static void ipc_applyH(const mjtNum* p, mjtNum* Hp, int N, const mjtNum* mdiag,
                       int nelem, const ipcElem* elems, const mjtNum* estr,
                       const ipcCC* ccache, int nacon,
                       int naff, const ipcAffine* aff, const mjtNum* affH) {
  for (int i=0; i < N; i++) Hp[i] = mdiag[i]*p[i];
  for (int a=0; a < naff; a++) {                  // affine 12x12 block (inertia + orthogonality GN Hessian)
    const mjtNum* Hb = affH + 144*a;
    for (int i=0; i < 12; i++) {
      if (aff[a].sdof[i/3] < 0) continue;         // pinned control point: no solver DOF
      mjtNum s = 0;
      for (int j=0; j < 12; j++) { if (aff[a].sdof[j/3] < 0) continue; s += Hb[12*i+j]*p[aff[a].sdof[j/3] + (j%3)]; }
      Hp[aff[a].sdof[i/3] + (i%3)] += s;
    }
  }
  for (int t=0; t < nelem; t++) {
    const ipcElem* el = &elems[t]; const mjtNum* es = estr + 12*t;   // es[3*a+k]=g_a, es[9+a]=Me_a
    mjtNum rel[3][3], s[3];
    for (int b=0; b < 3; b++) {                            // rel_b = p over edge b's endpoints
      int p0 = el->fv[ipc_eedge[b][0]], p1 = el->fv[ipc_eedge[b][1]];
      for (int k=0; k < 3; k++) rel[b][k] = (p0>=0 ? p[3*p0+k]:0) - (p1>=0 ? p[3*p1+k]:0);
      s[b] = es[3*b]*rel[b][0] + es[3*b+1]*rel[b][1] + es[3*b+2]*rel[b][2];   // g_b . rel_b
    }
    for (int a=0; a < 3; a++) {
      mjtNum ca = 2.0*(1.0+el->kD)*(ipc_Mab(el->M,a,0)*s[0] + ipc_Mab(el->M,a,1)*s[1] + ipc_Mab(el->M,a,2)*s[2]);
      mjtNum me = es[9+a];                                 // geometric tension (Me_eff, damped)
      int p0 = el->fv[ipc_eedge[a][0]], p1 = el->fv[ipc_eedge[a][1]];
      if (p0 >= 0) for (int k=0; k < 3; k++) Hp[3*p0+k] += ca*es[3*a+k] + me*rel[a][k];
      if (p1 >= 0) for (int k=0; k < 3; k++) Hp[3*p1+k] -= ca*es[3*a+k] + me*rel[a][k];
    }
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

// sparse symmetric Hessian for the IC(0) preconditioner: lower-tri CSR + column structure + workspace.
// Built/factored by the ipc_sp* helpers below mj_IPC; applied here as the CG preconditioner.
typedef struct {
  int N, nnz;
  int *rownnz, *rowadr, *colind;          // lower-tri CSR pattern
  mjtNum *val, *L;                        // H values, IC(0) factor (same pattern)
  int *crownnz, *crowadr, *crow, *cidx;   // column structure: rows in each col + their CSR value index
  mjtNum *w; int *jw;                     // IC(0) workspace: dense row accumulator + pattern marker
} ipcSparse;
static void ipc_icApply(const ipcSparse* sp, mjtNum* z, const mjtNum* r);   // defined below

// (the matrix-free PCG is now ipc_solveU below -- one unified solver for flex + affine; ipc_pcg was its
// naff==0 special case and has been folded in.)

// ===== genuine reduced-coordinate affine solve ("don't add the DOFs") =================================
// Each affine body has 12 control-point DOFs but they are NOT solver unknowns. A joint to the parent pins
// some, leaving nfree free DOFs z_b. The cp-INCREMENT obeys  dcp_b = Mb_b * dcp_parent + Nb_b * dz_b. The
// solver works only in z (sum of nfree); the per-body energy Hessians are reduced and assembled into a small
// tree-structured H_z by a composite-rigid-body (CRB) pass, factored directly. No 156-D system, no N^T H N
// over the full space, no projection -> no gradient floor.

// build per-body Mb (12x12), Nb (12 x nfree, stored 12x12), nfree, zadr from the joint constraint. Returns Nz.
// hinge child: constraint Cb*cp_b = Cp*cp_parent -> Mb = Cb^+ Cp, Nb = null(Cb). static hinge: Cp=0 -> Mb=0.
// free body (no joint): Nb = I, nfree=12, Mb=0.
static int ipc_buildReduction(ipcAffine* aff, int naff) {
  int zadr = 0;
  for (int a=0; a < naff; a++) {
    ipcAffine* af = &aff[a];
    mjtNum Cb[72], Cp[72]; int rows = 0;                          // up to 6 rows x 12
    if (af->isHinge) for (int i=0; i < af->njpt; i++) {
      if (af->pfull[i]) for (int c=0; c < 3; c++) {
        for (int j=0; j < 12; j++) { Cb[12*rows+j] = 0; Cp[12*rows+j] = 0; }
        for (int k=0; k < 4; k++) { Cb[12*rows+3*k+c] = af->jw[i][k]; Cp[12*rows+3*k+c] = af->jwp[i][k]; }
        rows++;
      } else {
        for (int j=0; j < 12; j++) { Cb[12*rows+j] = 0; Cp[12*rows+j] = 0; }
        for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) {
          Cb[12*rows+3*k+c] = af->jw[i][k]*af->pn[i][c]; Cp[12*rows+3*k+c] = af->jwp[i][k]*af->pn[i][c]; }
        rows++;
      }
    }
    af->nfree = 12 - rows; af->zadr = zadr; zadr += af->nfree;
    for (int i=0; i < 144; i++) { af->Mb[i] = 0; af->Nb[i] = 0; }
    // Nb = orthonormal null(Cb): orthonormalize Cb rows, sweep std basis projecting off rows + accepted cols
    mjtNum q[72]; int nq = 0;
    for (int r=0; r < rows; r++) {
      mjtNum v[12]; for (int j=0; j < 12; j++) v[j] = Cb[12*r+j];
      for (int pass=0; pass < 2; pass++) for (int s=0; s < nq; s++) { mjtNum d = 0;
        for (int j=0; j < 12; j++) d += v[j]*q[12*s+j]; for (int j=0; j < 12; j++) v[j] -= d*q[12*s+j]; }
      mjtNum nrm = 0; for (int j=0; j < 12; j++) nrm += v[j]*v[j]; nrm = sqrt(nrm);
      if (nrm > 1e-9) { for (int j=0; j < 12; j++) q[12*nq+j] = v[j]/nrm; nq++; }
    }
    int nf = 0;
    for (int e=0; e < 12 && nf < af->nfree; e++) {
      mjtNum v[12]; for (int j=0; j < 12; j++) v[j] = (j == e) ? 1.0 : 0.0;
      for (int pass=0; pass < 2; pass++) {
        for (int s=0; s < nq; s++) { mjtNum d = 0; for (int j=0; j < 12; j++) d += v[j]*q[12*s+j];
          for (int j=0; j < 12; j++) v[j] -= d*q[12*s+j]; }
        for (int s=0; s < nf; s++) { mjtNum d = 0; for (int j=0; j < 12; j++) d += v[j]*af->Nb[12*j+s];
          for (int j=0; j < 12; j++) v[j] -= d*af->Nb[12*j+s]; }
      }
      mjtNum nrm = 0; for (int j=0; j < 12; j++) nrm += v[j]*v[j]; nrm = sqrt(nrm);
      if (nrm > 1e-6) { for (int j=0; j < 12; j++) af->Nb[12*j+nf] = v[j]/nrm; nf++; }   // column nf of Nb
    }
    // Mb = Cb^+ Cp = Cb^T (Cb Cb^T)^-1 Cp, only for a body-body hinge (parent exists)
    if (rows > 0 && af->jparent >= 0) {
      mjtNum G[36]; for (int i=0; i < rows; i++) for (int j=0; j < rows; j++) { mjtNum s = 0;
        for (int k=0; k < 12; k++) s += Cb[12*i+k]*Cb[12*j+k]; G[rows*i+j] = s; }
      mju_cholFactor(G, rows, 1e-12);
      mjtNum Y[72]; for (int col=0; col < 12; col++) { mjtNum rhs[6], sol[6];
        for (int i=0; i < rows; i++) rhs[i] = Cp[12*i+col];
        mju_cholSolve(sol, G, rhs, rows);
        for (int i=0; i < rows; i++) Y[12*i+col] = sol[i]; }                 // Y = (Cb Cb^T)^-1 Cp  (rows x 12)
      for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) { mjtNum s = 0;  // Mb = Cb^T Y
        for (int k=0; k < rows; k++) s += Cb[12*k+i]*Y[12*k+j]; af->Mb[12*i+j] = s; }
    }
  }
  return zadr;
}

// reduced gradient norm (for the Newton convergence test): adjoint up the tree, project to z. Returns ||gz||.
static mjtNum ipc_reduceGrad(const ipcAffine* aff, int naff, const mjtNum* g, mjtNum* gz, mjtNum* adj) {
  for (int a=0; a < naff; a++) for (int i=0; i < 12; i++) adj[12*a+i] = g[12*a+i];
  for (int a=naff-1; a >= 0; a--) { const ipcAffine* af = &aff[a]; int p = af->jparent; if (p < 0) continue;
    for (int i=0; i < 12; i++) { mjtNum s = 0; for (int k=0; k < 12; k++) s += af->Mb[12*k+i]*adj[12*a+k]; adj[12*p+i] += s; } }
  mjtNum gn = 0;
  for (int a=0; a < naff; a++) { const ipcAffine* af = &aff[a]; int nf = af->nfree, z0 = af->zadr;
    for (int j=0; j < nf; j++) { mjtNum s = 0; for (int k=0; k < 12; k++) s += af->Nb[12*k+j]*adj[12*a+k];
      gz[z0+j] = -s; gn += s*s; } }
  return sqrt(gn);
}

// ===== UNIFIED solver: one preconditioned CG over reduced coords u = [u_na ; u_z] ======================
// u_na = the non-affine (flex + free-point) maximal DOF [0,Nna), mapped 1:1. u_z = the affine reduced DOF
// (Nz, joints eliminated). The map T: maximal dx[N] = T u is identity on [0,Nna) and the Nb/Mb tree map on
// the affine block [Nna,N). The Newton operator in u is Hu = T^T H T applied matrix-free: expand u->maximal
// via T, apply the existing maximal Hessian ipc_applyH, reduce via T^T. Preconditioner is block-diagonal:
// IC(0) of the non-affine Hessian (sp, dim Nna) on u_na, and the per-tree factor of the reduced affine
// Hessian on u_z. naff==0 => exactly ipc_pcg; no flex => equivalent to the old ipc_crbSolve.

// forward tree map T (parents before children): cp increment dcp_a = Nb_a u_z_a + Mb_a dcp_parent.
static void ipc_crbForward(const ipcAffine* aff, int naff, const mjtNum* uz, mjtNum* dcp /*12*naff*/) {
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a]; int pa = af->jparent; mjtNum* dca = dcp + 12*a;
    for (int i=0; i < 12; i++) { mjtNum s = 0;
      for (int j=0; j < af->nfree; j++) s += af->Nb[12*i+j]*uz[af->zadr+j];
      if (pa >= 0) for (int k=0; k < 12; k++) s += af->Mb[12*i+k]*dcp[12*pa+k];
      dca[i] = s; }
  }
}

// adjoint tree map T^T (children before parents): adj_a = f_a + sum_children Mb_c^T adj_c, outz_a = Nb_a^T adj_a.
// (Same recursion as ipc_reduceGrad but WITHOUT the negation -- this is the operator adjoint, not the RHS.)
static void ipc_crbAdjoint(const ipcAffine* aff, int naff, const mjtNum* f /*12*naff*/,
                           mjtNum* outz /*Nz*/, mjtNum* adj /*12*naff scratch*/) {
  for (int a=0; a < naff; a++) for (int i=0; i < 12; i++) adj[12*a+i] = f[12*a+i];
  for (int a=naff-1; a >= 0; a--) { const ipcAffine* af = &aff[a]; int pa = af->jparent; if (pa < 0) continue;
    for (int i=0; i < 12; i++) { mjtNum s = 0; for (int k=0; k < 12; k++) s += af->Mb[12*k+i]*adj[12*a+k];
      adj[12*pa+i] += s; } }
  for (int a=0; a < naff; a++) { const ipcAffine* af = &aff[a]; int nf = af->nfree, z0 = af->zadr;
    for (int j=0; j < nf; j++) { mjtNum s = 0; for (int k=0; k < 12; k++) s += af->Nb[12*k+j]*adj[12*a+k];
      outz[z0+j] = s; } }
}

// ABA/Featherstone two-sweep FACTORIZATION of the reduced affine Hessian (tree-structured): O(naff*12^3)
// rather than the dense O(Nz^3) Cholesky -- and it never forms the off-diagonal ancestor blocks. The per-body
// H is block diagonal, so eliminating a body's free DOF folds a Schur complement into its parent's composite.
// Bottom-up (children first). Per body: P (composite 12x12, scratch), D = N^T P N (factored pivot, nf x nf),
// A = N^T P M (nf x 12), K = D^-1 A (nf x 12). D/A/K feed ipc_crbApply.
static void ipc_crbFactor(const ipcAffine* aff, int naff, const mjtNum* H,
                          mjtNum* P /*144*naff*/, mjtNum* D /*144*naff*/,
                          mjtNum* A /*144*naff*/, mjtNum* K /*144*naff*/) {
  for (int a=0; a < naff; a++) for (int i=0; i < 144; i++) P[144*a+i] = H[144*a+i];   // P init = H_a
  for (int a=naff-1; a >= 0; a--) {
    const ipcAffine* af = &aff[a]; int nf = af->nfree, pa = af->jparent;
    const mjtNum* Pa = P + 144*a;
    mjtNum PN[144];                                          // PN = P N  (12 x nf)
    for (int i=0; i < 12; i++) for (int j=0; j < nf; j++) { mjtNum s = 0;
      for (int k=0; k < 12; k++) s += Pa[12*i+k]*af->Nb[12*k+j]; PN[12*i+j] = s; }
    mjtNum* Da = D + 144*a;                                  // D = N^T (P N)  (nf x nf), factored
    for (int i=0; i < nf; i++) for (int j=0; j < nf; j++) { mjtNum s = 0;
      for (int k=0; k < 12; k++) s += af->Nb[12*k+i]*PN[12*k+j]; Da[nf*i+j] = s; }
    mju_cholFactor(Da, nf, 1e-9);
    if (pa >= 0) {
      mjtNum PM[144];                                        // PM = P M  (12 x 12)
      for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) { mjtNum s = 0;
        for (int k=0; k < 12; k++) s += Pa[12*i+k]*af->Mb[12*k+j]; PM[12*i+j] = s; }
      mjtNum* Aa = A + 144*a;                                // A = N^T (P M)  (nf x 12)
      for (int i=0; i < nf; i++) for (int j=0; j < 12; j++) { mjtNum s = 0;
        for (int k=0; k < 12; k++) s += af->Nb[12*k+i]*PM[12*k+j]; Aa[12*i+j] = s; }
      mjtNum* Ka = K + 144*a;                                // K = D^-1 A  (nf x 12), solved column-wise
      for (int j=0; j < 12; j++) { mjtNum col[12], sol[12];
        for (int i=0; i < nf; i++) col[i] = Aa[12*i+j]; mju_cholSolve(sol, Da, col, nf);
        for (int i=0; i < nf; i++) Ka[12*i+j] = sol[i]; }
      mjtNum* Pp = P + 144*pa;                               // parent composite: Pp += M^T P M - A^T K
      for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) {
        mjtNum mtpm = 0; for (int k=0; k < 12; k++) mtpm += af->Mb[12*k+i]*PM[12*k+j];
        mjtNum atk = 0;  for (int k=0; k < nf; k++) atk  += Aa[12*k+i]*Ka[12*k+j];
        Pp[12*i+j] += mtpm - atk;
      }
    }
  }
}

// apply z = H_z^-1 r via the tree factorization (two sweeps). Backward (children first): G_a = N^T s_a + r_a,
// e_a = D_a^-1 G_a, fold s_p += M^T s_a - A^T e_a. Forward (parents first): z_a = e_a - K_a dcp_p,
// dcp_a = M dcp_p + N z_a. e_a is stashed in z[zadr..] between the sweeps; s and dcp are 12*naff scratch.
static void ipc_crbApply(const ipcAffine* aff, int naff, const mjtNum* D, const mjtNum* A, const mjtNum* K,
                         const mjtNum* r, mjtNum* z, mjtNum* s, mjtNum* dcp) {
  for (int a=0; a < naff; a++) for (int i=0; i < 12; i++) s[12*a+i] = 0;
  for (int a=naff-1; a >= 0; a--) {
    const ipcAffine* af = &aff[a]; int nf = af->nfree, z0 = af->zadr, pa = af->jparent;
    const mjtNum* sa = s + 12*a; mjtNum* ea = z + z0;
    mjtNum G[12];
    for (int i=0; i < nf; i++) { mjtNum t = 0; for (int k=0; k < 12; k++) t += af->Nb[12*k+i]*sa[k]; G[i] = t + r[z0+i]; }
    mju_cholSolve(ea, D + 144*a, G, nf);
    if (pa >= 0) { mjtNum* sp = s + 12*pa; const mjtNum* Aa = A + 144*a;
      for (int i=0; i < 12; i++) { mjtNum mts = 0; for (int k=0; k < 12; k++) mts += af->Mb[12*k+i]*sa[k];
        mjtNum ate = 0; for (int k=0; k < nf; k++) ate += Aa[12*k+i]*ea[k];
        sp[i] += mts - ate; } }
  }
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a]; int nf = af->nfree, z0 = af->zadr, pa = af->jparent;
    mjtNum* za = z + z0;
    if (pa >= 0) { const mjtNum* Ka = K + 144*a; const mjtNum* dp = dcp + 12*pa;
      for (int i=0; i < nf; i++) { mjtNum t = 0; for (int k=0; k < 12; k++) t += Ka[12*i+k]*dp[k]; za[i] -= t; } }
    mjtNum* dca = dcp + 12*a;
    for (int i=0; i < 12; i++) { mjtNum t = 0; for (int j=0; j < nf; j++) t += af->Nb[12*i+j]*za[j];
      if (pa >= 0) for (int k=0; k < 12; k++) t += af->Mb[12*i+k]*dcp[12*pa+k]; dca[i] = t; }
  }
}

// block-diagonal preconditioner apply: z = M^-1 r.  na block = IC(0) (sp), affine block = tree factor (D/A/K).
static void ipc_uPrecond(const ipcSparse* sp, int Nna, const ipcAffine* aff, int naff, int Nz,
                         const mjtNum* D, const mjtNum* A, const mjtNum* K, mjtNum* s, mjtNum* dcp,
                         const mjtNum* r, mjtNum* z) {
  if (Nna > 0) ipc_icApply(sp, z, r);
  if (Nz > 0)  ipc_crbApply(aff, naff, D, A, K, r + Nna, z + Nna, s, dcp);
}

// reduced matvec Hp = (T^T H T) p:  expand p->dxm via T, apply maximal H (ipc_applyH), reduce via T^T.
static void ipc_uMatvec(const mjtNum* p, mjtNum* Hp, int N, int Nna, int Nz,
                        const ipcAffine* aff, int naff, const mjtNum* mdiag, int ne, const ipcElem* elems,
                        const mjtNum* estr, const ipcCC* ccache, int nacon, const mjtNum* affH,
                        mjtNum* dxm, mjtNum* Hpm, mjtNum* adj) {
  if (naff == 0) {   // pure flex: no reduction, apply the maximal matrix-free Hessian directly (== old ipc_pcg)
    ipc_applyH(p, Hp, N, mdiag, ne, elems, estr, ccache, nacon, 0, aff, affH); return;
  }
  for (int i=0; i < Nna; i++) dxm[i] = p[i];                    // expand: na identity
  if (naff > 0) {                                               // affine: forward tree map (dcp in Hpm scratch)
    ipc_crbForward(aff, naff, p + Nna, Hpm);
    for (int a=0; a < naff; a++) for (int i=0; i < 12; i++) dxm[Nna+12*a+i] = Hpm[12*a+i];
  }
  ipc_applyH(dxm, Hpm, N, mdiag, ne, elems, estr, ccache, nacon, naff, aff, affH);
  for (int i=0; i < Nna; i++) Hp[i] = Hpm[i];                   // reduce: na identity
  if (naff > 0) ipc_crbAdjoint(aff, naff, Hpm + Nna, Hp + Nna, adj);   // affine: adjoint tree map
}

// unified preconditioned CG. RHS r_u = [-grad_na ; gz] (gz = the affine reduced gradient RHS from
// ipc_reduceGrad). bjf = prebuilt per-tree factors. Writes the maximal Newton direction dx[N].
static int ipc_solveU(mjtNum* dx, const mjtNum* grad, int N, int Nna, int Nz,
                      const ipcAffine* aff, int naff, const ipcSparse* sp, const mjtNum* mdiag,
                      int ne, const ipcElem* elems, const mjtNum* estr, const ipcCC* ccache, int nacon,
                      const mjtNum* affH, const mjtNum* D, const mjtNum* A, const mjtNum* K,
                      mjtNum* sfac, mjtNum* dcp, const mjtNum* gz,
                      mjtNum* r, mjtNum* z, mjtNum* p, mjtNum* Hp, mjtNum* usol,
                      mjtNum* dxm, mjtNum* Hpm, mjtNum* adj) {
  int Nu = Nna + Nz;
  for (int i=0; i < Nna; i++) r[i] = -grad[i];
  for (int j=0; j < Nz; j++) r[Nna+j] = gz[j];
  for (int i=0; i < Nu; i++) usol[i] = 0;
  mjtNum r0 = 0; for (int i=0; i < Nu; i++) r0 += r[i]*r[i];
  if (r0 < 1e-30) { for (int i=0; i < N; i++) dx[i] = 0; return 0; }
  ipc_uPrecond(sp, Nna, aff, naff, Nz, D, A, K, sfac, dcp, r, z);
  mjtNum rz = 0; for (int i=0; i < Nu; i++) { p[i] = z[i]; rz += r[i]*z[i]; }
  int it = 0;
  for (; it < 200; it++) {   // matches the old ipc_pcg cap -> naff==0 is bit-identical to the flex PCG
    ipc_uMatvec(p, Hp, N, Nna, Nz, aff, naff, mdiag, ne, elems, estr, ccache, nacon, affH, dxm, Hpm, adj);
    mjtNum pHp = 0; for (int i=0; i < Nu; i++) pHp += p[i]*Hp[i];
    if (pHp <= 1e-30) break;
    mjtNum alpha = rz/pHp, rr = 0;
    for (int i=0; i < Nu; i++) { usol[i] += alpha*p[i]; r[i] -= alpha*Hp[i]; rr += r[i]*r[i]; }
    if (rr < 1e-8*r0) break;
    ipc_uPrecond(sp, Nna, aff, naff, Nz, D, A, K, sfac, dcp, r, z);
    mjtNum rznew = 0; for (int i=0; i < Nu; i++) rznew += r[i]*z[i];
    mjtNum beta = rznew/rz; rz = rznew;
    for (int i=0; i < Nu; i++) p[i] = z[i] + beta*p[i];
  }
  for (int i=0; i < Nna; i++) dx[i] = usol[i];                  // expand solution u -> maximal dx
  if (naff > 0) { ipc_crbForward(aff, naff, usol + Nna, dxm);
    for (int a=0; a < naff; a++) for (int i=0; i < 12; i++) dx[Nna+12*a+i] = dxm[12*a+i]; }
  return it;
}

// IPC incremental-potential energy: inertia + edge-stretch penalty + contact barriers over the
// IPC incremental-potential energy: inertia + edge-stretch penalty + contact barriers over the
// per-iteration active contact list acon (cached so the line search doesn't re-enumerate all pairs)
// FEM membrane stretch energy: sum over elements of 1/4 e^T M e, e[edge] = L^2 - L0^2 (the invariant
// metric measure; matches MuJoCo's engine_passive). Shared by ipc_energyBase and ipc_energy.
static mjtNum ipc_stretchEnergy(int nelem, const ipcElem* elems, const mjtNum* x) {
  mjtNum E = 0;
  for (int t=0; t < nelem; t++) {
    mjtNum d[3][3], e[3]; ipc_elemEval(&elems[t], x, d, e);
    const mjtNum* M = elems[t].M; mjtNum kD = elems[t].kD;
    mjtNum de[3]; for (int a=0; a < 3; a++) de[a] = e[a] - elems[t].ep[a];   // elongation change (damping)
    for (int a=0; a < 3; a++) for (int b=0; b < 3; b++)
      E += 0.25*ipc_Mab(M,a,b)*(e[a]*e[b] + kD*de[a]*de[b]);   // elastic + dissipative incremental potl
  }
  return E;
}

// inertia + stretch energy only (no contact barrier); used to assemble E0 from cached gaps
static mjtNum ipc_energyBase(const mjModel* m, int nfv, int nelem, const ipcElem* elems,
                             const mjtNum* x, const mjtNum* xtil, const int* fidx,
                             const mjtNum* mass, mjtNum h, const ipcAffine* aff, int naff) {
  mjtNum E = 0, ih2 = 1.0/(h*h);
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum mh = mass[v]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  for (int a=0; a < naff; a++) E += ipc_affineGH(&aff[a], x, xtil, ih2, NULL, NULL);
  return E + ipc_stretchEnergy(nelem, elems, x);
}

static mjtNum ipc_energy(const mjModel* m, const mjData* d, int nfv, int nelem, const ipcElem* elems,
                         const mjtNum* x, const mjtNum* xtil, const int* fidx, const mjtNum* mass,
                         mjtNum h, mjtNum r, const mjtNum* rad, mjtNum ghat, mjtNum kappa,
                         const mjtNum* gv, const mjtNum* ge, const ipcCon* acon, int nacon,
                         const ipcAffine* aff, int naff, const mjtNum* xold) {
  mjtNum E = 0, ih2 = 1.0/(h*h);
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum mh = mass[v]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  for (int a=0; a < naff; a++) E += ipc_affineGH(&aff[a], x, xtil, ih2, NULL, NULL);
  E += ipc_stretchEnergy(nelem, elems, x);
  for (int c=0; c < nacon; c++) {
    mjtNum n[3], cw[4]; int idv[4], nidx;
    mjtNum ghc = ipc_conGhat(&acon[c], rad, ghat);
    mjtNum g = ipc_conGap(&acon[c], m, d, x, gv, ge, r, rad, n, idv, cw, &nidx, 1e30);
    if (g > 0 && g < ghc) E += kappa*ipc_Bg(g, ghc);
  }
  E += ipc_affineContactEnergy(m, d, aff, naff, x, xold, kappa);
  E += ipc_affineAffineEnergy(m, aff, naff, x, xold, kappa);
  return E;
}


static inline mjtNum ipc_fmin(mjtNum a, mjtNum b) { return a < b ? a : b; }

// append a candidate contact if its gap at x is below the (margin-inflated) detection threshold
static void ipc_addCand(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x,
                        const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad, mjtNum thresh,
                        ipcCon* cand, int* nc, int candmax) {
  if (*nc >= candmax) return;
  mjtNum n[3], cw[4]; int idv[4], nidx;
  mjtNum g = ipc_conGap(&con, m, d, x, gv, ge, r, rad, n, idv, cw, &nidx, thresh);
  if (g > 0 && g < thresh) cand[(*nc)++] = con;
}

// descend flex f's element BVH (built and AABB-refreshed by mj_flex at the step's xold), collecting the
// leaf element ids whose (radius-inflated) node AABB overlaps the query box [c +/- h]. Replaces the
// hand-rolled uniform spatial hash: same "nearby elements" query, but the engine's hierarchy. The node
// AABBs already include flex_radius, so a query half of thresh is a conservative superset (no pair within
// thresh is missed; the narrowphase then filters). stack must hold flex_bvhnum ints.
static int ipc_bvhBox(const mjModel* m, const mjData* d, int f, const mjtNum* c, const mjtNum* h,
                      int* stack, int* out, int maxout) {
  int bvhadr = m->flex_bvhadr[f];
  if (bvhadr < 0) return 0;
  const int* child = m->bvh_child + 2*bvhadr;
  const int* nodeid = m->bvh_nodeid + bvhadr;
  const mjtNum* aabb = d->bvh_aabb_dyn + 6*(bvhadr - m->nbvhstatic);
  int ns = 0, nout = 0; stack[ns++] = 0;
  while (ns) {
    int node = stack[--ns];
    const mjtNum* na = aabb + 6*node;                              // [center(3), halfsize(3)]
    if (mju_abs(na[0]-c[0]) > na[3]+h[0] ||
        mju_abs(na[1]-c[1]) > na[4]+h[1] ||
        mju_abs(na[2]-c[2]) > na[5]+h[2]) continue;                // box-box separation -> prune
    int c0 = child[2*node], c1 = child[2*node+1];
    if (c0 < 0 && c1 < 0) { if (nout < maxout) out[nout++] = nodeid[node]; }   // leaf -> element id
    else { if (c0 >= 0) stack[ns++] = c0; if (c1 >= 0) stack[ns++] = c1; }
  }
  return nout;
}

// build the candidate-contact list once per step, gated by a velocity-aware threshold so any pair
// that could close within the step is captured (the Newton loop then only re-tests candidates).
// Flex-flex and geom-feature-vs-flex pairs are found by querying the flex element BVH (ipc_bvhBox).
static int ipc_candidates(const mjModel* m, const mjData* d, const mjtNum* x, const mjtNum* gv,
                          const mjtNum* ge, int ngv, int nge, mjtNum r, const mjtNum* rad, mjtNum thresh,
                          mjtNum threshGeom, mjtNum maxdisp, int nfv, int npt, const int* fidx,
                          const int* flist, const int* fxadr, int nfd, const int* pt2flex,
                          ipcCon* cand, int candmax) {
  int nc = 0;
  for (int gi=0; gi < m->ngeom; gi++) {                                      // free point (flex vert / sphere) vs geom
    if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;       // skip non-colliding
    // geom-level cull: bound the points by the geom's true world AABB (the rotated local geom_aabb)
    // + per-point margin, before the per-face distance. Much tighter than the bounding sphere for the
    // elongated convex-decomposition slabs. Planes are infinite -> no cull (their distance is O(1) anyway).
    int isplane = (m->geom_type[gi] == mjGEOM_PLANE);
    mjtNum wc[3], wh[3];
    if (!isplane) {
      const mjtNum* la = m->geom_aabb + 6*gi; const mjtNum* gp = d->geom_xpos+3*gi; const mjtNum* gR = d->geom_xmat+9*gi;
      mju_mulMatVec3(wc, gR, la); for (int k=0; k < 3; k++) wc[k] += gp[k];           // world AABB center
      for (int k=0; k < 3; k++) wh[k] = mju_abs(gR[3*k])*la[3] + mju_abs(gR[3*k+1])*la[4] + mju_abs(gR[3*k+2])*la[5];
    }
    for (int v=0; v < npt; v++) {
      if (fidx[v] < 0) continue;
      mjtNum marg = thresh + rad[v];
      if (!isplane) {                                                                  // world-AABB cull
        if (mju_abs(x[3*v]-wc[0]) > wh[0]+marg || mju_abs(x[3*v+1]-wc[1]) > wh[1]+marg ||
            mju_abs(x[3*v+2]-wc[2]) > wh[2]+marg) continue;
      }
      ipcCon con = {2, {v, 0, 0, 0}, gi};
      ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, cand, &nc, candmax);
    }
  }

  // sphere-sphere (type 5): all pairs of rigid points -- nsph is tiny so the O(nsph^2) scan is cheap
  for (int p=nfv; p < npt; p++) for (int q=p+1; q < npt; q++) {
    ipcCon con = {5, {p, q, 0, 0}, -1};
    ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, cand, &nc, candmax);
  }

  // ---- BVH-based candidates over ALL dim-2 flexes: geom-feature, sphere-vs-flex, and flex-vs-flex VT/EE.
  // Each query is against one flex's element BVH (ipc_bvhBox); triangle/edge vertices are mapped from the
  // queried flex's local indices to the combined free-point space (fxadr[k] + local). flex-vs-flex contact
  // is SELF when the querying vertex/edge is in the queried flex (gated by that flex's selfcollide) and
  // INTER-FLEX otherwise (always on). Scratch buffers are sized for the largest flex. ----
  int maxbvh = 1, maxel = 1, maxen = 1;
  for (int k=0; k < nfd; k++) { int fk = flist[k];
    if (m->flex_bvhnum[fk] > maxbvh) maxbvh = m->flex_bvhnum[fk];
    if (m->flex_elemnum[fk] > maxel) maxel = m->flex_elemnum[fk];
    if (m->flex_edgenum[fk] > maxen) maxen = m->flex_edgenum[fk]; }
  int* stk    = (int*) mju_malloc(maxbvh*sizeof(int));
  int* outel  = (int*) mju_malloc(maxel*sizeof(int));
  int* stampG = (int*) mju_malloc(maxen*sizeof(int));
  for (int e=0; e < maxen; e++) stampG[e] = -1;
  int qid = 0;

  for (int k=0; k < nfd; k++) {                       // query flex fk's element BVH
    int fk = flist[k];
    int ne_k = m->flex_elemnum[fk], ea_k = m->flex_edgeadr[fk], off_k = fxadr[k];
    if (m->flex_bvhadr[fk] < 0 || ne_k == 0) continue;
    const int* el_k  = m->flex_elem + m->flex_elemdataadr[fk];
    const int* eme_k = m->flex_elemedge + m->flex_elemedgeadr[fk];
    mjtNum rk = m->flex_radius[fk];
    int doself_k = (m->flex_selfcollide[fk] != mjFLEXSELF_NONE);

    // rigid point (sphere) vs flex triangle (type 0). Generated first so the few sphere candidates are
    // never crowded out of candmax by the (potentially huge) geom-feature flood -> no ball tunneling.
    for (int p=nfv; p < npt; p++) {
      mjtNum thp = 3.0*ipc_fmin(r, ipc_fmin(rad[p], rk)) + 4.0*maxdisp;   // per-pair band: thinner radius sets it
      mjtNum qh[3] = {thp+rad[p], thp+rad[p], thp+rad[p]};
      int n = ipc_bvhBox(m, d, fk, &x[3*p], qh, stk, outel, ne_k);
      for (int i=0; i < n; i++) { int e = outel[i];
        ipcCon con = {0, {p, off_k+el_k[3*e], off_k+el_k[3*e+1], off_k+el_k[3*e+2]}, -1};
        ipc_addCand(con, m, d, x, gv, ge, r, rad, thp, cand, &nc, candmax); }
    }
    // geom-corner vs flex triangle (type 3); geom is static (one-sided) -> tighter threshGeom (the
    // convex-decomposition bin's ~1600 edges otherwise overflow candmax and drop the bag-bin contacts).
    mjtNum qhvG[3] = {threshGeom+rk, threshGeom+rk, threshGeom+rk};
    for (int c=0; c < ngv; c++) {
      int n = ipc_bvhBox(m, d, fk, &gv[3*c], qhvG, stk, outel, ne_k);
      for (int i=0; i < n; i++) { int e = outel[i];
        ipcCon con = {3, {c, off_k+el_k[3*e], off_k+el_k[3*e+1], off_k+el_k[3*e+2]}, -1};
        ipc_addCand(con, m, d, x, gv, ge, r, rad, threshGeom, cand, &nc, candmax); }
    }
    // geom-edge vs flex edge (type 4); dedup the shared triangle edges per query via stampG.
    for (int c=0; c < nge; c++) {
      const mjtNum* p0 = &ge[6*c]; const mjtNum* p1 = &ge[6*c+3]; mjtNum qc[3], qh[3];
      for (int kk=0; kk<3; kk++) { qc[kk]=0.5*(p0[kk]+p1[kk]); qh[kk]=0.5*mju_abs(p1[kk]-p0[kk])+threshGeom+rk; }
      int n = ipc_bvhBox(m, d, fk, qc, qh, stk, outel, ne_k); qid++;
      for (int i=0; i < n; i++) { int e = outel[i];
        for (int j=0; j<3; j++) { int e2 = eme_k[3*e+j]; if (stampG[e2]==qid) continue; stampG[e2]=qid;
          ipcCon con = {4, {c, off_k+m->flex_edge[2*(ea_k+e2)], off_k+m->flex_edge[2*(ea_k+e2)+1], 0}, -1};
          ipc_addCand(con, m, d, x, gv, ge, r, rad, threshGeom, cand, &nc, candmax); } }
    }
    // flex vertex vs flex triangle (type 0): self (same flex, gated by selfcollide) + inter-flex (always).
    // Asymmetric (vert vs tri), so all verts query every flex's BVH -- both directions are distinct contacts.
    for (int v=0; v < nfv; v++) {
      int kv = pt2flex[v];
      if (kv == k && !doself_k) continue;              // self-contact disabled for this flex
      mjtNum thv = 3.0*ipc_fmin(r, ipc_fmin(rad[v], rk)) + 4.0*maxdisp;   // per-pair band: thinner flex sets it
      mjtNum qh[3] = {thv+rad[v], thv+rad[v], thv+rad[v]};
      int n = ipc_bvhBox(m, d, fk, &x[3*v], qh, stk, outel, ne_k);
      for (int i=0; i < n; i++) { int e = outel[i];
        int A=off_k+el_k[3*e], B=off_k+el_k[3*e+1], C=off_k+el_k[3*e+2];
        if (kv == k && (v==A||v==B||v==C)) continue;   // skip the self-adjacent triangle
        ipcCon con = {0, {v, A, B, C}, -1};
        ipc_addCand(con, m, d, x, gv, ge, r, rad, thv, cand, &nc, candmax); }
    }
    // flex edge vs flex edge (type 1): symmetric, so canonical -- querying flex kj <= k, and e2 > e1 within
    // a flex. Self (kj==k) gated by selfcollide; inter-flex (kj<k) always.
    for (int kj=0; kj <= k; kj++) {
      int self = (kj == k); if (self && !doself_k) continue;
      int fj = flist[kj], ea_j = m->flex_edgeadr[fj], en_j = m->flex_edgenum[fj], off_j = fxadr[kj];
      for (int e1=0; e1 < en_j; e1++) {
        int a1 = off_j+m->flex_edge[2*(ea_j+e1)], b1 = off_j+m->flex_edge[2*(ea_j+e1)+1];
        mjtNum the = 3.0*ipc_fmin(r, ipc_fmin(rad[a1], rk)) + 4.0*maxdisp;   // per-pair band
        mjtNum qc[3], qh[3];
        for (int kk=0; kk<3; kk++) { qc[kk]=0.5*(x[3*a1+kk]+x[3*b1+kk]); qh[kk]=0.5*mju_abs(x[3*a1+kk]-x[3*b1+kk])+the+rad[a1]; }
        int n = ipc_bvhBox(m, d, fk, qc, qh, stk, outel, ne_k); qid++;
        for (int i=0; i < n; i++) { int e = outel[i];
          for (int j=0; j<3; j++) { int e2 = eme_k[3*e+j];
            if (self && e2 <= e1) continue;            // canonical within a flex
            if (stampG[e2] == qid) continue; stampG[e2] = qid;
            int a2 = off_k+m->flex_edge[2*(ea_k+e2)], b2 = off_k+m->flex_edge[2*(ea_k+e2)+1];
            if (a1==a2||a1==b2||b1==a2||b1==b2) continue;   // shared vertex -> adjacent, skip
            ipcCon con = {1, {a1, b1, a2, b2}, -1};
            ipc_addCand(con, m, d, x, gv, ge, r, rad, the, cand, &nc, candmax); } }
      }
    }
  }
  mju_free(stk); mju_free(outel); mju_free(stampG);
  return nc;
}

// ---------------------------------------------------------------------------------------------------
// Sparse symmetric Hessian helpers (struct ipcSparse declared above ipc_pcg). Lower-triangular CSR
// (colind sorted ascending per row, diagonal last) plus a column structure (the transpose, with the
// CSR value index of each entry) so the factorization and back-substitution can walk columns.
static int ipc_cmpll(const void* a, const void* b) {
  long long x = *(const long long*)a, y = *(const long long*)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}

// add coupling between free-dof nodes fi, fj (a full 3x3 block) to the lower-triangular key list
static void ipc_addBlock(long long* key, int* nk, int fi, int fj, int N) {
  if (fi < 0 || fj < 0) return;
  for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
    int I = 3*fi+a, J = 3*fj+b;
    if (J <= I) key[(*nk)++] = (long long)I*N + J;
  }
}

// build the lower-tri CSR pattern + column structure from mesh elements + candidate contacts
static void ipc_spBuild(const mjModel* m, ipcSparse* sp, int N, int ne, const ipcElem* elems,
                        int ncand, const ipcCon* cand, const mjtNum* cgap, const mjtNum* rad,
                        mjtNum ghat, const int* fidx, int naff, const ipcAffine* aff) {
  sp->N = N;
  // mesh+diagonal sorted-unique keys: a pure function of the model, rebuilt each step so the integrator
  // stays stateless (no cross-call globals -> reset/thread/multi-model safe). The active-only contact
  // pattern keeps the per-step cost small; a model-keyed cache is the fallback only if this ever shows up.
  int mcap = N + 81*ne + 144*naff + 1;   // affine clique: 16 blocks/body * up to 9 lower-tri keys each
  long long* mk = (long long*) mju_malloc(mcap*sizeof(long long));
  int nmk = 0;
  for (int i=0; i < N; i++) mk[nmk++] = (long long)i*N + i;          // diagonal
  for (int t=0; t < ne; t++) for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
    ipc_addBlock(mk, &nmk, elems[t].fv[i], elems[t].fv[j], N);       // membrane: element verts
  for (int a=0; a < naff; a++) for (int i=0; i < 4; i++) for (int j=0; j < 4; j++)   // affine 4-cp clique:
    if (aff[a].sdof[i] >= 0 && aff[a].sdof[i] < N && aff[a].sdof[j] >= 0 && aff[a].sdof[j] < N)  // skipped here
      ipc_addBlock(mk, &nmk, aff[a].sdof[i]/3, aff[a].sdof[j]/3, N);   // (N==Nna; affine DOF >= Nna own a per-tree factor)
  qsort(mk, nmk, sizeof(long long), ipc_cmpll);
  int spNmk = 0; for (int i=0; i < nmk; i++) if (i==0 || mk[i] != mk[i-1]) mk[spNmk++] = mk[i];  // dedup in place
  // contact keys (variable per step): collect + sort, then merge with the cached mesh keys
  int ccap = 1; for (int c=0; c < ncand; c++) ccap += 144;
  long long* ck = (long long*) mju_malloc(ccap*sizeof(long long));
  int nck = 0;
  for (int c=0; c < ncand; c++) {
    // flex-vs-geom contacts (types 3,4) and point-vs-geom (type 2) couple only flex verts that already
    // share a mesh element (or a single point), so their couplings are already in the mesh pattern -- skip.
    // Types that couple nodes NOT sharing an element add new pattern entries and MUST be included: flex
    // self-contact (0,1), sphere-vs-flex (0, sphere node <-> triangle), and sphere-vs-sphere (5).
    if (cand[c].type >= 2 && cand[c].type != 5) continue;
    mjtNum ghc = ipc_conGhat(&cand[c], rad, ghat);   // couple active + near-active contacts (within 2x their
    if (cgap[c] >= 2.0*ghc) continue;                 // activation gap) so the IC0 precond keeps the coupling
                                                       // for contacts that activate mid-Newton; far pairs skipped
    int v[4], nv; ipc_conVerts(&cand[c], v, &nv);   // skip couplings touching an affine DOF (fidx >= N/3==Nna/3):
    for (int i=0; i < nv; i++) for (int j=0; j < nv; j++) {   // they are off the IC0 block (per-tree factor owns them)
      int bi = fidx[v[i]], bj = fidx[v[j]];
      if (bi >= 0 && bi < N/3 && bj >= 0 && bj < N/3) ipc_addBlock(ck, &nck, bi, bj, N);
    }
  }
  qsort(ck, nck, sizeof(long long), ipc_cmpll);
  // merge the two sorted lists into a sorted-unique key list (avoids re-sorting the big mesh list)
  long long* key = (long long*) mju_malloc((spNmk + nck + 1)*sizeof(long long));
  int nk = 0, ia = 0, ib = 0; long long prev = -1;
  while (ia < spNmk || ib < nck) {
    long long v = (ib >= nck || (ia < spNmk && mk[ia] <= ck[ib])) ? mk[ia++] : ck[ib++];
    if (v != prev) { key[nk++] = v; prev = v; }
  }
  mju_free(ck); mju_free(mk);
  // dedup -> nnz
  int nnz = 0;
  for (int i=0; i < nk; i++) if (i == 0 || key[i] != key[i-1]) nnz++;
  sp->nnz = nnz;
  sp->rownnz = (int*) mju_malloc(N*sizeof(int));
  sp->rowadr = (int*) mju_malloc(N*sizeof(int));
  sp->colind = (int*) mju_malloc(nnz*sizeof(int));
  sp->val    = (mjtNum*) mju_malloc(nnz*sizeof(mjtNum));
  sp->L      = (mjtNum*) mju_malloc(nnz*sizeof(mjtNum));
  for (int i=0; i < N; i++) sp->rownnz[i] = 0;
  int e = 0;
  for (int i=0; i < nk; i++) {
    if (i > 0 && key[i] == key[i-1]) continue;
    int row = (int)(key[i] / N), col = (int)(key[i] % N);
    sp->colind[e] = col; sp->rownnz[row]++; e++;
  }
  sp->rowadr[0] = 0;
  for (int i=1; i < N; i++) sp->rowadr[i] = sp->rowadr[i-1] + sp->rownnz[i-1];
  // column structure (transpose with CSR value index)
  sp->crownnz = (int*) mju_malloc(N*sizeof(int));
  sp->crowadr = (int*) mju_malloc(N*sizeof(int));
  sp->crow    = (int*) mju_malloc(nnz*sizeof(int));
  sp->cidx    = (int*) mju_malloc(nnz*sizeof(int));
  for (int i=0; i < N; i++) sp->crownnz[i] = 0;
  for (int i=0; i < N; i++) for (int t=0; t < sp->rownnz[i]; t++) sp->crownnz[sp->colind[sp->rowadr[i]+t]]++;
  sp->crowadr[0] = 0;
  for (int i=1; i < N; i++) sp->crowadr[i] = sp->crowadr[i-1] + sp->crownnz[i-1];
  int* fill = sp->jw = (int*) mju_malloc(N*sizeof(int));   // reuse jw as temp fill counter
  for (int i=0; i < N; i++) fill[i] = 0;
  for (int i=0; i < N; i++) for (int t=0; t < sp->rownnz[i]; t++) {
    int idx = sp->rowadr[i]+t, col = sp->colind[idx];
    int pos = sp->crowadr[col] + fill[col]++;
    sp->crow[pos] = i; sp->cidx[pos] = idx;
  }
  sp->w  = (mjtNum*) mju_malloc(N*sizeof(mjtNum));
  for (int i=0; i < N; i++) { sp->w[i] = 0; sp->jw[i] = 0; }
  mju_free(key);
}

static void ipc_spFree(ipcSparse* sp) {
  mju_free(sp->rownnz); mju_free(sp->rowadr); mju_free(sp->colind);
  mju_free(sp->val); mju_free(sp->L);
  mju_free(sp->crownnz); mju_free(sp->crowadr); mju_free(sp->crow); mju_free(sp->cidx);
  mju_free(sp->w); mju_free(sp->jw);
}

// find the CSR value index of lower-tri entry (row, col<=row); -1 if not in pattern (binary search)
static int ipc_spIdx(const ipcSparse* sp, int row, int col) {
  int lo = sp->rowadr[row], hi = lo + sp->rownnz[row] - 1;
  while (lo <= hi) { int mid = (lo+hi)/2; int c = sp->colind[mid];
    if (c == col) return mid; else if (c < col) lo = mid+1; else hi = mid-1; }
  return -1;
}

// incomplete LDL^T factorization (symmetric ILU, no fill): sp->val (lower-tri H) -> unit-lower L in
// the off-diagonals of sp->L + signed diagonal D in the diagonal slot. No square root, so it handles
// indefinite H (negative D is fine). Left-looking with a dense row accumulator; the column structure
// supplies L[m,k]. The diagonal magnitude is floored at `shift` (keeping its sign) to avoid 1/0.
static void ipc_ic0(ipcSparse* sp, mjtNum shift) {
  int N = sp->N;
  const int *rn = sp->rownnz, *ra = sp->rowadr, *ci = sp->colind;
  const int *crn = sp->crownnz, *cra = sp->crowadr, *crow = sp->crow, *cidx = sp->cidx;
  mjtNum* L = sp->L; const mjtNum* H = sp->val; mjtNum* w = sp->w; int* jw = sp->jw;
  for (int i=0; i < N; i++) {
    int ai = ra[i], ni = rn[i];
    for (int t=0; t < ni; t++) { int j = ci[ai+t]; w[j] = H[ai+t]; jw[j] = 1; }   // gather row i
    for (int t=0; t < ni-1; t++) {                                  // off-diagonals k < i, ascending
      int k = ci[ai+t];
      mjtNum Dk = L[ra[k]+rn[k]-1];                                 // D[k]
      mjtNum Lik = w[k] / Dk;                                       // unit-lower L[i,k]
      w[k] = Lik;
      int ck = cra[k], nck = crn[k];                                // walk column k: rows m > k
      for (int s=0; s < nck; s++) {
        int m = crow[ck+s]; if (m <= k) continue;
        if (jw[m]) { mjtNum Lmk = (m == i) ? Lik : L[cidx[ck+s]]; w[m] -= Lik*Dk*Lmk; }
      }
    }
    mjtNum Di = w[i];                                               // signed pivot
    if (Di < shift && Di > -shift) Di = (Di >= 0 ? shift : -shift);
    L[ai+ni-1] = Di;
    for (int t=0; t < ni-1; t++) L[ai+t] = w[ci[ai+t]];             // store unit-L L[i,j], j<i
    for (int t=0; t < ni; t++) { int j = ci[ai+t]; w[j] = 0; jw[j] = 0; }
  }
}

// apply the incomplete-LDL^T preconditioner z = (L D L^T)^-1 r: forward unit-L solve L u = r, scale by
// D^-1, back unit-L^T solve L^T z = D^-1 u. (L diagonal slot stores D; off-diagonals are unit-L.)
static void ipc_icApply(const ipcSparse* sp, mjtNum* z, const mjtNum* r) {
  int N = sp->N;
  const int *rn = sp->rownnz, *ra = sp->rowadr, *ci = sp->colind;
  const int *crn = sp->crownnz, *cra = sp->crowadr, *crow = sp->crow, *cidx = sp->cidx;
  const mjtNum* L = sp->L;
  for (int i=0; i < N; i++) z[i] = r[i];
  for (int i=0; i < N; i++) {                                        // forward: L u = r (unit-lower)
    int ai = ra[i], ni = rn[i];
    mjtNum s = z[i];
    for (int t=0; t < ni-1; t++) s -= L[ai+t]*z[ci[ai+t]];
    z[i] = s;
  }
  for (int i=0; i < N; i++) z[i] /= L[ra[i]+rn[i]-1];                // scale by D^-1
  for (int i=N-1; i >= 0; i--) {                                     // back: L^T z = D^-1 u (unit-upper)
    int ck = cra[i], nck = crn[i];
    mjtNum s = z[i];
    for (int t=0; t < nck; t++) { int m = crow[ck+t]; if (m > i) s -= L[cidx[ck+t]]*z[m]; }
    z[i] = s;
  }
}

// IPC-style variational integrator (PROTOTYPE, phases 0a/0b): a single 2D flex -- inertia + edge-
// stretch penalty + vertex-triangle SELF-CONTACT log-barrier -- by projected-Newton with a step-
// capped (crude-CCD) line search, starting each step from the last collision-free state.
// Intersection-free and freeze-free in validation, but PROTOTYPE quality: dense solve + per-step
// malloc (small flex only), vertex-triangle only (no edge-edge), step-cap is not a rigorous CCD,
// kappa hardcoded. Falls back to Euler if there is no 2D flex.
// adaptive barrier stiffness (IPC, Li et al. 2020 sec.4): a fixed kappa cannot serve both light and heavy
// contacts -- too soft lets the minimum gap collapse toward 0, which ill-conditions the log barrier and
// (because the CCD then caps each Newton step to keep gap > 0.2*gap) collapses the step size and stalls
// Newton. kappa persists across steps; it is doubled only when tightness is BROAD (>5% of active contacts
// below 0.2 of their per-contact ghat) rather than for a single constrained outlier (which used to pin it at
// the cap and diverge), and decayed when contacts are broadly open. Clamped to [IPC_KAPPA0, IPC_KAPPAMAX].
#define IPC_KAPPA0   1000.0
#define IPC_KAPPAMAX 1.0e6

// adaptive barrier stiffness persists across steps in a file-scope global rather than mjData, so models
// using integrator=ipc don't have to declare nuserdata. NOT reset by mj_resetData (a known limitation,
// acceptable while the solver design is still in flux); g_kappaM re-inits it to the floor on a model change
// so a stale value can't leak into a different model.
static mjtNum g_kappa = 0;
static const mjModel* g_kappaM = NULL;

// radii of gyration r_j = sqrt(S_jj/m), S_jj = (I_kk+I_ll-I_jj)/2 (second moment from principal inertia)
static void ipc_radiiGyr(const mjModel* m, int b, mjtNum r[3]) {
  mjtNum mb = m->body_mass[b], Ix = m->body_inertia[3*b], Iy = m->body_inertia[3*b+1], Iz = m->body_inertia[3*b+2];
  mjtNum S[3] = { 0.5*(Iy+Iz-Ix), 0.5*(Ix+Iz-Iy), 0.5*(Ix+Iy-Iz) };
  for (int j=0; j < 3; j++) r[j] = sqrt((S[j] > mjMINVAL ? S[j] : mjMINVAL) / (mb > mjMINVAL ? mb : mjMINVAL));
}

// world positions of an affine body's 4 control points at body pose (pos, quat): cp_k = pos + R(quat)*Xr_k
static void ipc_affineCP(const ipcAffine* af, const mjtNum* pos, const mjtNum* quat, mjtNum cp[4][3]) {
  for (int k=0; k < 4; k++) {
    mjtNum v[3]; mju_rotVecQuat(v, af->Xr[k], quat);
    for (int c=0; c < 3; c++) cp[k][c] = pos[c] + v[c];
  }
}

// recover body pose (-> qpos) and velocity (-> qvel) from the 4 solved control points. F = D_curr*Dinv is
// the body rotation directly (body-frame tetra); warm-start mju_mat2Rot from quat_old (~1-2 iterations).
static void ipc_affineReadback(const ipcAffine* af, const mjtNum cp[4][3], mjtNum h,
                               const mjtNum* pos_old, const mjtNum* quat_old, mjtNum* qpos, mjtNum* qvel) {
  mjtNum Dc[9], F[9];
  for (int c=0; c < 3; c++) for (int j=0; j < 3; j++) Dc[3*c+j] = cp[j+1][c] - cp[0][c];   // world edges
  for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {                                    // F = Dc * Dinv
    mjtNum s = 0; for (int k=0; k < 3; k++) s += Dc[3*i+k]*af->Dinv[3*k+j]; F[3*i+j] = s;
  }
  mjtNum qbody[4]; for (int c=0; c < 4; c++) qbody[c] = quat_old[c];   // warm start
  mju_mat2Rot(qbody, F);                                              // closest rotation to F = R_body
  mjtNum v[3], pos[3];
  mju_rotVecQuat(v, af->Xr[0], qbody);
  for (int c=0; c < 3; c++) pos[c] = cp[0][c] - v[c];                 // body origin = cp0 - R_body*Xr0
  for (int c=0; c < 3; c++) qpos[af->qadr+c] = pos[c];
  for (int c=0; c < 4; c++) qpos[af->qadr+3+c] = qbody[c];
  for (int c=0; c < 3; c++) qvel[af->vadr+c] = (pos[c]-pos_old[c])/h;
  mjtNum dang[3];
  mju_subQuat(dang, qbody, quat_old);                                // local-frame rotational difference
  for (int c=0; c < 3; c++) qvel[af->vadr+3+c] = dang[c]/h;
}

// body rotation quaternion from the 4 control points (F = D_curr*Dinv -> closest rotation), warm-started
static void ipc_affineQuat(const ipcAffine* af, const mjtNum cp[4][3], const mjtNum* qws, mjtNum* qout) {
  mjtNum Dc[9], F[9];
  for (int c=0; c < 3; c++) for (int j=0; j < 3; j++) Dc[3*c+j] = cp[j+1][c] - cp[0][c];
  for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {
    mjtNum s = 0; for (int k=0; k < 3; k++) s += Dc[3*i+k]*af->Dinv[3*k+j]; F[3*i+j] = s; }
  for (int c=0; c < 4; c++) qout[c] = qws[c];
  mju_mat2Rot(qout, F);
}
// signed rotation angle about a unit axis of the increment from qo to qn (dq = qn * qo^-1, world frame)
// relative rotation vector (axis*angle) of the increment from qo to qn: dq = qn * qo^-1, then quat->vel
static void ipc_twistVec(const mjtNum* qn, const mjtNum* qo, mjtNum out[3]) {
  mjtNum qoi[4] = {qo[0], -qo[1], -qo[2], -qo[3]}, dq[4];
  mju_mulQuat(dq, qn, qoi);
  mju_quat2Vel(out, dq, 1.0);
}

void mj_IPC(const mjModel* m, mjData* d) {
  mjtNum h = m->opt.timestep;
  if (m != g_kappaM) { g_kappa = IPC_KAPPA0; g_kappaM = m; }
  mjtNum kappa = g_kappa > 0 ? g_kappa : IPC_KAPPA0;
  // all dim-2 flexes participate in the IPC solve (was: only the first). Their vertices are concatenated
  // into the free-point array in flex order; fxadr[k] is the free-point offset of dim-2 flex flist[k].
  int nfd = 0;
  for (int i=0; i < m->nflex; i++) if (m->flex_dim[i] == 2) nfd++;
  // affine bodies: free joint, or a single hinge whose parent is static (pendulum) or itself affine (chain).
  // isAff propagates down the tree (bodies processed parent-before-child, so the parent's flag is set first).
  char* isAff = (char*) mju_malloc((m->nbody > 0 ? m->nbody : 1)*sizeof(char));
  for (int b=0; b < m->nbody; b++) {
    isAff[b] = 0;
    int nj = m->body_jntnum[b];
    if (nj == 1 && m->jnt_type[m->body_jntadr[b]] == mjJNT_FREE) { isAff[b] = 1; continue; }
    if (nj >= 1 && nj <= 3) {                                  // 1-3 hinges sharing an anchor (ball/universal)
      int allhinge = 1;
      for (int j=m->body_jntadr[b]; j < m->body_jntadr[b]+nj; j++) if (m->jnt_type[j] != mjJNT_HINGE) allhinge = 0;
      if (allhinge && (m->body_dofnum[m->body_parentid[b]] == 0 || isAff[m->body_parentid[b]])) isAff[b] = 1;
    }
  }
  int naff = 0, naffpt = 0;   // affine bodies; every body owns its own non-degenerate 4-control-point tetra
  for (int b=0; b < m->nbody; b++) if (isAff[b]) { naff++; naffpt += 4; }
  if (nfd == 0 && naff == 0) { mj_Euler(m, d); mju_free(isAff); return; }
  ipcAffine* aff = (ipcAffine*) mju_malloc((naff > 0 ? naff : 1)*sizeof(ipcAffine));
  int* bidx = (int*) mju_malloc((m->nbody > 0 ? m->nbody : 1)*sizeof(int));   // body id -> affine index (-1)
  for (int b=0; b < m->nbody; b++) bidx[b] = -1;
  for (int b=0, a=0; b < m->nbody; b++) {
    if (!isAff[b]) continue;
    bidx[b] = a;
    ipcAffine* af = &aff[a];
    int ja = m->body_jntadr[b];
    af->body = b; af->qadr = m->jnt_qposadr[ja]; af->vadr = m->jnt_dofadr[ja]; af->ja = ja;
    int isfree = (m->jnt_type[ja] == mjJNT_FREE);
    int parstatic = (m->body_dofnum[m->body_parentid[b]] == 0);
    af->isHinge = !isfree; af->njnt = m->body_jntnum[b];
    af->jstatic = (!isfree && parstatic);
    af->jparent = (!isfree && !parstatic) ? bidx[m->body_parentid[b]] : -1;
    for (int c=0; c < 3; c++) af->hpt[c] = 0;
    mjtNum ipos[3], iquat[4], Rin[9];
    for (int c=0; c < 3; c++) ipos[c] = m->body_ipos[3*b+c];
    for (int c=0; c < 4; c++) iquat[c] = m->body_iquat[4*b+c];
    mju_quat2Mat(Rin, iquat);                                    // inertial (principal) axes in the body frame
    mjtNum mb = m->body_mass[b], Ix = m->body_inertia[3*b], Iy = m->body_inertia[3*b+1], Iz = m->body_inertia[3*b+2];
    mjtNum S[3] = { 0.5*(Iy+Iz-Ix), 0.5*(Ix+Iz-Iy), 0.5*(Ix+Iy-Iz) }, r[3];
    ipc_radiiGyr(m, b, r);
    // EVERY body (free, hinge-to-static, hinge-child) uses its own radii-of-gyration tetra: COM + principal
    // axes, always non-degenerate. The inertia is the body's real inertia expressed in this basis; joints are
    // a separate coincidence penalty, so the tetra is NEVER placed on a joint axis (no coplanar degeneracy).
    for (int c=0; c < 3; c++) af->Xr[0][c] = ipos[c];
    for (int j=0; j < 3; j++) for (int c=0; c < 3; c++) af->Xr[j+1][c] = ipos[c] + r[j]*Rin[3*c+j];
    a++;
    // Dinv = ([Xr1-Xr0, Xr2-Xr0, Xr3-Xr0])^-1 (rows c, cols m) for the orthogonality frame F = D_curr*Dinv
    mjtNum Drest[9];
    for (int c=0; c < 3; c++) for (int mm=0; mm < 3; mm++) Drest[3*c+mm] = af->Xr[mm+1][c] - af->Xr[0][c];
    ipc_mat3inv(af->Dinv, Drest);
    // B (rows [1, Xr_i^T]) and its inverse: maps node positions <-> affine coords (feature + joint weights)
    mjtNum B[16], Binv[16];
    for (int i=0; i < 4; i++) { B[4*i] = 1; for (int c=0; c < 3; c++) B[4*i+1+c] = af->Xr[i][c]; }
    ipc_mat4inv(Binv, B);
    for (int i=0; i < 16; i++) af->Binv[i] = Binv[i];
    // affine inertia M_cp4 = Binv^T M_aff Binv, M_aff = [[M, fm^T],[fm, S2]] for the RIGID COMPOSITE of this
    // body PLUS its welded (0-joint) descendants (head, hands): their mass is in qacc_smooth (which the FK
    // predictor uses), so it must be in the solve inertia too -- otherwise the parent of a welded body has
    // too-light inertia, the solve under-resolves, and the read-back injects spurious joint energy.
    // M = total mass, fm = first moment about the body origin, S2 = second moment about the body origin.
    mjtNum S2[9], Maff[16], tmp[16], Mtot = mb, fm[3];
    for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {
      mjtNum s = 0; for (int k=0; k < 3; k++) s += Rin[3*i+k]*S[k]*Rin[3*j+k];
      S2[3*i+j] = s + mb*ipos[i]*ipos[j];
    }
    for (int c=0; c < 3; c++) fm[c] = mb*ipos[c];
    for (int dby=1; dby < m->nbody; dby++) {           // welded descendants whose nearest jointed ancestor is b
      if (m->body_jntnum[dby] != 0) continue;
      int pa = m->body_parentid[dby]; while (pa > 0 && m->body_jntnum[pa] == 0) pa = m->body_parentid[pa];
      if (pa != b) continue;
      const mjtNum* Rb = d->xmat + 9*b; mjtNum cp[3], rel[3], Rp[9];
      for (int c=0; c < 3; c++) rel[c] = d->xipos[3*dby+c] - d->xpos[3*b+c];
      mju_mulMatTVec3(cp, Rb, rel);                            // welded body's COM in b's frame
      mju_mulMatTMat(Rp, Rb, d->ximat+9*dby, 3, 3, 3);         // its principal axes in b's frame
      mjtNum md = m->body_mass[dby], Id0 = m->body_inertia[3*dby], Id1 = m->body_inertia[3*dby+1], Id2 = m->body_inertia[3*dby+2];
      mjtNum Sd[3] = { 0.5*(Id1+Id2-Id0), 0.5*(Id0+Id2-Id1), 0.5*(Id0+Id1-Id2) };
      Mtot += md; for (int c=0; c < 3; c++) fm[c] += md*cp[c];
      for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {
        mjtNum s = 0; for (int k=0; k < 3; k++) s += Rp[3*i+k]*Sd[k]*Rp[3*j+k];
        S2[3*i+j] += s + md*cp[i]*cp[j];
      }
    }
    for (int i=0; i < 16; i++) Maff[i] = 0;
    Maff[0] = Mtot;
    for (int c=0; c < 3; c++) { Maff[c+1] = fm[c]; Maff[4*(c+1)] = fm[c]; }
    for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) Maff[4*(i+1)+(j+1)] = S2[3*i+j];
    for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) {        // tmp = Maff * Binv
      mjtNum s = 0; for (int k=0; k < 4; k++) s += Maff[4*i+k]*Binv[4*k+j]; tmp[4*i+j] = s;
    }
    for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) {        // Mcp4 = Binv^T * tmp
      mjtNum s = 0; for (int k=0; k < 4; k++) s += Binv[4*k+i]*tmp[4*k+j]; af->Mcp4[i][j] = s;
    }
    // orthogonality (rigidity) stiffness. The residual non-rigidity is O(stress/kappaO) and the lossy qpos
    // round-trip turns it into a contact error under strong forces; merely cranking kappaO masks that
    // (and over-conditions). The real fix is exact rigidity (AL multiplier on orthogonality) or substepping.
    mjtNum rbar2 = (r[0]*r[0] + r[1]*r[1] + r[2]*r[2]) / 3.0;
    af->kappaO = 1.0e4 * mb / (h*h) * rbar2;
    af->ghatC = 0.2 * ipc_fmin(r[0], ipc_fmin(r[1], r[2]));
    af->cdampH = 0.1 * mb / (h*h);
    // collision features as affine combinations of the control points: box -> 8 corners (radius 0),
    // capsule/cylinder -> 2 segment ends (radius), sphere -> center. weights w = Binv^T [1, P_body].
    af->nfeat = 0; af->ncap = 0;
    for (int gi=m->body_geomadr[b]; gi < m->body_geomadr[b]+m->body_geomnum[b] && af->nfeat < 24; gi++) {
      if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;
      int type = m->geom_type[gi], nf = 0, feat0 = af->nfeat; mjtNum cg[8][3], frad = 0;
      if (type == mjGEOM_BOX) {
        for (int cr=0; cr < 8; cr++) { cg[cr][0]=(cr&1?1:-1)*m->geom_size[3*gi];
          cg[cr][1]=(cr&2?1:-1)*m->geom_size[3*gi+1]; cg[cr][2]=(cr&4?1:-1)*m->geom_size[3*gi+2]; }
        nf = 8;
      } else if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
        cg[0][0]=cg[0][1]=0; cg[0][2]=-m->geom_size[3*gi+1];
        cg[1][0]=cg[1][1]=0; cg[1][2]= m->geom_size[3*gi+1]; nf = 2; frad = m->geom_size[3*gi];
      } else if (type == mjGEOM_SPHERE) {
        cg[0][0]=cg[0][1]=cg[0][2]=0; nf = 1; frad = m->geom_size[3*gi];
      }
      for (int f=0; f < nf && af->nfeat < 24; f++) {
        mjtNum v[3], Pb[3]; mju_rotVecQuat(v, cg[f], m->geom_quat+4*gi);   // P_body = geom_pos + R(geom_quat)*cg
        for (int c=0; c < 3; c++) Pb[c] = m->geom_pos[3*gi+c] + v[c];
        for (int i=0; i < 4; i++)
          af->cfeat[af->nfeat][i] = Binv[i] + Binv[4+i]*Pb[0] + Binv[8+i]*Pb[1] + Binv[12+i]*Pb[2];
        af->cfrad[af->nfeat++] = frad;
      }
      if ((type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) && nf == 2 && af->ncap < 12) {  // pair the 2 ends
        af->caps[af->ncap][0] = feat0; af->caps[af->ncap][1] = feat0+1; af->capgeom[af->ncap] = gi; af->ncap++;
      }
    }
    // welded (0-joint) descendant geoms (head, hands) become features of this affine ancestor, so they collide
    // with the floor instead of passing through. Their pose is fixed relative to b, so map the geom's WORLD
    // pose into b's frame (Pb = R_b^T (geom_world - xpos_b)) and take the affine weights as for an own geom.
    for (int dby=1; dby < m->nbody && af->nfeat < 24; dby++) {
      if (m->body_jntnum[dby] != 0) continue;
      int pa = m->body_parentid[dby]; while (pa > 0 && m->body_jntnum[pa] == 0) pa = m->body_parentid[pa];
      if (pa != b) continue;
      for (int gi=m->body_geomadr[dby]; gi < m->body_geomadr[dby]+m->body_geomnum[dby] && af->nfeat < 24; gi++) {
        if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;
        int type = m->geom_type[gi], nf = 0, feat0 = af->nfeat; mjtNum cg[8][3], frad = 0;
        if (type == mjGEOM_BOX) {
          for (int cr=0; cr < 8; cr++) { cg[cr][0]=(cr&1?1:-1)*m->geom_size[3*gi];
            cg[cr][1]=(cr&2?1:-1)*m->geom_size[3*gi+1]; cg[cr][2]=(cr&4?1:-1)*m->geom_size[3*gi+2]; }
          nf = 8;
        } else if (type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) {
          cg[0][0]=cg[0][1]=0; cg[0][2]=-m->geom_size[3*gi+1];
          cg[1][0]=cg[1][1]=0; cg[1][2]= m->geom_size[3*gi+1]; nf = 2; frad = m->geom_size[3*gi];
        } else if (type == mjGEOM_SPHERE) { cg[0][0]=cg[0][1]=cg[0][2]=0; nf = 1; frad = m->geom_size[3*gi]; }
        for (int f=0; f < nf && af->nfeat < 24; f++) {
          mjtNum world[3], rel[3], Pb[3];
          mju_mulMatVec3(world, d->geom_xmat+9*gi, cg[f]);              // world = geom_xpos + R(geom)*cg
          for (int c=0; c < 3; c++) world[c] += d->geom_xpos[3*gi+c];
          for (int c=0; c < 3; c++) rel[c] = world[c] - d->xpos[3*b+c];
          mju_mulMatTVec3(Pb, d->xmat+9*b, rel);                        // Pb = R_b^T (world - xpos_b)
          for (int i=0; i < 4; i++)
            af->cfeat[af->nfeat][i] = Binv[i] + Binv[4+i]*Pb[0] + Binv[8+i]*Pb[1] + Binv[12+i]*Pb[2];
          af->cfrad[af->nfeat++] = frad;
        }
        if ((type == mjGEOM_CAPSULE || type == mjGEOM_CYLINDER) && nf == 2 && af->ncap < 12) {
          af->caps[af->ncap][0] = feat0; af->caps[af->ncap][1] = feat0+1; af->capgeom[af->ncap] = gi; af->ncap++;
        }
      }
    }
    // hinge joints: exact linear equality constraints (axis points coincide), enforced later by null-space
    // projection. k=1: 2 points on the axis fully coincide (-> 1-DOF hinge). k=3 (ball): the anchor coincides
    // (3 rotational DOF). k=2 (universal): the anchor coincides + rotation about the 3rd axis a3=a1xa2 is
    // locked by a single row. Each point is an affine combo of THIS body's control points (jw), and of the
    // PARENT's (jwp); for a hinge to a static parent it is pinned to a fixed world target (janch).
    if (af->isHinge) {
      int k = af->njnt; mjtNum dl = (r[0]+r[1]+r[2]) / 3.0;
      for (int c=0; c < 3; c++) af->hpt[c] = d->xanchor[3*ja+c];               // shared anchor (world)
      for (int i=0; i < k; i++) for (int c=0; c < 3; c++) af->jaxis[i][c] = d->xaxis[3*(ja+i)+c];
      mjtNum P[2][3], nrm[2][3]; int full[2] = {1,1}, np;
      if (k == 1) { np = 2;
        for (int c=0; c < 3; c++) { P[0][c] = af->hpt[c] - dl*af->jaxis[0][c]; P[1][c] = af->hpt[c] + dl*af->jaxis[0][c]; }
      } else { np = 1; for (int c=0; c < 3; c++) P[0][c] = af->hpt[c];          // ball/universal: anchor coincides
        if (k == 2) { np = 2; full[1] = 0;                                     // + lock the 3rd-axis rotation
          mjtNum a3[3], lk[3];
          mju_cross(a3, af->jaxis[0], af->jaxis[1]); mju_normalize3(a3);
          mju_cross(lk, a3, af->jaxis[0]); mju_normalize3(lk);
          for (int c=0; c < 3; c++) { P[1][c] = af->hpt[c] + dl*af->jaxis[0][c]; nrm[1][c] = lk[c]; }
        }
      }
      af->njpt = np;
      const mjtNum* RBb = d->xmat + 9*b;  const mjtNum* pBb = d->xpos + 3*b;
      int pb = m->body_parentid[b];
      const mjtNum* RPb = d->xmat + 9*pb; const mjtNum* pPb = d->xpos + 3*pb;
      for (int i=0; i < np; i++) {
        af->pfull[i] = full[i];
        if (!full[i]) for (int c=0; c < 3; c++) af->pn[i][c] = nrm[i][c];
        mjtNum Ab[3], rel[3];                                                  // point P[i] in THIS body's tetra
        for (int c=0; c < 3; c++) rel[c] = P[i][c] - pBb[c];
        mju_mulMatTVec3(Ab, RBb, rel);
        for (int kk=0; kk < 4; kk++)
          af->jw[i][kk] = af->Binv[kk] + af->Binv[4+kk]*Ab[0] + af->Binv[8+kk]*Ab[1] + af->Binv[12+kk]*Ab[2];
        if (af->jstatic) { for (int c=0; c < 3; c++) af->janch[i][c] = P[i][c]; }
        else {                                                                 // same world point in the parent
          const ipcAffine* ap = &aff[af->jparent];
          mjtNum Apb[3], relp[3];
          for (int c=0; c < 3; c++) relp[c] = P[i][c] - pPb[c];
          mju_mulMatTVec3(Apb, RPb, relp);
          for (int kk=0; kk < 4; kk++)
            af->jwp[i][kk] = ap->Binv[kk] + ap->Binv[4+kk]*Apb[0] + ap->Binv[8+kk]*Apb[1] + ap->Binv[12+kk]*Apb[2];
        }
      }
    }
  }
  int* flist = (int*) mju_malloc(nfd*sizeof(int));   // the dim-2 flex ids
  int* fxadr = (int*) mju_malloc(nfd*sizeof(int));   // free-point offset of each dim-2 flex
  int nfv = 0, ne = 0;                               // total dim-2 flex verts / elements (all flexes)
  for (int i=0, k=0; i < m->nflex; i++) if (m->flex_dim[i] == 2) {
    flist[k] = i; fxadr[k] = nfv; nfv += m->flex_vertnum[i]; ne += m->flex_elemnum[i]; k++;
  }
  int f = (nfd > 0) ? flist[0] : -1;   // flex 0 sets the global barrier scale (ghat) if any flex exists
  mjtNum r = (f >= 0) ? m->flex_radius[f] : 0, ghat = r;   // contact activates within ghat of the surface gap
  // free-point -> global flex-vertex index (into flex_vertbodyid / flexvert_xpos), and -> dim-2 flex slot k
  int* pt2vg   = (int*) mju_malloc((nfv > 0 ? nfv : 1)*sizeof(int));
  int* pt2flex = (int*) mju_malloc((nfv > 0 ? nfv : 1)*sizeof(int));
  for (int k=0; k < nfd; k++) { int va_k = m->flex_vertadr[flist[k]], nv_k = m->flex_vertnum[flist[k]];
    for (int lv=0; lv < nv_k; lv++) { pt2vg[fxadr[k] + lv] = va_k + lv; pt2flex[fxadr[k] + lv] = k; } }

  // free points = bodies with 3 slide DOFs: the flex vertices (elastic) plus any standalone 3-slide body
  // carrying a sphere geom (rigid point). Generic so rigid/ABD bodies can extend this later -- a free point
  // just needs dof/qpos addr, mass, radius, and its body (for the local-slide-frame rotation R); only the
  // xold POSITION source differs (flex vertex vs geom center). Count the spheres first to size the arrays.
  char* isflexvert = (char*) mju_malloc((m->nbody > 0 ? m->nbody : 1)*sizeof(char));
  for (int b=0; b < m->nbody; b++) isflexvert[b] = 0;
  for (int v=0; v < nfv; v++) isflexvert[m->flex_vertbodyid[pt2vg[v]]] = 1;
  int nsph = 0;
  for (int b=0; b < m->nbody; b++) {
    if (isflexvert[b] || m->body_dofnum[b] != 3 || m->jnt_type[m->body_jntadr[b]] != mjJNT_SLIDE) continue;
    for (int g=m->body_geomadr[b]; g < m->body_geomadr[b]+m->body_geomnum[b]; g++)
      if (m->geom_type[g] == mjGEOM_SPHERE) { nsph++; break; }
  }
  int npt = nfv + nsph + naffpt;
  int* dofadr = (int*) mju_malloc(npt*sizeof(int));
  int* qpadr  = (int*) mju_malloc(npt*sizeof(int));   // qpos address (NOT dof address: differs after
  int* fidx   = (int*) mju_malloc(npt*sizeof(int));   // free/ball joints, which have more qpos than dof)
  mjtNum* mass = (mjtNum*) mju_malloc(npt*sizeof(mjtNum));
  mjtNum* rad  = (mjtNum*) mju_malloc(npt*sizeof(mjtNum));   // per-point radius (flex_radius / sphere size)
  int* pbody  = (int*) mju_malloc(npt*sizeof(int));          // body id, for the slide-frame rotation R
  int* pgeom  = (int*) mju_malloc(npt*sizeof(int));          // sphere geom id (>=0); -1 for a flex vertex
  char* pcp   = (char*) mju_malloc(npt*sizeof(char));        // 1 = affine-body control point (its xold/xtil
  for (int v=0; v < npt; v++) pcp[v] = 0;                    // and readback are handled by the affine loop)
  int nfree = 0;
  for (int k=0; k < nfd; k++) {                               // flex vertices, per dim-2 flex
    int fi = flist[k]; mjtNum rk = m->flex_radius[fi];
    int va_k = m->flex_vertadr[fi], nv_k = m->flex_vertnum[fi];
    for (int lv=0; lv < nv_k; lv++) {
      int v = fxadr[k] + lv, bid = m->flex_vertbodyid[va_k + lv];
      dofadr[v] = -1; qpadr[v] = -1; fidx[v] = -1; mass[v] = 0; rad[v] = rk; pbody[v] = bid; pgeom[v] = -1;
      if (m->body_dofnum[bid] == 3) {
        int da = m->body_dofadr[bid];
        dofadr[v] = da; qpadr[v] = m->jnt_qposadr[m->body_jntadr[bid]]; fidx[v] = nfree++;
        mass[v] = d->qM[m->M_rowadr[da] + m->M_rownnz[da] - 1];   // diagonal (point mass)
      }
    }
  }
  int p = nfv;
  for (int b=0; b < m->nbody; b++) {                         // append the rigid sphere points
    if (isflexvert[b] || m->body_dofnum[b] != 3 || m->jnt_type[m->body_jntadr[b]] != mjJNT_SLIDE) continue;
    int gsph = -1;
    for (int g=m->body_geomadr[b]; g < m->body_geomadr[b]+m->body_geomnum[b]; g++)
      if (m->geom_type[g] == mjGEOM_SPHERE) { gsph = g; break; }
    if (gsph < 0) continue;
    int da = m->body_dofadr[b];
    dofadr[p] = da; qpadr[p] = m->jnt_qposadr[m->body_jntadr[b]]; fidx[p] = nfree++;
    mass[p] = d->qM[m->M_rowadr[da] + m->M_rownnz[da] - 1];
    rad[p] = m->geom_size[3*gsph]; pbody[p] = b; pgeom[p] = gsph;
    p++;
  }
  for (int a=0; a < naff; a++) {                             // append each affine body's 4 control points
    ipcAffine* af = &aff[a];                                 // (every body owns all 4; joints are penalties)
    for (int k=0; k < 4; k++) {
      af->cp[k] = p;
      dofadr[p] = af->vadr; qpadr[p] = af->qadr;             // (shared joint addrs; handled by the aff loop)
      fidx[p] = nfree++; af->sdof[k] = 3*fidx[p];
      mass[p] = 0;                                           // inertia comes from the affine M_cp block,
      rad[p] = 0; pbody[p] = af->body; pgeom[p] = -1; pcp[p] = 1;   // not the per-point diagonal
      p++;
    }
  }
  mju_free(isflexvert);
  int N = 3*nfree, Na = (N > 0 ? N : 1);
  // FEM membrane elements (all dim-2 flexes): consume MuJoCo's precomputed flex_stiffness metric (from the
  // model's <elasticity young=.../> tag) per element + the rest squared edge lengths. No hand-rolled springs.
  // Element vertex indices are mapped from per-flex local (el_k) to the combined free-point space (fxadr[k]+).
  ipcElem* elems = (ipcElem*) mju_malloc((ne > 0 ? ne : 1)*sizeof(ipcElem));
  for (int k=0, et=0; k < nfd; k++) {
    int fi = flist[k], sa_k = m->flex_stiffnessadr[fi], ea_k = m->flex_edgeadr[fi], ne_k = m->flex_elemnum[fi];
    const int* el_k = m->flex_elem + m->flex_elemdataadr[fi];
    const int* eme_k = m->flex_elemedge + m->flex_elemedgeadr[fi];
    mjtNum kD_k = (h > 0) ? m->flex_damping[fi]/h : 0;          // stiffness-prop. Rayleigh damping
    for (int t=0; t < ne_k; t++, et++) {
      for (int i=0; i < 3; i++) { int gp = fxadr[k] + el_k[3*t+i]; elems[et].vg[i] = gp; elems[et].fv[i] = fidx[gp]; }
      for (int j=0; j < 6; j++) elems[et].M[j] = (sa_k >= 0) ? m->flex_stiffness[sa_k + 21*t + j] : 0;
      for (int a=0; a < 3; a++) {
        mjtNum L0 = m->flexedge_length0[ea_k + eme_k[3*t+a]]; elems[et].Lr2[a] = L0*L0; elems[et].ep[a] = 0;
      }
      elems[et].kD = kD_k;
    }
  }
  int amax = npt*64 + 1024;                   // capacity of the active-contact list
  ipcCon* acon = (ipcCon*) mju_malloc(amax*sizeof(ipcCon));
  ipcCC* ccache = (ipcCC*) mju_malloc(amax*sizeof(ipcCC));
  int candmax = npt*192 + 8192;               // capacity of the per-step candidate list (sized for the
                                              // geom-feature-heavy bag-in-bin contact: ~160k at npt~1100)
  ipcCon* cand = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon));
  mjtNum* cgap = (mjtNum*) mju_malloc(candmax*sizeof(mjtNum));   // per-candidate gap at x (try->ccd/E0)
  ipcCon* candLS = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon)); // line-search subset (can activate this step)
  // precompute static-geom sharp features (vertices/edges) once per step (geoms are fixed here).
  // box -> 8 verts/12 edges; mesh -> all verts / all hull-poly edges. Cap = sum over colliding geoms.
  int gvcap = 1, gecap = 1;
  for (int gi=0; gi < m->ngeom; gi++) {
    if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;          // skip non-colliding
    int type = m->geom_type[gi];
    if (type == mjGEOM_BOX) { gvcap += 8; gecap += 12; }
    else if (type == mjGEOM_MESH) {
      int mid = m->geom_dataid[gi];
      gvcap += m->mesh_vertnum[mid];
      int pa = m->mesh_polyadr[mid], pn = m->mesh_polynum[mid];
      for (int p=0; p < pn; p++) gecap += m->mesh_polyvertnum[pa+p];             // upper bound (pre-dedup)
    }
  }
  int ngv = 0, nge = 0;
  mjtNum* gv = (mjtNum*) mju_malloc(3*gvcap*sizeof(mjtNum));
  mjtNum* ge = (mjtNum*) mju_malloc(6*gecap*sizeof(mjtNum));
  for (int gi=0; gi < m->ngeom; gi++) {
    if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;          // skip non-colliding
    ngv += ipc_geomVerts(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, gv+3*ngv);
    nge += ipc_geomEdges(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, ge+6*nge);
  }
  mjtNum* x    = (mjtNum*) mju_malloc(3*npt*sizeof(mjtNum));
  mjtNum* xtil = (mjtNum*) mju_malloc(3*npt*sizeof(mjtNum));
  mjtNum* xold = (mjtNum*) mju_malloc(3*npt*sizeof(mjtNum));
  mjtNum* xn   = (mjtNum*) mju_malloc(3*npt*sizeof(mjtNum));
  mjtNum* grad = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* dx   = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* mdiag= (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // inertia diagonal (for the H*p apply)
  mjtNum* rcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // CG residual / search-dir / Hp buffers
  mjtNum* zcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* pcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* Hpv  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* usol = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // unified solver: u-space solution + matvec scratch
  mjtNum* dxm  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* Hpm  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* affH = (mjtNum*) mju_malloc((naff > 0 ? 144*naff : 1)*sizeof(mjtNum)); // per-affine 12x12 Hessian
  mjtNum* estr = (mjtNum*) mju_malloc((ne > 0 ? 12*ne : 1)*sizeof(mjtNum)); // per-element g_a (9) + Me_a (3)
                                                                           // (full, PSD-projected, this Newton iter)

  const mjtNum* vx = d->flexvert_xpos;
  for (int v=0; v < npt; v++) {
    if (pcp[v]) continue;   // affine control point: xold/xtil set in the affine loop below
    // xold position source: flex vertex from flexvert_xpos, rigid sphere from its geom center
    if (pgeom[v] < 0) for (int c=0; c < 3; c++) xold[3*v+c] = vx[3*pt2vg[v]+c];
    else              for (int c=0; c < 3; c++) xold[3*v+c] = d->geom_xpos[3*pgeom[v]+c];
    if (fidx[v] >= 0) {
      int da = dofadr[v];
      // qvel/qacc_smooth are in the body's local slide frame; map to WORLD via the body rotation R
      // (= d->xmat). R=I for unrotated bodies (cloth, the balls); non-identity for the rotated bag.
      const mjtNum* R = d->xmat + 9*pbody[v];
      mjtNum vw[3], aw[3];
      mju_mulMatVec3(vw, R, d->qvel + da);
      mju_mulMatVec3(aw, R, d->qacc_smooth + da);
      for (int c=0; c < 3; c++)
        xtil[3*v+c] = xold[3*v+c] + h*vw[c] + h*h*aw[c];
    } else {
      for (int c=0; c < 3; c++) xtil[3*v+c] = xold[3*v+c];   // pinned: fixed
    }
  }
  // predicted body poses for the affine inertial target: integrate qpos one unconstrained step
  // (qvel + h*qacc_smooth), run FK, capture the body poses, restore. Uniform over any joint structure
  // (free / hinge / articulated tree), and keeps shared anchor nodes consistent between parent and child.
  mjtNum* xpos_pred = NULL; mjtNum* xquat_pred = NULL;
  if (naff > 0) {
    xpos_pred  = (mjtNum*) mju_malloc(3*m->nbody*sizeof(mjtNum));
    xquat_pred = (mjtNum*) mju_malloc(4*m->nbody*sizeof(mjtNum));
    mjtNum* qpos0 = (mjtNum*) mju_malloc((m->nq > 0 ? m->nq : 1)*sizeof(mjtNum));
    mjtNum* qvt   = (mjtNum*) mju_malloc((m->nv > 0 ? m->nv : 1)*sizeof(mjtNum));
    for (int i=0; i < m->nq; i++) qpos0[i] = d->qpos[i];
    for (int i=0; i < m->nv; i++) qvt[i] = d->qvel[i] + h*d->qacc_smooth[i];
    mj_integratePos(m, d->qpos, qvt, h);
    mj_kinematics(m, d);
    for (int i=0; i < 3*m->nbody; i++) xpos_pred[i]  = d->xpos[i];
    for (int i=0; i < 4*m->nbody; i++) xquat_pred[i] = d->xquat[i];
    for (int i=0; i < m->nq; i++) d->qpos[i] = qpos0[i];
    mj_kinematics(m, d);                              // restore current poses
    mju_free(qpos0); mju_free(qvt);
  }
  // affine control points: xold = FK(current pose), xtil = FK(predicted pose), via each node's OWNER body
  // (shared nodes are set once, by the parent that owns them). Pinned axis nodes are on the rotation axis,
  // so FK leaves them fixed -> xtil == xold automatically.
  for (int a=0; a < naff; a++) {
    ipcAffine* af = &aff[a];
    mjtNum cp0[4][3], cpp[4][3];
    ipc_affineCP(af, d->xpos+3*af->body, d->xquat+4*af->body, cp0);
    ipc_affineCP(af, xpos_pred+3*af->body, xquat_pred+4*af->body, cpp);
    for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) {
      xold[3*af->cp[k]+c] = cp0[k][c];
      xtil[3*af->cp[k]+c] = cpp[k][c];
    }
  }
  for (int i=0; i < 3*npt; i++) x[i] = xold[i];   // start from last collision-free state (feasibility)

  mjtNum ih2 = 1.0/(h*h);
  for (int i=0; i < N; i++) mdiag[i] = 0;                          // inertia diagonal (fixed per step)
  for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
    int fi = fidx[v]; for (int c=0; c < 3; c++) mdiag[3*fi+c] = mass[v]*ih2;
  }
  // build the candidate-contact list once per step: detection threshold inflated by the predictor
  // displacement so any pair that could close during the step is captured (verified by gap checks)
  mjtNum maxdisp = 0;
  for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
    mjtNum dd[3]; for (int c=0; c < 3; c++) dd[c] = xtil[3*v+c]-xold[3*v+c];
    mjtNum L = sqrt(mju_dot3(dd, dd)); if (L > maxdisp) maxdisp = L;
  }
  mjtNum thresh = 3*ghat + 4*maxdisp;
  // Static geom features (the bin) collide one-sided with the flex -- only the flex side moves, so a pair
  // closes at most at the flex speed. Their detection margin can therefore be half the two-sided sphere/self
  // margin (which budgets 2*maxdisp of approach from each side). Using the full thresh here makes the
  // 36-piece convex-decomposition bin (~1600 edges) generate ~600k candidates that overflow candmax and
  // drop the edge-edge contacts -> the bag sinks into the bin. ghat + 2*maxdisp keeps the count ~160k.
  mjtNum threshGeom = ghat + 2.0*maxdisp;
  int ncand = ipc_candidates(m, d, x, gv, ge, ngv, nge, r, rad, thresh, threshGeom, maxdisp, nfv, npt, fidx,
                             flist, fxadr, nfd, pt2flex, cand, candmax);
  // warm start: with no candidate contacts within thresh, the predictor x~ is collision-free (the thresh
  // margin covers the step displacement), so it is a far better feasible initial guess than xold and Newton
  // converges in ~1 iteration instead of ~2 -- halving the cost of contact-free steps. NOT for affine bodies:
  // their contacts aren't in ncand, so x~ (the unconstrained rigid step) may penetrate -> start from xold.
  if (ncand == 0 && naff == 0) for (int i=0; i < 3*npt; i++) x[i] = xtil[i];
  // sparse Hessian pattern (mesh + candidate-contact couplings) for the IC(0) preconditioner, once/step
  // Rayleigh damping uses the previous-step elongations e_prev = e(xold); xold is fixed over the Newton
  // loop, so cache e_prev once per step. Only needed when damping is on.
  if (nfd > 0 && m->flex_damping[f] > 0)
    for (int t=0; t < ne; t++) { mjtNum dtmp[3][3]; ipc_elemEval(&elems[t], xold, dtmp, elems[t].ep); }
  ipcSparse sp = {0};   // IC(0) preconditioner Hessian -- only built/used on the PCG path (!useCRB)
  // mark active inter-flex/sphere candidates (gap already within their per-contact ghat) so the IC0 pattern
  // couples only those (~hundreds) instead of all ~15k candidates (~99% inactive). gap here == iter-0 gap
  // (x is unchanged until the Newton loop); geom contacts (types 2/3/4) add no coupling so they're skipped.
  for (int c=0; c < ncand; c++) {
    if (cand[c].type >= 2 && cand[c].type != 5) continue;
    mjtNum nn[3], cw[4]; int iv[4], nidx;
    cgap[c] = ipc_conGap(&cand[c], m, d, x, gv, ge, r, rad, nn, iv, cw, &nidx, thresh);
  }
  // UNIFIED solve: one PCG in reduced coords u = [non-affine maximal DOF [0,Nna) ; affine reduced DOF (Nz)].
  // The affine control points are the last 12*naff DOF of N (contiguous, see setup), so Nna = N - 12*naff is
  // exactly the flex+free-point block. The IC(0) sparse Hessian sp covers ONLY that non-affine block; the
  // affine block is preconditioned by its per-tree reduced-Hessian factor. (Both old paths -- the jointed
  // CRB direct solve and the flex PCG -- are special cases of ipc_solveU below.)
  int Nna = N - 12*naff;
  if (Nna > 0) ipc_spBuild(m, &sp, Nna, ne, elems, ncand, cand, cgap, rad, ghat, fidx, naff, aff);
  // precompute element -> CSR scatter indices once per step (the pattern is fixed over the Newton loop), so
  // the per-Newton assembly avoids a binary-search ipc_spIdx per matrix entry. -1 marks skipped entries
  // (upper-tri or pinned vertex). Order matches the assembly loop: idx = (i*3+j)*9 + a*3+b.
  int* escat = (Nna > 0) ? (int*) mju_malloc(ne*81*sizeof(int)) : NULL;
  if (Nna > 0) for (int t=0; t < ne; t++) {
    const ipcElem* elem = &elems[t]; int k = 0;
    for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {
      int fi = elem->fv[i], fj = elem->fv[j];
      for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
        int R = 3*fi+a, C = 3*fj+b;
        escat[t*81 + k++] = (fi < 0 || fj < 0 || C > R) ? -1 : ipc_spIdx(&sp, R, C);
      }
    }
  }
  mjtNum g0 = -1;   // initial gradient norm (set on the first Newton iteration), for the relative stop test
  // active-set re-test: after the first Newton iter, only candidates with gap < ghat are re-tested for the
  // barrier (the contact interface). cgap is kept a valid LOWER BOUND for the rest by decrementing it each
  // iter by delta = 4*alpha*max|dx| (a bound on how fast any gap can shrink), and refreshing it exactly
  // when a candidate enters the set. So the lower bound crosses ghat no later than the true gap (no active
  // contact is ever missed in the barrier), and the CCD's cgap-based step cap stays conservative (safe).
  // GENUINE reduced-coordinate solve for jointed-affine, no-flex scenes (humanoid, pendulums): the free DOFs
  // z are the only unknowns (useCRB computed above). Per-body Mb/Nb derived once; the reduced Hessian is
  // assembled per-body by a composite (CRB) tree pass and factored directly. flex/free-affine -> PCG below.
  int Nz = 0; mjtNum *crbGz = NULL, *crbAdj = NULL, *crbHp = NULL, *crbHc = NULL,
                     *crbP = NULL, *crbD = NULL, *crbA = NULL, *crbK = NULL, *crbS = NULL, *crbDcp = NULL;
  if (naff > 0) {
    Nz = ipc_buildReduction(aff, naff);
    crbGz  = (mjtNum*) mju_malloc((size_t)Nz*sizeof(mjtNum));      // affine reduced gradient RHS (gz)
    crbAdj = (mjtNum*) mju_malloc((size_t)12*naff*sizeof(mjtNum)); // matvec tree adjoint scratch
    crbHp  = (mjtNum*) mju_malloc((size_t)144*naff*sizeof(mjtNum));// per-body H = affH + contact
    crbHc  = (mjtNum*) mju_malloc((size_t)144*naff*sizeof(mjtNum));// per-body contact Hessian
    crbP   = (mjtNum*) mju_malloc((size_t)144*naff*sizeof(mjtNum));// ABA composite (factor scratch)
    crbD   = (mjtNum*) mju_malloc((size_t)144*naff*sizeof(mjtNum));// per-body pivot factors D
    crbA   = (mjtNum*) mju_malloc((size_t)144*naff*sizeof(mjtNum));// per-body A = N^T P M
    crbK   = (mjtNum*) mju_malloc((size_t)144*naff*sizeof(mjtNum));// per-body K = D^-1 A
    crbS   = (mjtNum*) mju_malloc((size_t)12*naff*sizeof(mjtNum)); // apply scratch (cp-space accumulator)
    crbDcp = (mjtNum*) mju_malloc((size_t)12*naff*sizeof(mjtNum)); // apply scratch (cp increments)
  }
  int nact0 = 0, nlt02 = 0, nlt06 = 0;   // active contacts at iter 0: total, ratio<0.2, ratio<0.6 (kappa adapt)
  for (int it=0; it < 40 && N > 0; it++) {
    for (int i=0; i < N; i++) grad[i] = 0;
    if (Nna > 0) {                                            // IC0 sparse Hessian over the non-affine block
      for (int i=0; i < sp.nnz; i++) sp.val[i] = 0;
      for (int fi=0; fi < Nna/3; fi++) for (int c=0; c < 3; c++)   // inertia: diagonal
        sp.val[ipc_spIdx(&sp, 3*fi+c, 3*fi+c)] += mdiag[3*fi+c];
    }
    for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
      int fi = fidx[v]; mjtNum mh = mass[v]*ih2;
      for (int c=0; c < 3; c++) grad[3*fi+c] += mh*(x[3*v+c]-xtil[3*v+c]);
    }
    // FEM membrane stretch (P1, engine_passive's metric form): per element accumulate the gradient
    // (= -force), cache g_a/Me_a in estr for the matrix-free H*p, AND scatter the full element Hessian
    // (Gauss-Newton 2 sum_ab M[a,b] g_a g_b^T + geometric sum_a Me_a (Laplacian_a (x) I)) into sp.
    for (int t=0; t < ne; t++) {
      const ipcElem* elem = &elems[t]; mjtNum* es = estr + 12*t;
      mjtNum d[3][3], e3[3]; ipc_elemEval(elem, x, d, e3);
      mjtNum kD = elem->kD, ee[3];                           // ee = effective (damped) elongation:
      for (int a=0; a < 3; a++) ee[a] = (1.0+kD)*e3[a] - kD*elem->ep[a];   // (L^2-L0^2) + kD(L^2-Lprev^2)
      mjtNum Me[3];                                          // Me[b] = sum_a M[a,b] ee[a] (edge tension)
      for (int b=0; b < 3; b++)
        Me[b] = ipc_Mab(elem->M,0,b)*ee[0] + ipc_Mab(elem->M,1,b)*ee[1] + ipc_Mab(elem->M,2,b)*ee[2];
      for (int a=0; a < 3; a++) { for (int k=0; k < 3; k++) es[3*a+k] = d[a][k]; es[9+a] = Me[a]; }
      for (int b=0; b < 3; b++) {                            // grad[n] += sum_b Me[b] g_b[n] (= -force)
        int p0 = elem->fv[ipc_eedge[b][0]], p1 = elem->fv[ipc_eedge[b][1]];
        if (p0 >= 0) for (int k=0; k < 3; k++) grad[3*p0+k] += Me[b]*d[b][k];
        if (p1 >= 0) for (int k=0; k < 3; k++) grad[3*p1+k] -= Me[b]*d[b][k];
      }
      mjtNum He[9][9]; for (int I=0; I < 9; I++) for (int J=0; J < 9; J++) He[I][J] = 0;
      mjtNum G[3][9];                                         // G_a = d(e_a)/dx over the 3 local verts
      for (int a=0; a < 3; a++) { for (int i=0; i < 9; i++) G[a][i] = 0;
        int p = ipc_eedge[a][0], q = ipc_eedge[a][1];
        for (int k=0; k < 3; k++) { G[a][3*p+k] = d[a][k]; G[a][3*q+k] = -d[a][k]; } }
      for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {   // Gauss-Newton part (x(1+kD) from damping)
        mjtNum m2 = 2.0*(1.0+kD)*ipc_Mab(elem->M, a, b);
        for (int I=0; I < 9; I++) for (int J=0; J < 9; J++) He[I][J] += m2*G[a][I]*G[b][J]; }
      for (int a=0; a < 3; a++) {                            // geometric stress part
        int p = ipc_eedge[a][0], q = ipc_eedge[a][1];
        for (int k=0; k < 3; k++) {
          He[3*p+k][3*p+k] += Me[a]; He[3*q+k][3*q+k] += Me[a];
          He[3*p+k][3*q+k] -= Me[a]; He[3*q+k][3*p+k] -= Me[a]; } }
      for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {   // scatter lower-tri into sp (precomputed idx)
        int base = t*81 + (i*3+j)*9;
        for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
          int id = escat[base + a*3+b]; if (id >= 0) sp.val[id] += He[3*i+a][3*j+b]; } }
    }
    // affine bodies: inertia (M_cp star) + orthogonality GN Hessian -> grad and affH (matrix-free apply).
    // The 12x12 block goes into the matvec (ipc_applyH) and, reduced, into the per-tree preconditioner -- it
    // is NOT scattered into the IC0 sparse Hessian (the affine block lives in the per-tree factor, not sp).
    for (int a=0; a < naff; a++) {
      mjtNum g12[12], H12[12][12];
      ipc_affineGH(&aff[a], x, xtil, ih2, g12, H12);
      mjtNum* Hb = affH + 144*a;
      for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) Hb[12*i+j] = H12[i][j];
      for (int i=0; i < 12; i++) grad[aff[a].sdof[i/3] + (i%3)] += g12[i];
    }
    // (hinge joints are eliminated exactly by the reduced coordinates -- the per-body cp gradient/Hessian
    // above are reduced into the free DOFs z inside ipc_solveU; nothing joint-related is added here.)
    // assemble active contacts. First Newton iter: scan all candidates (initializes cgap exactly). Later
    // iters: only re-test the active set {cgap < ghat} (cgap is a maintained lower bound, so this set
    // contains every candidate that is or could be active). grad/acon/ccache get the active barrier blocks.
    int nacon = 0;
    if (it == 0) {
      for (int c=0; c < ncand; c++)
        ipc_try(cand[c], m, d, x, gv, ge, r, rad, ghat, kappa, fidx, grad, acon, ccache, &nacon, amax, &cgap[c]);
    } else {
      for (int c=0; c < ncand; c++) if (cgap[c] < ghat)
        ipc_try(cand[c], m, d, x, gv, ge, r, rad, ghat, kappa, fidx, grad, acon, ccache, &nacon, amax, &cgap[c]);
    }
    if (naff > 0) for (int i=0; i < 144*naff; i++) crbHc[i] = 0;   // per-body contact Hessian (affine precond)
    ipc_affineContact(m, d, aff, naff, x, xold, kappa, grad, ccache, &nacon, amax, naff > 0 ? crbHc : NULL);
    ipc_affineAffineContact(m, aff, naff, x, xold, kappa, grad, ccache, &nacon, amax, naff > 0 ? crbHc : NULL);  // self-collision
    if (Nna > 0) for (int c=0; c < nacon; c++) {             // contact GN -> IC0 sparse Hessian (non-affine block)
      const ipcCC* cc = &ccache[c];
      for (int p=0; p < cc->nidx; p++) for (int q=0; q < cc->nidx; q++) {
        int fp = cc->f[p], fq = cc->f[q]; if (fp < 0 || fq < 0) continue;
        if (3*fp >= Nna || 3*fq >= Nna) continue;   // affine DOF: not in sp (handled by the per-tree factor)
        mjtNum w2 = cc->bdd*cc->cw[p]*cc->cw[q];
        for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
          int R = 3*fp+a, C = 3*fq+b; if (C > R) continue;
          int id = ipc_spIdx(&sp, R, C);              // -1 if this coupling was not in the pattern:
          if (id >= 0) sp.val[id] += w2*cc->n[a]*cc->n[b];   // drop it (degrades preconditioner, never corrupts)
        } }
    }
    // iter-0 cgap holds the exact start-of-step gaps -> active-contact tightness stats for the kappa adapt
    if (it == 0) for (int c=0; c < ncand; c++) if (cgap[c] > 0) {
      mjtNum ghc_c = ipc_conGhat(&cand[c], rad, ghat);
      mjtNum ratio = cgap[c] / ghc_c;
      if (cgap[c] < ghc_c) { nact0++; if (ratio < 0.2) nlt02++; if (ratio < 0.6) nlt06++; }
    }
    // same tightness stats for affine contacts, so kappa adapts (and the rest gap tightens) in rigid scenes
    if (it == 0) for (int a=0; a < naff; a++) {
      const ipcAffine* af = &aff[a]; mjtNum ghc = af->ghatC;
      for (int ft=0; ft < af->nfeat; ft++) {
        mjtNum pw[3] = {0,0,0};
        for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) pw[c] += af->cfeat[ft][k]*x[3*af->cp[k]+c];
        for (int gi=0; gi < m->ngeom; gi++) {
          if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;
          if (!ipc_bodyAnchored(m, m->geom_bodyid[gi])) continue;
          mjtNum n[3], g = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, pw, n, ghc+af->cfrad[ft])
                           - af->cfrad[ft];
          if (g > 0 && g < ghc) { mjtNum ratio = g/ghc; nact0++; if (ratio < 0.2) nlt02++; if (ratio < 0.6) nlt06++; }
        }
      }
    }
    // converged when the residual force has dropped well below its initial magnitude. Relative (rather
    // than the old absolute 1e-8) so the test is scale-invariant across scenes/units: the absolute floor
    // depends on mass/h^2, kappa, mesh, so a fixed 1e-8 sat at/below the achievable gradient floor for
    // stiff-contact steps -> Newton ground out useless iterations after it had already converged.
    // residual norm: reduced (||grad_z||, the genuine reduced gradient) for the CRB solve, else projected ||grad||
    // unified reduced residual norm: ||[grad_na ; gz]|| spanning the non-affine block and the affine reduced
    // gradient gz = -Nb^T adj (ipc_reduceGrad, on the affine block grad[Nna..N)). crbGz holds gz for the solve.
    mjtNum gnorm, gna2 = 0;
    for (int i=0; i < Nna; i++) gna2 += grad[i]*grad[i];
    mjtNum gaff = 0;
    if (naff > 0) {
      for (int a=0; a < naff; a++) for (int i=0; i < 144; i++) crbHp[144*a+i] = affH[144*a+i] + crbHc[144*a+i];
      gaff = ipc_reduceGrad(aff, naff, grad + Nna, crbGz, crbAdj);   // crbGz = gz (affine RHS), gaff = ||gz||
    }
    gnorm = sqrt(gna2 + gaff*gaff);
    if (g0 < 0) g0 = gnorm;                                   // initial residual (first Newton iteration)
    // the affine GN orthogonality converges only LINEARLY near the floor, so chasing 1e-7*g0 wastes ~25 iters
    // for no physical change. With affine present, stop at 1e-6*g0 (meaningful); pure flex keeps tight 1e-7.
    mjtNum ctol = (naff > 0) ? 1e-6 : 1e-7;
    if (gnorm < ctol*g0 + 1e-9) break;
    // ONE solver: per-tree affine preconditioner factors (from inertia+ortho+contact), then the unified PCG.
    if (naff > 0) ipc_crbFactor(aff, naff, crbHp, crbP, crbD, crbA, crbK);
    if (Nna > 0) ipc_ic0(&sp, 1e-9);                          // incomplete-LDL^T factor (signed pivots)
    ipc_solveU(dx, grad, N, Nna, Nz, aff, naff, &sp, mdiag, ne, elems, estr, ccache, nacon,
               affH, crbD, crbA, crbK, crbS, crbDcp, crbGz, rcg, zcg, pcg, Hpv, usol, dxm, Hpm, crbAdj);
    mjtNum maxdx = 0;   // max free-vertex displacement of the Newton direction (for the lower-bound decay)
    for (int v=0; v < N/3; v++) { mjtNum n2 = dx[3*v]*dx[3*v]+dx[3*v+1]*dx[3*v+1]+dx[3*v+2]*dx[3*v+2];
                                  if (n2 > maxdx) maxdx = n2; }
    maxdx = sqrt(maxdx);
    // proper IPC line search: alpha bounded by the rigorous additive CCD over the candidate contacts
    // (alpha -> ~1 when nothing is closing, so the full Newton step resolves contacts; no pair can
    // cross), then Armijo backtrack on the energy (which iterates the candidate set, so it sees
    // contacts activating during the step). Replaces the arbitrary 0.4*ghat step-cap.
    mjtNum cap = ipc_ccd(m, d, x, dx, gv, ge, r, rad, nfv, fidx, cand, ncand, cgap);
    { mjtNum acap = ipc_affineContactCCD(m, d, aff, naff, x, dx); if (acap < cap) cap = acap; }
    { mjtNum acap = ipc_affineAffineCCD(m, aff, naff, x, dx); if (acap < cap) cap = acap; }
    // E0 = energy at the current x. Reuse the gaps ipc_try just cached for the barrier term, so no
    // candidate closest-point is recomputed here (the line-search trials below are at xn, so they do).
    mjtNum E0 = ipc_energyBase(m, npt, ne, elems, x, xtil, fidx, mass, h, aff, naff);
    for (int c=0; c < ncand; c++) if (cgap[c] > 0) {
      mjtNum ghc = ipc_conGhat(&cand[c], rad, ghat);
      if (cgap[c] < ghc) E0 += kappa*ipc_Bg(cgap[c], ghc);
    }
    E0 += ipc_affineContactEnergy(m, d, aff, naff, x, xold, kappa);
    E0 += ipc_affineAffineEnergy(m, aff, naff, x, xold, kappa);
    // line-search subset: a candidate can contribute to the barrier at xn = x + alpha*dx (alpha in
    // [0,cap], cap<=1) only if its gap can drop below ghat. Its gap drop over a full step is bounded
    // by the sum of |dx| over its flex vertices, so keep only c with cgap[c] < ghat + that bound.
    int nls = 0;
    for (int c=0; c < ncand; c++) {
      int vv[4], nvv; ipc_conVerts(&cand[c], vv, &nvv);
      mjtNum lub = 0;
      for (int q=0; q < nvv; q++) { int fq = fidx[vv[q]]; if (fq < 0) continue;
        lub += sqrt(dx[3*fq]*dx[3*fq] + dx[3*fq+1]*dx[3*fq+1] + dx[3*fq+2]*dx[3*fq+2]); }
      if (cgap[c] < ghat + lub) candLS[nls++] = cand[c];
    }
    mjtNum gdx = 0; for (int i=0; i < N; i++) gdx += grad[i]*dx[i];
    mjtNum alpha = cap;
    int lsok = 0;
    for (int ls=0; ls < 30; ls++) {
      for (int i=0; i < 3*npt; i++) xn[i] = x[i];
      for (int v=0; v < npt; v++) if (fidx[v] >= 0) { int fi = fidx[v];
        for (int c=0; c < 3; c++) xn[3*v+c] = x[3*v+c] + alpha*dx[3*fi+c]; }
      if (ipc_energy(m, d, npt, ne, elems, xn, xtil, fidx, mass, h, r, rad, ghat, kappa,
                     gv, ge, candLS, nls, aff, naff, xold) <= E0 + 1e-4*alpha*gdx) { lsok = 1; break; }
      alpha *= 0.5;
    }
    // stagnation guard: if the line search exhausts all backtracks without an energy decrease, the
    // (descent) Newton step cannot reduce the energy even at alpha -> 0, i.e. we are sitting at the
    // minimum to numerical precision. Stop instead of grinding out further no-progress iterations.
    if (!lsok) break;
    for (int i=0; i < 3*npt; i++) x[i] = xn[i];
    // keep cgap a valid lower bound: every gap can shrink by at most 4 vertices * the max vertex step
    mjtNum dgap = 4.0*alpha*maxdx;
    for (int c=0; c < ncand; c++) cgap[c] -= dgap;
  }
  // adapt kappa for the next step: stiffen if the min gap collapsed (barrier too soft), relax if it stayed
  // comfortably open. Wide hysteresis [0.2, 0.8] (as a fraction of each contact's own ghat_c) avoids
  // oscillation; gentle relax so it doesn't undo. Per-contact ratio so a thin participant in a tight
  // channel doesn't ratchet kappa just because its gap is small in absolute terms.
  // robust adaptation: ramp only when tightness is BROAD (>5% of active contacts deep in their barrier),
  // not on a single constrained outlier (one 4um string contact used to pin kappa at the cap and diverge);
  // decay when contacts are broadly open (>95% past 0.6 of their ghat_c); hold otherwise. The CCD already
  // guarantees non-penetration, so kappa only needs to keep the bulk well-conditioned, not chase one gap.
  if (ncand > 0 || naff > 0) {
    if (nact0 > 0 && nlt02*20 > nact0 && kappa < IPC_KAPPAMAX) {
      kappa *= 2.0; if (kappa > IPC_KAPPAMAX) kappa = IPC_KAPPAMAX;
    } else if ((nact0 == 0 || nlt06*20 <= nact0) && kappa > IPC_KAPPA0) {
      kappa *= 0.8; if (kappa < IPC_KAPPA0) kappa = IPC_KAPPA0;
    }
  }
  g_kappa = kappa;   // persist adapted kappa for the next step
  for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
    if (pcp[v]) continue;   // affine control point: pose recovered in the affine readback loop below
    int da = dofadr[v];
    int qa = qpadr[v];   // qpos by joint qposadr; qvel/accel by dof address
    // x/xold are world; the slide dofs are in the body-local frame -> map the world displacement back
    // through R^T (= d->xmat^T). For an unrotated body this is the identity (dp_local == dp_world).
    const mjtNum* R = d->xmat + 9*pbody[v];
    mjtNum dpw[3], dpl[3];
    for (int c=0; c < 3; c++) dpw[c] = x[3*v+c] - xold[3*v+c];
    mju_mulMatTVec3(dpl, R, dpw);
    for (int c=0; c < 3; c++) { d->qvel[da+c] = dpl[c]/h; d->qpos[qa+c] += dpl[c]; }
  }
  // affine bodies: recover qpos/qvel from each body's OWN 4 control points. Fit every body's rotation at
  // xold and x; a free body writes its full pose, a hinge body writes the angle about its axis as the body's
  // world twist minus its parent's (zero for a hinge to a static parent). Each body is read independently --
  // no shared control points -- so a coplanar/parallel joint configuration is no longer special.
  mjtNum* qoA = (mjtNum*) mju_malloc((naff > 0 ? 4*naff : 1)*sizeof(mjtNum));
  mjtNum* qnA = (mjtNum*) mju_malloc((naff > 0 ? 4*naff : 1)*sizeof(mjtNum));
  for (int a=0; a < naff; a++) {
    ipcAffine* af = &aff[a];
    mjtNum cpo[4][3], cpn[4][3];
    for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) { cpo[k][c] = xold[3*af->cp[k]+c]; cpn[k][c] = x[3*af->cp[k]+c]; }
    ipc_affineQuat(af, cpo, d->xquat+4*af->body, qoA+4*a);
    ipc_affineQuat(af, cpn, d->xquat+4*af->body, qnA+4*a);
  }
  for (int a=0; a < naff; a++) {
    ipcAffine* af = &aff[a];
    if (!af->isHinge) {
      mjtNum cp[4][3];
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) cp[k][c] = x[3*af->cp[k]+c];
      mjtNum pos_old[3], quat_old[4];
      for (int c=0; c < 3; c++) pos_old[c]  = d->qpos[af->qadr+c];   // start-of-step pose (qpos not yet
      for (int c=0; c < 4; c++) quat_old[c] = d->qpos[af->qadr+3+c]; // overwritten for this body)
      ipc_affineReadback(af, cp, h, pos_old, quat_old, d->qpos, d->qvel);
    } else {
      // relative rotation vector (body twist minus parent twist), projected onto the k hinge axes by
      // least-squares -> k angle increments. For k=1 this is just the component along the axis.
      int k = af->njnt;
      mjtNum wrel[3]; ipc_twistVec(qnA+4*a, qoA+4*a, wrel);
      if (af->jparent >= 0) { mjtNum wp[3]; ipc_twistVec(qnA+4*af->jparent, qoA+4*af->jparent, wp);
        for (int c=0; c < 3; c++) wrel[c] -= wp[c]; }
      mjtNum AtA[9] = {0}, Atb[3] = {0}, th[3] = {0};               // solve (A^T A) th = A^T wrel
      for (int i=0; i < k; i++) {
        for (int j=0; j < k; j++) { mjtNum s = 0; for (int c=0; c < 3; c++) s += af->jaxis[i][c]*af->jaxis[j][c]; AtA[3*i+j] = s; }
        mjtNum s = 0; for (int c=0; c < 3; c++) s += af->jaxis[i][c]*wrel[c]; Atb[i] = s;
      }
      if (k == 1) { th[0] = Atb[0]/AtA[0]; }
      else if (k == 2) { mjtNum det = AtA[0]*AtA[4] - AtA[1]*AtA[3];
        th[0] = (AtA[4]*Atb[0] - AtA[1]*Atb[1])/det; th[1] = (-AtA[3]*Atb[0] + AtA[0]*Atb[1])/det; }
      else { mjtNum Mi[9]; ipc_mat3inv(Mi, AtA); for (int i=0; i < 3; i++) { mjtNum s = 0; for (int j=0; j < 3; j++) s += Mi[3*i+j]*Atb[j]; th[i] = s; } }
      for (int i=0; i < k; i++) { d->qpos[af->qadr+i] += th[i]; d->qvel[af->vadr+i] = th[i]/h; }
    }
  }
  mju_free(qoA); mju_free(qnA);
  d->time += h;
  if (naff > 0) { mju_free(crbGz); mju_free(crbAdj); mju_free(crbHp); mju_free(crbHc);
                mju_free(crbP); mju_free(crbD); mju_free(crbA); mju_free(crbK); mju_free(crbS); mju_free(crbDcp); }
  if (Nna > 0) ipc_spFree(&sp);
  mju_free(escat);
  mju_free(dofadr); mju_free(qpadr); mju_free(fidx); mju_free(mass); mju_free(elems);
  mju_free(rad); mju_free(pbody); mju_free(pgeom); mju_free(pcp); mju_free(aff);
  mju_free(xpos_pred); mju_free(xquat_pred); mju_free(isAff); mju_free(bidx);
  mju_free(flist); mju_free(fxadr); mju_free(pt2vg); mju_free(pt2flex);
  mju_free(acon); mju_free(ccache); mju_free(cand); mju_free(cgap); mju_free(candLS);
  mju_free(gv); mju_free(ge); mju_free(estr); mju_free(affH);
  mju_free(x); mju_free(xtil); mju_free(xold); mju_free(xn);
  mju_free(grad); mju_free(dx); mju_free(mdiag);
  mju_free(rcg); mju_free(zcg); mju_free(pcg); mju_free(Hpv); mju_free(usol); mju_free(dxm); mju_free(Hpm);
}


// -------------------------------------------------------------------------------------------------
// Thin wrappers exposing the internal IPC kernels to the unit tests (engine_ipc_test.cc). These let
// the barrier and the geometry/contact distance functions be checked directly, without stepping a
// model. Not a supported API; kept here so the kernels themselves stay static.

// C-IPC offset barrier on surface gap g (support (0,ghat)): value, and optionally 1st/2nd derivative.
mjtNum mj_ipcBarrier(mjtNum g, mjtNum ghat, mjtNum* d1, mjtNum* d2) {
  if (g > 0 && g < ghat) {
    if (d1) *d1 = ipc_Bd(g, ghat);
    if (d2) *d2 = ipc_Bdd(g, ghat);
    return ipc_Bg(g, ghat);
  }
  if (d1) *d1 = 0;
  if (d2) *d2 = 0;
  return 0;
}

// point-triangle distance (closest point cp and barycentric weights w optional)
mjtNum mj_ipcPtTri(const mjtNum* p, const mjtNum* a, const mjtNum* b, const mjtNum* c) {
  mjtNum cp[3], w[3];
  return ipc_ptTri(p, a, b, c, cp, w);
}

// segment-segment distance
mjtNum mj_ipcSegSeg(const mjtNum* p1, const mjtNum* p2, const mjtNum* q1, const mjtNum* q2) {
  mjtNum cp1[3], cp2[3], st[2];
  return ipc_segSeg(p1, p2, q1, q2, cp1, cp2, st);
}

// signed distance (+ outward unit normal n) from geom gi's surface to world point x, at d's pose
mjtNum mj_ipcGeomDist(const mjModel* m, const mjData* d, int gi, const mjtNum* x, mjtNum* n) {
  return ipc_geomDist(m, gi, d->geom_xpos + 3*gi, d->geom_xmat + 9*gi, x, n, 1e30);
}

// world-space sharp vertices / edges of geom gi at d's pose (out sized by caller); returns the count
int mj_ipcGeomVerts(const mjModel* m, const mjData* d, int gi, mjtNum* out) {
  return ipc_geomVerts(m, gi, d->geom_xpos + 3*gi, d->geom_xmat + 9*gi, out);
}
int mj_ipcGeomEdges(const mjModel* m, const mjData* d, int gi, mjtNum* out) {
  return ipc_geomEdges(m, gi, d->geom_xpos + 3*gi, d->geom_xmat + 9*gi, out);
}
