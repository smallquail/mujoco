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

// This integrator implements barrier-free augmented-Lagrangian (AL) penetration-free contact
// (Li et al., arXiv:2512.12151): the log-barrier is replaced by an augmented Lagrangian
// with a per-pair multiplier + active-set, and intersection-freedom is maintained by advancing
// a CCD-bounded committed position. Flex-only: 2D flex nodal IPC (the rigid/affine path was removed).

#include "engine/engine_ipc.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>   // qsort (sparse Hessian pattern build)

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_forward.h"      // mj_Euler (fallback)
#include "engine/engine_support.h"      // mj_mulM, mj_integratePos, mj_differentiatePos (per-tree mass + manifold)
#include "engine/engine_core_smooth.h"  // mj_solveM, mj_kinematics, mj_comPos (per-tree M^-1, FK at trial q)
#include "engine/engine_core_util.h"    // mj_local2Global (anchor body-local -> world for the live gap)
#include "engine/engine_util_solve.h"   // mju_cholFactor, mju_cholSolve (dense Newton solve)
#include "engine/engine_util_blas.h"    // mju_dot3, mju_mulMatVec3
#include "engine/engine_util_spatial.h" // mju_cross (flex bending)
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

// Contact STANDOFF: every active contact rests at a small positive gap delta = min(ghc, IPC_DELTACAP). The AL
// multiplier holds the constraint there, so the MaxStepSize CCD always sees a positive gap (nonzero TOI, no lock)
// and the committed trajectory stays STRICTLY intersection-free (no penetration). delta is a small geometric skin:
// it shrinks for thin participants (ghc small -> thin cloth keeps its no-penetration guarantee) and is capped
// so thick participants don't carry a fat layer.
#define IPC_DELTACAP 0.001   // 1 mm standoff cap
#define IPC_CDAMP_FLEX 0.1   // flex-contact normal dashpot coeff (cde = IPC_CDAMP_FLEX*m/h^2)
#define IPC_SOFT_K 1.0       // soft flex-rigid penalty stiffness factor: k = IPC_SOFT_K*mass*ih2 (penetrate+recover; tunable)
// ---- augmented-Lagrangian (AL) solve (pure-flex path) parameters ----
#define IPC_MU_SCALE_FEM 5e7   // per-vertex AL stiffness: mu_v = mass*IPC_MU_SCALE_FEM*h^2
#define IPC_DECAY 0.3          // cnt-aging stiffness decay: scale = pow(IPC_DECAY,c)*mu
#define IPC_VEL_TOL 0.05       // newton velocity tolerance (m/s); abs dx tol = IPC_VEL_TOL*h (L-inf dx checker)
#define IPC_TOI_THRESH 0.1     // terminate when beta >= 1 - IPC_TOI_THRESH (feasibility threshold)
#define IPC_ALPHA_LB 1e-6      // advance xfree only if CCD alpha > this (alpha lower bound)
#define IPC_FLEX_MIN_ITER 1    // minimum Newton iterations. Terminate once beta is feasible
                               // (beta >= 1 - IPC_TOI_THRESH) AND (newton_iter+1 >= min_iter OR newton_converged). The
                               // persistent active set + per-element PSD projection make the single-Newton step sound.
#define IPC_ASET_AGE 25        // active-set update: evict a persistent pair once abs(cnt) > IPC_ASET_AGE
#define IPC_ASET_TOI 1e-6      // active-set update: admit a new broad-phase pair iff its CCD toi < 1 - IPC_ASET_TOI
static inline mjtNum ipc_off(mjtNum ghc) { return ghc < IPC_DELTACAP ? ghc : IPC_DELTACAP; }
static long g_nact = 0;   // [PROF] active (fc>0) contacts in the last inner assembly

// one active contact. type: 0 vertex-triangle self, 1 edge-edge self, 2 flex-vertex vs geom
// surface, 3 geom-corner vs flex-triangle, 4 geom-edge vs flex-edge. idx/gi meaning per type
// (see ipc_conGap). The geom side is static, so its features (gv/ge) are precomputed once per step.
// lam = AL multiplier (rides in copies). The contact is LINEARIZED at the intersection-free state xfree
// each outer iter (paper Eq.10): ld0 = gap(xfree), ln = normal, lcw[liv] = dg/dx weights at the involved free pts.
// Then c(x) = ld0 + sum_p lcw[p]*ln.(x[liv[p]]-xfree[liv[p]]) - delta is LINEAR in x -> exact constant contact
// Hessian (mu*grad d grad d^T, no grad^2 d) -> the inner Newton converges in ~1 step.
typedef struct { int type; int idx[4]; int gi; mjtNum lam;
                 mjtNum ld0, ln[3], lcw[4]; int liv[4], lniv;
                 int cnt; mjtNum s; } ipcCon;   // cnt = active-set state machine (0.3^c stiffness decay +
                 // aging); s = materialized AL slack for the d0 bake (slack update -> assemble -> lambda update un-bake)

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

// per-pair AL stiffness mu = min over the pair's flex/sphere vertices of mu_v.
// FORCE-FORM (this is the crux): the reference incremental-potential objective is K + h^2*Psi with K = 0.5*M*(x-xtil)^2
// (RAW mass), so its mu_v = mass*IPC_MU_SCALE_FEM*h^2 gives mu/inertia = IPC_MU_SCALE_FEM*h^2 = 200 (a strong penalty).
// OUR objective is that IP divided by h^2 -> inertia = mass/h^2, elastic force-form, so the FORCE-FORM mu is the IP mu
// divided by h^2 = mass*IPC_MU_SCALE_FEM (the h^2 cancels). Returning the IP-form mu*h^2 verbatim (the old bug) made the
// penalty h^4 = 1250x too SOFT vs our inertia -> contacts couldn't hold (the CCD did all the work -> FROZEN) and the
// dual-ascent step lam-=craw*mu was 1250x too small (-> SLOW). mu = IPC_MU_SCALE_FEM*mass restores mu/inertia = 200,
// consistent with the dashpot in the same block (cde = IPC_CDAMP_FLEX*mass*ih2, also force-form). REPLACES scalar kappa.
static mjtNum ipc_muPair(const ipcCon* con, const mjtNum* mass, mjtNum ih2) {
  (void)ih2;   // force-form mu has no explicit h factor (the h^2 cancelled against the IP-form objective)
  int vv[4], nvv; ipc_conVerts(con, vv, &nvv);
  // min over NONZERO masses: MuJoCo flex verts pinned to a rigid attachment carry mass 0 (their inertia is in the
  // rigid body); a plain min hits those -> mu=0 -> lam/mu = 0/0 = NaN. The reference FEM verts all have mass; MuJoCo guard.
  mjtNum mmin = 1e30;
  for (int q = 0; q < nvv; q++) { mjtNum mv = mass[vv[q]]; if (mv > 0 && mv < mmin) mmin = mv; }
  if (mmin >= 1e29) mmin = 1e-9;   // all involved verts massless (degenerate) -> tiny mu, no division blow-up
  return IPC_MU_SCALE_FEM*mmin;
}

// cnt -> decay exponent c (AL normal-contact aging): c = cnt>=0 ? cnt : max(-cnt-6, 0).
static inline int ipc_cntExp(int cnt) { return cnt >= 0 ? cnt : (-cnt-6 > 0 ? -cnt-6 : 0); }

// stable per-pair hash for the persistent cnt store: contact type + sorted vertex/feature indices.
static unsigned long ipc_pairHash(const ipcCon* con) {
  int id[4]; for (int k=0; k < 4; k++) id[k] = con->idx[k];
  for (int i=0; i < 3; i++) for (int j=0; j < 3-i; j++)   // sort idx ascending (order-independent key)
    if (id[j] > id[j+1]) { int t=id[j]; id[j]=id[j+1]; id[j+1]=t; }
  unsigned long hh = (unsigned long)con->type * 1000003ul + (unsigned long)(con->gi+1);
  for (int k=0; k < 4; k++) hh = hh*1000003ul + (unsigned long)(id[k]+1);
  return hh;
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
                      const int* fidx, const ipcCon* cand, int ncand, const mjtNum* cgap, const int* pt2flex,
                      int* approut) {
  mjtNum alpha = 1.0;
  if (approut) for (int c=0; c < ncand; c++) approut[c] = 0;   // Alg.3: per-pair "the proxy approaches this contact"
  for (int c=0; c < ncand; c++) {
    const ipcCon* con = &cand[c];
    // mean-removal (don't throttle COHERENT motion) is valid ONLY for a true SAME-FLEX self-contact. For INTER-flex
    // (e.g. drawstring vs bag) the two sides move independently, so removing the mean underestimates the closing
    // speed and lets one tunnel through the other. Gate it on same-flex (both sides in the same flex via pt2flex).
    int other = (con->type == 0) ? con->idx[1] : con->idx[2];
    int v[4], nv, self = (con->type <= 1) && (con->idx[0] < nfv) && (other < nfv)
                         && (pt2flex[con->idx[0]] == pt2flex[other]);
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
    mjtNum g0 = cgap[c];   // true gap at x (>= standoff delta at rest, so the CCD always has room: no lock)
    if (g0 <= 0) continue;   // already at/under the surface: the AL + energy own it
    if (l <= 0.8*g0) continue;   // full alpha=1 step shrinks gap by <= l, stays above the 20% floor
    if (approut) approut[c] = 1;   // reaches the bisection -> the proxy closes this pair's gap this step (Alg.3 add)
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
typedef struct { mjtNum n[3], cw[8], bdd; int f[8], nidx; } ipcCC;   // up to 8 involved free dofs
// SOFT flex-rigid contact pair: rigid sphere ri vs a flex triangle (type 0, idx=tri verts), a static geom
// (type 2, gi), or another sphere (type 5, idx[0]=rj). One-sided penalty only; never in the hard set/CCD.
typedef struct { int type, ri, idx[3], gi; } ipcSoft;

// One membrane element (2D triangle): the 3 flex vertices (global vg, free-dof fv, -1 if pinned),
// the FEM stiffness metric M (symmetric 3x3 over the element's 3 edges, read from flex_stiffness),
// and the rest squared edge lengths Lr2. We consume MuJoCo's P1 FEM membrane energy (engine_passive,
// Kharevych et al.): E = 1/4 e^T M e with e[edge] = L^2 - L0^2 (frame-invariant). The force is the
// gradient (matches mj_flexPassiveStretch exactly) and the Hessian is its Gauss-Newton part.
typedef struct { int vg[3], fv[3]; mjtNum M[6]; mjtNum Lr2[3];
                 mjtNum kD;        // Rayleigh damping coefficient = flex_damping/h (constant)
                 mjtNum ep[3];     // elongations e_a at the previous step (xold), refreshed each step
               } ipcElem;
// One bending "flap" (an interior edge + the two opposite triangle apexes): the 4 flex vertices (global vg,
// free-dof fv, -1 if pinned) and the constant 4x4 quadratic-bending operator Q (MuJoCo's precomputed
// flex_bending; Wardetzky/Garg "Discrete Quadratic Curvature"/"Cubic Shells"). Flat-rest assumption -> the
// curved-reference term b[17e+16] is dropped. Energy 1/2 x^T (Q (x) I3) x, force/Hessian = Q (x) I3 -- constant,
// PSD by construction (rank-1 Q = cos*stiffness*c c^T; concave-rest edges with trace(Q)<0 are skipped at setup).
typedef struct { int vg[4], fv[4]; mjtNum Q[16]; mjtNum b16; } ipcBend;
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


// assemble ONE held contact's AL gradient G =
// scale*d*d_grad and Hessian H = scale*d_grad d_grad^T (rank-1 PSD GN block, ccache->bdd=scale), where the
// SLACK is already folded into d. Linearized at xfree (ld0/ln/lcw/liv this iter), evaluated at the optimizer x:
//   d_raw(x) = (ld0 - delta) + sum_p lcw*ln.(x-xfree)        (the un-baked linearized gap c_raw)
//   d(x)     = d_raw(x) - con.s - con.lam/mu                 (slack-baked, == the slack-update d0 bake)
// scale = pow(IPC_DECAY, c)*mu_pair with c = ipc_cntExp(con.cnt). con.s set by ipc_updateSlack (loop step N2).
// Keep mj's one-sided normal dashpot cde (active+closing) as an mj stabilizer; ablate later.
static void ipc_try(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x, const mjtNum* xfree,
                    const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad, mjtNum ghat,
                    const mjtNum* xold, const mjtNum* mass, mjtNum ih2,
                    const int* fidx, mjtNum* grad,
                    ipcCon* acon, ipcCC* ccache, int* nacon, int amax, mjtNum* gout) {
  mjtNum mu = ipc_muPair(&con, mass, ih2);
  mjtNum craw = con.ld0 - ipc_off(ipc_conGhat(&con, rad, ghat));   // c_raw(x) (un-baked linearized gap)
  for (int p=0; p < con.lniv; p++) { int v = con.liv[p];
    for (int k=0; k < 3; k++) craw += con.lcw[p]*con.ln[k]*(x[3*v+k] - xfree[3*v+k]); }
  *gout = craw;   // linearized gap, for the inner active-set test (cgap < ghat)
  if (craw >= ghat || *nacon >= amax) return;   // beyond detection range -> not HELD; or list full
  mjtNum dd = craw - con.s - con.lam/mu;                 // slack-baked residual d: G=scale*d*d_grad
  int cexp = ipc_cntExp(con.cnt);
  mjtNum scale = mu;
  for (int e=0; e < cexp; e++) scale *= IPC_DECAY;       // scale = pow(IPC_DECAY, c)*mu_pair (decays G AND H)
  if (dd < 0) g_nact++;                                  // [PROF] count violated (slack-baked d<0)
  mjtNum cde = 0, dn = 0;
  if (dd <= 0) {                                  // ONE-SIDED normal dashpot: violated contacts only (mj stabilizer)
    for (int p=0; p < con.lniv; p++) { int v = con.liv[p];
      for (int k=0; k < 3; k++) dn += con.lcw[p]*con.ln[k]*(x[3*v+k] - xold[3*v+k]); }
    if (dn < 0) { mjtNum mmin = 1e30;
      for (int p=0; p < con.lniv; p++) { mjtNum mv = mass[con.liv[p]]; if (mv < mmin) mmin = mv; }
      cde = IPC_CDAMP_FLEX*mmin*ih2; }
  }
  ipcCC* cc = &ccache[*nacon];
  cc->bdd = scale + cde; cc->nidx = con.lniv;            // GN Hessian block = scale*d_grad d_grad^T (+ dashpot)
  for (int k=0; k < 3; k++) cc->n[k] = con.ln[k];
  mjtNum gco = scale*dd + cde*dn;                        // gradient coeff: scale*d (+ dashpot)
  for (int p=0; p < con.lniv; p++) { int fp = fidx[con.liv[p]];
    cc->cw[p] = con.lcw[p]; cc->f[p] = fp;
    if (fp < 0) continue;
    for (int i=0; i < 3; i++) grad[3*fp+i] += gco*con.lcw[p]*con.ln[i]; }
  acon[*nacon] = con;
  (*nacon)++;
}

// slack update (loop step N2, at the optimizer x): materialize the AL slack s = max(0, c_raw - lam/mu)
// for every held candidate so the subsequent assemble (ipc_try)/energy (ipc_energy) use the exact slack-baked
// d = c_raw - s - lam/mu, and the lambda update (ipc_flexLamUpdate) can un-bake it. c_raw is the LINEARIZED gap at x
// (ld0 set this iter by linearize at xfree). Order is FIXED: linearize -> slack update -> assemble -> ... -> lambda.
static void ipc_updateSlack(ipcCon* cand, int ncand, const int* held, const mjtNum* x, const mjtNum* xfree,
                            const mjtNum* rad, mjtNum ghat, const mjtNum* mass, mjtNum ih2) {
  for (int c=0; c < ncand; c++) {
    if (!held[c]) { cand[c].s = 0; continue; }
    ipcCon* con = &cand[c];
    mjtNum mu = ipc_muPair(con, mass, ih2);
    mjtNum craw = con->ld0 - ipc_off(ipc_conGhat(con, rad, ghat));   // c_raw(x) = (ld0-delta) + d_grad.(x-xfree)
    for (int p=0; p < con->lniv; p++) { int v = con->liv[p];
      for (int k=0; k < 3; k++) craw += con->lcw[p]*con->ln[k]*(x[3*v+k] - xfree[3*v+k]); }
    mjtNum t = craw - con->lam/mu;
    con->s = (t > 0) ? t : 0;        // s = max(0, c_raw - lam/mu); baked d = c_raw - s - lam/mu (in ipc_try/energy)
  }
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

// AL contact multiplier per FREE POINT (flex vertex / rigid sphere): the cross-step warm-start store for the
// ipc_try contacts, whose LIVE multiplier rides in ipcCon.lam. Seeded into cand[].lam at step start, sunk back
// after the step (binding = max over a contact's free-point participants). npt-sized.
static mjtNum* g_pal = NULL; static int g_palN = 0;
static long g_pcgN = 0;   // [PROF] PCG iterations (accumulated per step, printed + reset in MJ_IPC_PROF)

// per-pair cnt state machine (active-set aging + 0.3^c stiffness decay) persisted across the per-iter
// candidate re-query AND across steps. Two open-addressing hash tables keyed by ipc_pairHash (type+sorted idx+gi):
// g_cntOld* is read-only this step (last step's final cnts), g_cntNew* is written this step. They swap at step
// start (g_cntStepBegin) so the table never grows unbounded as contact identities churn (string-in-bag). A
// re-queried/new pair re-hydrates cnt from OLD (miss -> 0); after each update_lambda the cnt is sunk into NEW.
static unsigned long *g_cntKeyO = NULL, *g_cntKeyN = NULL;
static int *g_cntValO = NULL, *g_cntValN = NULL;
static int g_cntCap = 0, g_cntUsedN = 0;
static void ipc_cntStepBegin(int candmax) {        // size to ~4x the candidate ceiling (load factor < 0.25); swap+clear NEW
  int cap = 1; while (cap < 4*candmax + 16) cap <<= 1;
  if (cap != g_cntCap) {
    mju_free(g_cntKeyO); mju_free(g_cntKeyN); mju_free(g_cntValO); mju_free(g_cntValN); g_cntCap = cap;
    g_cntKeyO = (unsigned long*) mju_malloc((size_t)cap*sizeof(unsigned long));
    g_cntKeyN = (unsigned long*) mju_malloc((size_t)cap*sizeof(unsigned long));
    g_cntValO = (int*) mju_malloc((size_t)cap*sizeof(int));
    g_cntValN = (int*) mju_malloc((size_t)cap*sizeof(int));
    for (int i=0; i < cap; i++) { g_cntKeyO[i] = 0; g_cntValO[i] = 0; }
  } else {
    unsigned long* tk = g_cntKeyO; g_cntKeyO = g_cntKeyN; g_cntKeyN = tk;   // NEW (just written) becomes OLD
    int* tv = g_cntValO; g_cntValO = g_cntValN; g_cntValN = tv;
  }
  for (int i=0; i < g_cntCap; i++) { g_cntKeyN[i] = 0; g_cntValN[i] = 0; }  // key 0 = empty slot
  g_cntUsedN = 0;
}
static int ipc_cntGet(unsigned long key) {         // re-hydrate cnt for a pair from OLD (miss -> 0)
  if (g_cntCap == 0 || key == 0) return 0;
  unsigned long mask = (unsigned long)(g_cntCap - 1), h = key & mask;
  for (int probe=0; probe < g_cntCap; probe++) {
    if (g_cntKeyO[h] == 0) return 0;
    if (g_cntKeyO[h] == key) return g_cntValO[h];
    h = (h + 1) & mask;
  }
  return 0;
}
static void ipc_cntSet(unsigned long key, int val) {   // sink updated cnt into NEW (insert/overwrite)
  if (g_cntCap == 0 || key == 0) return;
  unsigned long mask = (unsigned long)(g_cntCap - 1), h = key & mask;
  for (int probe=0; probe < g_cntCap; probe++) {
    if (g_cntKeyN[h] == 0) {
      if (g_cntUsedN >= g_cntCap/2) return;   // table near full (shouldn't happen at <0.25 load): drop, harmless
      g_cntKeyN[h] = key; g_cntValN[h] = val; g_cntUsedN++; return;
    }
    if (g_cntKeyN[h] == key) { g_cntValN[h] = val; return; }
    h = (h + 1) & mask;
  }
}

// lambda update + cnt state machine (loop step N8a, at the optimizer x): for every held candidate,
// un-bake d to the raw linearized gap c_raw, then the two-branch AL dual update (the slack condition c_raw>lam/mu)
// with the cnt aging machine (using the lambda value from BEFORE this update):
//   if (c_raw - lam/mu > 0)  { lam = 0;  cnt += (cnt>=0 ? +1 : -1); }                  // inactive (slack>0)
//   else                     { lam -= c_raw*mu;  cnt = (cnt==0 || cnt>5) ? 0 : -1; }   // active: lam grows
// (NO max(0,.) clamp.) Then SINK lam into the per-free-point store g_pal for next-step warm start. cnt rides in
// cand[c].cnt and is persisted per-pair by the caller.
static void ipc_flexLamUpdate(const mjModel* m, const mjData* d, const mjtNum* x, const mjtNum* xfree, const mjtNum* gv,
                              const mjtNum* ge, mjtNum r, const mjtNum* rad, mjtNum ghat,
                              const mjtNum* mass, mjtNum ih2,
                              ipcCon* cand, int ncand, const int* held, mjtNum* g_pal, int npt) {
  for (int c=0; c < ncand; c++) {
    if (!held[c]) { cand[c].lam = 0; continue; }
    ipcCon* con = &cand[c];
    mjtNum mu = ipc_muPair(con, mass, ih2);
    mjtNum lam0 = con->lam;                                          // lambda BEFORE this update (the dual update reads pre-update)
    mjtNum craw = con->ld0 - ipc_off(ipc_conGhat(con, rad, ghat));   // c_raw(x) (un-baked: == baked_d + s + lam/mu)
    for (int p=0; p < con->lniv; p++) { int v = con->liv[p];
      for (int k=0; k < 3; k++) craw += con->lcw[p]*con->ln[k]*(x[3*v+k] - xfree[3*v+k]); }
    if (craw - lam0/mu > 0) {                        // inactive (slack s>0): multiplier off, age the inactive counter
      con->lam = 0;
      con->cnt += (con->cnt >= 0) ? 1 : -1;
    } else {                                         // active: ascend lam by c_raw*mu (grows since c_raw<lam/mu)
      con->lam = lam0 - craw*mu;
      con->cnt = (con->cnt == 0 || con->cnt > 5) ? 0 : -1;
    }
  }
  for (int p=0; p < npt; p++) g_pal[p] = 0;            // sink: per-free-point binding multiplier (warm-start src)
  for (int c=0; c < ncand; c++) { if (cand[c].lam <= 0) continue;
    int vv[4], nvv; ipc_conVerts(&cand[c], vv, &nvv);
    for (int q=0; q < nvv; q++) if (cand[c].lam > g_pal[vv[q]]) g_pal[vv[q]] = cand[c].lam;
  }
}

// active-set update (MERGE/AGING of the persistent active set).
// Inputs: the existing persistent set aset[0..*naset) (cnt rides in each ipcCon), and the fresh broad-phase
// candidates cand[0..ncand) with a per-candidate "closing this step" admission flag cadmit[c] (== the
// CCD time-of-impact < 1-1e-6). Rule:
//   - KEEP an existing aset pair iff abs(cnt) <= IPC_ASET_AGE (else evict).
//   - ADD a new broad-phase candidate iff cadmit[c] AND its pairHash is not already present (dedup by 64-bit-ish
//     pairHash). New entries seed cnt=0, lam=0; we re-hydrate cnt from the cross-step store and lam from
//     the per-point warm-start g_pal so the cnt aging and AL multiplier survive across steps (our persistence path).
// The merged set is written back into aset/*naset. The new merged set's fields (ld0/ln/.../s) are refreshed by the
// next iter's linearize_constraints + update_slack, so only type/idx/gi/lam/cnt need to be carried here.
static void ipc_mergeActiveSet(ipcCon* aset, int* naset, const ipcCon* cand, int ncand, const int* cadmit,
                               ipcCon* amerge, const mjtNum* g_pal, int candmax) {
  // dedup hash over the merged set: open addressing keyed by pairHash (key 0 = empty slot; a true hash of 0 is
  // astronomically unlikely and at worst causes one duplicate, harmless). Presence-only -> no value array needed.
  int cap = 1; while (cap < 4*(*naset + ncand) + 16) cap <<= 1;
  static unsigned long* mkey = NULL; static int mcap = 0;
  if (cap > mcap) { mju_free(mkey); mcap = cap;
    mkey = (unsigned long*) mju_malloc((size_t)cap*sizeof(unsigned long)); }
  unsigned long mask = (unsigned long)(cap - 1);
  for (int i=0; i < cap; i++) mkey[i] = 0;
  int nm = 0;
  // 1) keep existing pairs with abs(cnt) <= IPC_ASET_AGE (aging eviction). MJ_IPC_NOMERGE=1 ablates -> per-iter set.
  if (!getenv("MJ_IPC_NOMERGE"))
  for (int c=0; c < *naset; c++) {
    int cnt = aset[c].cnt; if ((cnt < 0 ? -cnt : cnt) > IPC_ASET_AGE) continue;
    unsigned long key = ipc_pairHash(&aset[c]), h = key & mask;
    while (mkey[h] != 0) { if (mkey[h] == key) break; h = (h + 1) & mask; }
    if (mkey[h] == key) continue;   // already present (shouldn't happen within the existing set, but safe)
    mkey[h] = key; amerge[nm++] = aset[c];
  }
  // 2) add new admitted broad-phase candidates not already present
  for (int c=0; c < ncand; c++) {
    if (!cadmit[c] || nm >= candmax) continue;
    unsigned long key = ipc_pairHash(&cand[c]), h = key & mask;
    while (mkey[h] != 0) { if (mkey[h] == key) break; h = (h + 1) & mask; }
    if (mkey[h] == key) continue;   // dedup: pair already in the merged set
    ipcCon con = cand[c];
    int vv[4], nvv; ipc_conVerts(&con, vv, &nvv);   // warm-start lam from the per-point binding multiplier
    mjtNum s = 0; for (int q=0; q < nvv; q++) if (g_pal[vv[q]] > s) s = g_pal[vv[q]];
    con.lam = s; con.cnt = ipc_cntGet(ipc_pairHash(&con)); con.s = 0;   // re-hydrate cnt across steps
    mkey[h] = key; amerge[nm++] = con;
  }
  for (int c=0; c < nm; c++) aset[c] = amerge[c];
  *naset = nm;
}

static inline mjtNum ipc_fmin(mjtNum a, mjtNum b) { return a < b ? a : b; }

static void ipc_applyH(const mjtNum* p, mjtNum* Hp, int N, const mjtNum* mdiag,
                       int nelem, const ipcElem* elems, const mjtNum* estr,
                       int nbe, const ipcBend* bends,
                       const ipcCC* ccache, int nacon) {
  for (int i=0; i < N; i++) Hp[i] = mdiag[i]*p[i];
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
      mjtNum me = es[9+a] > 0.0 ? es[9+a] : 0.0;           // (b) clamp compressive geom -> SPD operator
                                                           // (match the assembled max(Me,0) clamp; was full Me)
      int p0 = el->fv[ipc_eedge[a][0]], p1 = el->fv[ipc_eedge[a][1]];
      if (p0 >= 0) for (int k=0; k < 3; k++) Hp[3*p0+k] += ca*es[3*a+k] + me*rel[a][k];
      if (p1 >= 0) for (int k=0; k < 3; k++) Hp[3*p1+k] -= ca*es[3*a+k] + me*rel[a][k];
    }
  }
  for (int bt=0; bt < nbe; bt++) {                         // bending: constant 4x4 Q (x) I3 per flap (matrix-free)
    const ipcBend* bn = &bends[bt]; const mjtNum* Q = bn->Q;
    for (int i=0; i < 4; i++) { int fi = bn->fv[i]; if (fi < 0) continue;
      for (int k=0; k < 3; k++) { mjtNum s = 0;
        for (int j=0; j < 4; j++) { int fj = bn->fv[j]; if (fj < 0) continue; s += Q[4*i+j]*p[3*fj+k]; }
        Hp[3*fi+k] += s; }
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

// sparse Hessian for the block-Jacobi preconditioner: per-vertex 3x3 diagonal blocks only (lower-tri CSR,
// 6 entries/vertex). Built by ipc_spBuild below mj_IPC; ipc_jacobiApply reads these blocks. Inter-vertex
// couplings are applied matrix-free in ipc_applyH, so they are not stored (IC0, their only consumer, is gone).
typedef struct {
  int N, nnz;
  int *rownnz, *rowadr, *colind;          // lower-tri CSR pattern
  mjtNum *val;                            // H values (block-Jacobi reads the 3x3 diagonal blocks)
} ipcSparse;
static void ipc_jacobiApply(const ipcSparse* sp, mjtNum* z, const mjtNum* r);  // per-vertex 3x3 block-Jacobi (below)

// UNIFIED PCG context: one solver loop (ipc_solveU) serves both the dense-packed flex/sphere DOFs and the
// articulated trees, dispatched per family. FLEX (rigid==0): the kinetic is the diagonal mdiag, the Hessian
// apply/precond are the EXISTING ipc_applyH/ipc_jacobiApply (byte-identical -- the dense 3*nfree packing is kept).
// ARTICULATED (rigid==1): the kinetic block is ih2*M applied via mj_mulM, preconditioned by its exact inverse
// (ih2*M)^-1 = h2*M^-1 via mj_solveM, plus the one-sided contact rank-1 terms (active set acti, dir cb, stiff kvec).
typedef struct {
  int rigid;
  const ipcSparse* sp; const mjtNum* mdiag;                  // flex precond + kinetic diagonal
  int nelem; const ipcElem* elems; const mjtNum* estr; int nbe; const ipcBend* bends;  // flex elastic
  const ipcCC* ccache; int nacon;                            // flex contact
  const mjModel* m; mjData* d; mjtNum ih2, h2;               // articulated kinetic / precond (mj_mulM / mj_solveM)
  int na; const int* acti; const mjtNum* cb; const mjtNum* kvec;  // articulated active contacts
} ipcCtx;

static void ipc_apply(const mjtNum* p, mjtNum* Hp, int N, const ipcCtx* c) {
  if (!c->rigid) { ipc_applyH(p, Hp, N, c->mdiag, c->nelem, c->elems, c->estr, c->nbe, c->bends, c->ccache, c->nacon); return; }
  mj_mulM(c->m, c->d, Hp, p); for (int i=0; i < N; i++) Hp[i] *= c->ih2;          // ih2*M*p (M frozen at q_n)
  for (int a=0; a < c->na; a++) { const mjtNum* b = c->cb + c->acti[a]*N;
    mjtNum ks = c->kvec[c->acti[a]]*mju_dot(b, p, N); for (int i=0; i < N; i++) Hp[i] += ks*b[i]; }
}
static void ipc_precond(mjtNum* z, const mjtNum* r, int N, const ipcCtx* c) {
  if (!c->rigid) { ipc_jacobiApply(c->sp, z, r); return; }
  mj_solveM(c->m, c->d, z, r, 1); for (int i=0; i < N; i++) z[i] *= c->h2;        // (ih2*M)^-1 = h2*M^-1
}

// matrix-free preconditioned CG, RHS r = -grad. Writes the Newton direction dx[N] = -H^-1 grad. Apply + precond
// dispatch via ctx; tol is the relative residual-norm-squared stop. Flex passes tol=1e-8 + the flex ctx -> this is
// byte-identical to the former pure-flex PCG (same arithmetic, same operation order).
static int ipc_solveU(mjtNum* dx, const mjtNum* grad, int N, const ipcCtx* ctx, mjtNum tol,
                      mjtNum* r, mjtNum* z, mjtNum* p, mjtNum* Hp, mjtNum* usol) {
  for (int i=0; i < N; i++) r[i] = -grad[i];
  for (int i=0; i < N; i++) usol[i] = 0;
  mjtNum r0 = 0; for (int i=0; i < N; i++) r0 += r[i]*r[i];
  if (r0 < 1e-30) { for (int i=0; i < N; i++) dx[i] = 0; return 0; }
  ipc_precond(z, r, N, ctx);
  mjtNum rz = 0; for (int i=0; i < N; i++) { p[i] = z[i]; rz += r[i]*z[i]; }
  int it = 0;
  for (; it < 200; it++) {
    ipc_apply(p, Hp, N, ctx);
    mjtNum pHp = 0; for (int i=0; i < N; i++) pHp += p[i]*Hp[i];
    if (pHp <= 1e-30) break;
    mjtNum alpha = rz/pHp, rr = 0;
    for (int i=0; i < N; i++) { usol[i] += alpha*p[i]; r[i] -= alpha*Hp[i]; rr += r[i]*r[i]; }
    if (rr < tol*r0) break;
    g_pcgN += 1;   // [PROF] total PCG iterations this step
    ipc_precond(z, r, N, ctx);
    mjtNum rznew = 0; for (int i=0; i < N; i++) rznew += r[i]*z[i];
    mjtNum beta = rznew/rz; rz = rznew;
    for (int i=0; i < N; i++) p[i] = z[i] + beta*p[i];
  }
  for (int i=0; i < N; i++) dx[i] = usol[i];
  return it;
}

// IPC incremental-potential energy: inertia + edge-stretch penalty + AL contact merit over the
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

// quadratic bending energy: sum over flaps of 1/2 x^T (Q (x) I3) x (Q the constant per-edge bending operator).
static mjtNum ipc_bendEnergy(int nbe, const ipcBend* bends, const mjtNum* x) {
  mjtNum E = 0;
  for (int t=0; t < nbe; t++) {
    const ipcBend* bn = &bends[t]; const mjtNum* Q = bn->Q;
    for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) { mjtNum q = Q[4*i+j]; if (q == 0) continue;
      for (int k=0; k < 3; k++) E += 0.5*q*x[3*bn->vg[i]+k]*x[3*bn->vg[j]+k]; }
    if (bn->b16 != 0) {   // curved-reference: E += b16 * det[ed0,ed1,ed2], ed_a = x[v_{a+1}] - x[v0] (gradient = b16*frc)
      const mjtNum* x0 = x + 3*bn->vg[0]; mjtNum ed[3][3];
      for (int a=0; a < 3; a++) { const mjtNum* xa = x + 3*bn->vg[a+1];
        for (int k=0; k < 3; k++) ed[a][k] = xa[k] - x0[k]; }
      mjtNum cr[3]; mju_cross(cr, ed[1], ed[2]);   // ed1 x ed2
      E += bn->b16 * (ed[0][0]*cr[0] + ed[0][1]*cr[1] + ed[0][2]*cr[2]);
    }
  }
  return E;
}

static mjtNum ipc_energy(const mjModel* m, const mjData* d, int nfv, int nelem, const ipcElem* elems,
                         int nbe, const ipcBend* bends,
                         const mjtNum* x, const mjtNum* xtil, const int* fidx, const mjtNum* mass,
                         mjtNum h, mjtNum r, const mjtNum* rad, mjtNum ghat,
                         const mjtNum* gv, const mjtNum* ge, const ipcCon* acon, int nacon,
                         const mjtNum* xold, const mjtNum* xfree,
                         int nrigid, const mjtNum* rmass) {
  mjtNum E = 0, ih2 = 1.0/(h*h);
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum mh = mass[v]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  for (int i=0; i < nrigid; i++) {              // rigid kinetic energy (state slot nfv+i); contact-free here
    int v = nfv + i; mjtNum mh = rmass[i]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  E += ipc_stretchEnergy(nelem, elems, x);
  E += ipc_bendEnergy(nbe, bends, x);
  for (int c=0; c < nacon; c++) {
    // AL penalty energy E = 0.5*scale*d^2 over the slack-baked d: exact parity with
    // ipc_try's gradient. d = c_raw - s - lam/mu, scale = pow(IPC_DECAY, cnt-exp)*mu_pair. Keep mj's dashpot energy.
    mjtNum mu = ipc_muPair(&acon[c], mass, ih2);
    mjtNum craw = acon[c].ld0 - ipc_off(ipc_conGhat(&acon[c], rad, ghat));
    for (int p=0; p < acon[c].lniv; p++) { int v = acon[c].liv[p];
      for (int k=0; k < 3; k++) craw += acon[c].lcw[p]*acon[c].ln[k]*(x[3*v+k] - xfree[3*v+k]); }
    mjtNum dd = craw - acon[c].s - acon[c].lam/mu;
    int cexp = ipc_cntExp(acon[c].cnt); mjtNum scale = mu;
    for (int e=0; e < cexp; e++) scale *= IPC_DECAY;
    E += 0.5*scale*dd*dd;
    mjtNum dn = 0;                                          // matching one-sided dashpot energy (see ipc_try)
    for (int p=0; p < acon[c].lniv; p++) { int v = acon[c].liv[p];
      for (int k=0; k < 3; k++) dn += acon[c].lcw[p]*acon[c].ln[k]*(x[3*v+k] - xold[3*v+k]); }
    if (dd <= 0 && dn < 0) { mjtNum mmin = 1e30;
      for (int p=0; p < acon[c].lniv; p++) { mjtNum mv = mass[acon[c].liv[p]]; if (mv < mmin) mmin = mv; }
      E += 0.5*(IPC_CDAMP_FLEX*mmin/(h*h))*dn*dn; }
  }
  return E;
}

// SOFT flex-rigid contact assembly -- the penalty analogue of ipc_try. One-sided spring active only on
// penetration (gap<0): grad += k*gap*(cw (x) n); GN Hessian k*(cw (x) n)(cw (x) n)^T into the SHARED ccache,
// so ipc_applyH + the block-Jacobi pick it up. The rigid sphere ri uses state slot nfv+ri (center) and solver
// index nfree_flex+ri; flex verts use fidx. NEVER added to the active set / CCD -> the sphere may dip in and
// the spring recovers it over the next steps. k = IPC_SOFT_K*mass*ih2 (moderate: ~the kinetic Hessian).
static void ipc_softTry(const ipcSoft* s, const mjModel* m, const mjData* d, const mjtNum* x, int nfv,
                        const mjtNum* rrad, const mjtNum* rad, const mjtNum* rmass, mjtNum ih2, int nfree_flex,
                        const int* fidx, mjtNum* grad, ipcCC* ccache, int* nacon, int amax) {
  if (*nacon >= amax) return;
  const mjtNum* ci = &x[3*(nfv + s->ri)];
  mjtNum n[3], cw[4], gap, k = IPC_SOFT_K*rmass[s->ri]*ih2; int f[4], ni;
  if (s->type == 0) {                                            // sphere vs flex triangle
    int A=s->idx[0], B=s->idx[1], C=s->idx[2];
    mjtNum cp[3], w[3], dd = ipc_ptTri(ci, &x[3*A], &x[3*B], &x[3*C], cp, w);
    for (int c=0; c < 3; c++) n[c] = (ci[c]-cp[c])/dd;
    gap = dd - (rrad[s->ri] + rad[A]);
    f[0]=nfree_flex+s->ri; cw[0]=1; f[1]=fidx[A]; cw[1]=-w[0]; f[2]=fidx[B]; cw[2]=-w[1]; f[3]=fidx[C]; cw[3]=-w[2]; ni=4;
  } else if (s->type == 2) {                                     // sphere vs static geom
    int gi=s->gi;
    mjtNum dd = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, ci, n, rrad[s->ri]+1.0);
    gap = dd - rrad[s->ri];
    f[0]=nfree_flex+s->ri; cw[0]=1; ni=1;
  } else {                                                       // sphere vs sphere
    int rj=s->idx[0]; const mjtNum* cj = &x[3*(nfv+rj)]; mjtNum dv[3], dd=0;
    for (int c=0; c < 3; c++) { dv[c]=ci[c]-cj[c]; dd+=dv[c]*dv[c]; } dd=mju_sqrt(dd);
    for (int c=0; c < 3; c++) n[c]=dv[c]/dd;
    gap = dd - (rrad[s->ri]+rrad[rj]);
    f[0]=nfree_flex+s->ri; cw[0]=1; f[1]=nfree_flex+rj; cw[1]=-1; ni=2;
    mjtNum mo = rmass[rj]; if (mo < rmass[s->ri]) k = IPC_SOFT_K*mo*ih2;
  }
  if (gap >= 0) return;                                          // penalty only on penetration
  mjtNum gco = k*gap;
  ipcCC* cc = &ccache[*nacon]; cc->bdd = k; cc->nidx = ni;
  for (int c=0; c < 3; c++) cc->n[c] = n[c];
  for (int p=0; p < ni; p++) { cc->cw[p]=cw[p]; cc->f[p]=f[p];
    if (f[p] < 0) continue;
    for (int c=0; c < 3; c++) grad[3*f[p]+c] += gco*cw[p]*n[c]; }
  (*nacon)++;
}

// SOFT flex-rigid penalty energy (line-search parity with ipc_softTry): 0.5*k*gap^2 over penetrating pairs.
static mjtNum ipc_softEnergy(const ipcSoft* soft, int nsoft, const mjModel* m, const mjData* d,
                             const mjtNum* x, int nfv, const mjtNum* rrad, const mjtNum* rad,
                             const mjtNum* rmass, mjtNum ih2) {
  mjtNum E = 0;
  for (int si=0; si < nsoft; si++) { const ipcSoft* s = &soft[si];
    const mjtNum* ci = &x[3*(nfv+s->ri)]; mjtNum gap, k = IPC_SOFT_K*rmass[s->ri]*ih2;
    if (s->type == 0) { int A=s->idx[0], B=s->idx[1], C=s->idx[2];
      mjtNum cp[3], w[3], dd = ipc_ptTri(ci, &x[3*A], &x[3*B], &x[3*C], cp, w); gap = dd - (rrad[s->ri]+rad[A]);
    } else if (s->type == 2) { int gi=s->gi; mjtNum nn[3];
      mjtNum dd = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, ci, nn, rrad[s->ri]+1.0); gap = dd - rrad[s->ri];
    } else { int rj=s->idx[0]; const mjtNum* cj = &x[3*(nfv+rj)]; mjtNum dd=0;
      for (int c=0; c < 3; c++) { mjtNum t=ci[c]-cj[c]; dd+=t*t; } dd=mju_sqrt(dd); gap = dd - (rrad[s->ri]+rrad[rj]);
      mjtNum mo = rmass[rj]; if (mo < rmass[s->ri]) k = IPC_SOFT_K*mo*ih2; }
    if (gap < 0) E += 0.5*k*gap*gap;
  }
  return E;
}

// append a candidate contact if its gap at x is below the (margin-inflated) detection threshold
static void ipc_addCand(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x,
                        const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad, mjtNum thresh,
                        const mjtNum* dfrom, const mjtNum* dto, mjtNum ghat,
                        ipcCon* cand, int* nc, int candmax) {
  if (*nc >= candmax) return;
  mjtNum n[3], cw[4]; int idv[4], nidx;
  mjtNum g = ipc_conGap(&con, m, d, x, gv, ge, r, rad, n, idv, cw, &nidx, thresh);
  if (g <= 0 || g >= thresh) return;
  // closing-bound prune: over the step the gap changes by at most |sum_p cw[p]*(dto-dfrom)[idv[p]]|
  // (Cauchy-Schwarz, |n|=1), so a pair beyond its per-contact ghc + that bound cannot become active this
  // step -> drop it. Replaces the crude GLOBAL 4*maxdisp band (which inflated by the fastest vertex anywhere,
  // flooding correlated bulk motion like a settling bag+string). No-tunnel safe: the per-outer re-query at
  // xfree (dfrom=xfree, dto=x) recaptures any pair whose closest feature flips under the inner step.
  mjtNum rel[3] = {0,0,0};
  for (int p=0; p < nidx; p++) { int vp = idv[p]; if (vp < 0) continue;
    for (int c=0; c < 3; c++) rel[c] += cw[p]*(dto[3*vp+c] - dfrom[3*vp+c]); }
  if (g < ipc_conGhat(&con, rad, ghat) + sqrt(mju_dot3(rel, rel))) cand[(*nc)++] = con;
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
                          mjtNum threshGeom, mjtNum maxdisp, const mjtNum* dfrom, const mjtNum* dto, mjtNum ghat,
                          int nfv, int npt, const int* fidx,
                          const int* flist, const int* fxadr, int nfd, const int* pt2flex,
                          ipcCon* cand, int candmax) {
  (void)npt;   // increment A: candidates are flex-only (npt==nfv); rigid bodies carry no hard contact
  int nc = 0;
  for (int gi=0; gi < m->ngeom; gi++) {                                      // free point (flex vert) vs geom
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
    for (int v=0; v < nfv; v++) {                       // flex verts only (rigid bodies carry no hard barrier)
      if (fidx[v] < 0) continue;
      mjtNum marg = thresh + rad[v];
      if (!isplane) {                                                                  // world-AABB cull
        if (mju_abs(x[3*v]-wc[0]) > wh[0]+marg || mju_abs(x[3*v+1]-wc[1]) > wh[1]+marg ||
            mju_abs(x[3*v+2]-wc[2]) > wh[2]+marg) continue;
      }
      ipcCon con = {2, {v, 0, 0, 0}, gi};
      ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, dfrom, dto, ghat, cand, &nc, candmax);
    }
  }

  // (Rigid sphere-sphere and sphere-vs-flex HARD contacts removed in increment A: rigid bodies live in their
  // own solver block and carry no AL/CCD contact yet. Soft flex-rigid contact is a separate next increment.)

  // ---- BVH-based candidates over ALL dim-2 flexes: geom-feature and flex-vs-flex VT/EE.
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

    // (rigid sphere-vs-flex-triangle hard contact removed in increment A -- see note above.)
    // geom-corner vs flex triangle (type 3); geom is static (one-sided) -> tighter threshGeom (the
    // convex-decomposition bin's ~1600 edges otherwise overflow candmax and drop the bag-bin contacts).
    mjtNum qhvG[3] = {threshGeom+rk, threshGeom+rk, threshGeom+rk};
    for (int c=0; c < ngv; c++) {
      int n = ipc_bvhBox(m, d, fk, &gv[3*c], qhvG, stk, outel, ne_k);
      for (int i=0; i < n; i++) { int e = outel[i];
        ipcCon con = {3, {c, off_k+el_k[3*e], off_k+el_k[3*e+1], off_k+el_k[3*e+2]}, -1};
        ipc_addCand(con, m, d, x, gv, ge, r, rad, threshGeom, dfrom, dto, ghat, cand, &nc, candmax); }
    }
    // geom-edge vs flex edge (type 4); dedup the shared triangle edges per query via stampG.
    for (int c=0; c < nge; c++) {
      const mjtNum* p0 = &ge[6*c]; const mjtNum* p1 = &ge[6*c+3]; mjtNum qc[3], qh[3];
      for (int kk=0; kk<3; kk++) { qc[kk]=0.5*(p0[kk]+p1[kk]); qh[kk]=0.5*mju_abs(p1[kk]-p0[kk])+threshGeom+rk; }
      int n = ipc_bvhBox(m, d, fk, qc, qh, stk, outel, ne_k); qid++;
      for (int i=0; i < n; i++) { int e = outel[i];
        for (int j=0; j<3; j++) { int e2 = eme_k[3*e+j]; if (stampG[e2]==qid) continue; stampG[e2]=qid;
          ipcCon con = {4, {c, off_k+m->flex_edge[2*(ea_k+e2)], off_k+m->flex_edge[2*(ea_k+e2)+1], 0}, -1};
          ipc_addCand(con, m, d, x, gv, ge, r, rad, threshGeom, dfrom, dto, ghat, cand, &nc, candmax); } }
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
        ipc_addCand(con, m, d, x, gv, ge, r, rad, thv, dfrom, dto, ghat, cand, &nc, candmax); }
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
            ipc_addCand(con, m, d, x, gv, ge, r, rad, the, dfrom, dto, ghat, cand, &nc, candmax); } }
      }
    }
  }
  mju_free(stk); mju_free(outel); mju_free(stampG);
  return nc;
}

// ---------------------------------------------------------------------------------------------------
// Sparse Hessian for the block-Jacobi preconditioner: per-vertex 3x3 diagonal blocks ONLY (lower-tri CSR,
// 6 entries/vertex). ipc_jacobiApply reads exactly these blocks; the inter-vertex (membrane/contact)
// couplings are applied matrix-free in ipc_applyH, so they need no storage now that IC0 (their only
// consumer) is gone. Building just the block-diagonal pattern is O(N) -- no key list, no qsort, no merge.
static void ipc_spBuild(ipcSparse* sp, int N) {
  sp->N = N;
  int nv = N/3, nnz = 6*nv;   // N == 3*nfree; 6 lower-tri entries per 3x3 vertex block
  sp->nnz = nnz;
  sp->rownnz = (int*) mju_malloc(N*sizeof(int));
  sp->rowadr = (int*) mju_malloc(N*sizeof(int));
  sp->colind = (int*) mju_malloc((nnz > 0 ? nnz : 1)*sizeof(int));
  sp->val    = (mjtNum*) mju_malloc((nnz > 0 ? nnz : 1)*sizeof(mjtNum));
  int e = 0;
  for (int v=0; v < nv; v++) for (int c=0; c < 3; c++) {   // row 3v+c: lower-tri cols 3v..3v+c (diagonal last)
    int r = 3*v + c;
    sp->rowadr[r] = e; sp->rownnz[r] = c + 1;
    for (int cc=0; cc <= c; cc++) sp->colind[e++] = 3*v + cc;
  }
}

static void ipc_spFree(ipcSparse* sp) {
  mju_free(sp->rownnz); mju_free(sp->rowadr); mju_free(sp->colind); mju_free(sp->val);
}

// find the CSR value index of lower-tri entry (row, col<=row); -1 if not in pattern (binary search)
static int ipc_spIdx(const ipcSparse* sp, int row, int col) {
  int lo = sp->rowadr[row], hi = lo + sp->rownnz[row] - 1;
  while (lo <= hi) { int mid = (lo+hi)/2; int c = sp->colind[mid];
    if (c == col) return mid; else if (c < col) lo = mid+1; else hi = mid-1; }
  return -1;
}

// per-vertex 3x3 block-Jacobi preconditioner z = blockdiag(H)^-1 r: invert each 3-DOF vertex's 3x3 diagonal
// Hessian block (inertia + contact GN n*n^T) and apply it. Captures the per-vertex contact-normal coupling a
// scalar diagonal misses, with no triangular solve -- the well-conditioned AL formulation makes this simple
// preconditioner sufficient (matching Li et al. arXiv:2512.12151 sec 5.1). H is symmetric lower-tri in sp->val.
static void ipc_jacobiApply(const ipcSparse* sp, mjtNum* z, const mjtNum* r) {
  int N = sp->N;
  for (int v = 0; v + 2 < N; v += 3) {
    mjtNum B[9];
    for (int a = 0; a < 3; a++) for (int b = 0; b <= a; b++) {
      int id = ipc_spIdx(sp, v+a, v+b);
      mjtNum h = (id >= 0) ? sp->val[id] : 0;
      B[3*a+b] = h; B[3*b+a] = h;
    }
    mjtNum Bi[9]; ipc_mat3inv(Bi, B);
    for (int a = 0; a < 3; a++) { mjtNum s = 0; for (int b = 0; b < 3; b++) s += Bi[3*a+b]*r[v+b]; z[v+a] = s; }
  }
}

// IPC-style variational integrator (integrator="ipc"): owns the full step, minimizing the per-step incremental
// potential (inertia + flex edge-stretch + bending energy) with penetration-free contact by a barrier-free
// AUGMENTED-LAGRANGIAN method (paper arXiv 2512.12151) -- the contact multiplier carries the force at a
// fixed low stiffness (no log barrier, no kappa adaptation, no TOI-lock); the inner optimizer is solved freely and
// the committed output is a conservative-CCD blend from the last intersection-free state. Covers flex self-contact
// (vertex-triangle + edge-edge) and flex-vs-geom. Flex-only: falls back to Euler if no 2D flex.
// contact-stiffness floor: the AL's penalty stiffness mu is auto-set to 0.1*max(inertia diag) each step
// (matched to the system Hessian scale, paper Eq.20); this is only the pre-mdiag init placeholder.
#define IPC_KAPPA0   1000.0

// A tracked GENERAL rigid penalty contact, ANCHORED at q_n, driven by MuJoCo's own collision result d->contact.
// End A = geom[0]'s body, end B = geom[1]'s body; either may be static (welded to world). The contact point
// con->pos is stored as a body-LOCAL anchor in each body (pA, pB). The FROZEN normal n (geom[0]->geom[1]) and the
// FROZEN nv gradient direction b = J_B^T n - J_A^T n (mj_jac, stored separately) fix the push-out direction and
// the Gauss-Newton Hessian; the SCALAR gap is recomputed LIVE through full nonlinear FK each Newton iter (the foot
// cannot tunnel). k = m_eff/h^2, m_eff = 1/(b^T M^-1 b) the contact-space effective mass (= the sphere's mass/h^2
// as a special case, and the reduced mass for body-vs-body).
typedef struct {
  int    bodyA, bodyB, movA, movB;
  mjtNum pA[3], pB[3];     // contact point in each body's local frame (anchors)
  mjtNum xA0[3], xB0[3];   // world anchors at q_n (== con->pos); used live for the static side
  mjtNum n[3];             // frozen unit normal, geom[0]->geom[1]
  mjtNum dist0, k;         // signed gap at detection (<0 = penetration); penalty stiffness
} ipcGcon;

// LIVE signed gap of contact c at the CURRENT FK config (caller must have set d->qpos and run mj_kinematics).
// gap = dist0 + n.(xB - xA), xA/xB = world anchors via FK (a static side keeps its frozen q_n anchor). The
// (xB - xA) = (geom1 - geom0) ordering matches con->dist so deeper penetration -> more-negative gap; at q_n the
// anchors coincide (both == con->pos) so gap == dist0 exactly. xA,xB are full NONLINEAR FK of the material contact
// points, so this does NOT reduce to the frozen-J linear gap that tunnels.
static mjtNum gconGap(mjData* d, const ipcGcon* c) {
  mjtNum xA[3], xB[3];
  if (c->movA) mj_local2Global(d, xA, NULL, c->pA, NULL, c->bodyA, mjSAMEFRAME_NONE); else mju_copy3(xA, c->xA0);
  if (c->movB) mj_local2Global(d, xB, NULL, c->pB, NULL, c->bodyB, mjSAMEFRAME_NONE); else mju_copy3(xB, c->xB0);
  return c->dist0 + mju_dot3(c->n, xB) - mju_dot3(c->n, xA);
}

// Energy of the per-tree incremental potential at config q: kinetic 0.5/h^2 d(q~,q)^T M d(q~,q) PLUS the simple
// one-sided penalty 0.5*k*min(0,gap)^2 over the tracked contacts. FK is refreshed at q (mj_kinematics only, M
// stays frozen) so gconGap sees the live anchors.
static mjtNum ipc_treeE(const mjModel* m, mjData* d, const mjtNum* q, const mjtNum* qtil,
                        mjtNum ih2, int nv, mjtNum* rbuf, mjtNum* mrbuf, int nc, const ipcGcon* C) {
  mju_copy(d->qpos, q, m->nq);
  mj_kinematics(m, d);
  mj_differentiatePos(m, rbuf, 1.0, qtil, q);
  mj_mulM(m, d, mrbuf, rbuf);                                  // M*r matrix-free (M frozen at q_n via d->qM)
  mjtNum E = 0.5*ih2*mju_dot(rbuf, mrbuf, nv);                 // kinetic incremental potential
  for (int c=0; c < nc; c++) { mjtNum gap = gconGap(d, &C[c]); if (gap < 0) E += 0.5*C[c].k*gap*gap; }
  return E;
}

// PER-TREE variational implicit-Euler core, generalized-coordinate form (the destination architecture: ONE
// energy minimized over the nv tangent; flex vertices and slide spheres are the trivial 1-body-tree case, an
// articulated body is one tree whose mass block is its dense M). Built rigid-first: handles the all-rigid (no-
// flex) case; the flex elastic/AL folds into the SAME Newton later. Smooth forces enter EXPLICITLY through the
// predictor (qacc_smooth) so with no contact this is semi-implicit Euler exactly (round-off match to mjINT_EULER).
//   q~ = q_n (+) h*(v_n + h*qacc_smooth)  [mj_integratePos];  minimize 0.5/h^2 d(q~,q)^T M d(q~,q) + penalty
//   v_{n+1} = (q_n (-) q_{n+1})/h  [mj_differentiatePos].  M = M(q_n) dense-frozen; Newton solved densely.
// CONTACT: the SIMPLE one-sided penalty 0.5*k*min(0,gap)^2 over MuJoCo's OWN collision result d->contact (all geom
// types, articulated bodies). Each contact is anchored at q_n with a FROZEN normal/direction b (via mj_jac) but a
// LIVE nonlinear gap (FK each Newton iter) -> a fast body can't tunnel. One rank-1 term k*b b^T per contact carries
// both bodies' self + cross blocks. Rigid bodies NEVER enter a multiplier / CCD / feasibility cap -- penetrate +
// recover (frictionless, no solref yet). The penetration-free AL guarantee stays flex-flex (the flex path). nv is
// small (rigid-only), so a dense Cholesky is used; the per-tree block-Jacobi PCG replaces it at larger scale.
static void mj_ipcTree(const mjModel* m, mjData* d) {
  mjtNum h = m->opt.timestep; int nv = m->nv, nq = m->nq; mjtNum ih2 = 1.0/(h*h);
  // ---- detect once at q_n from d->contact (populated by the predictor mj_forward: for the no-flex case mj_step
  // leaves CONTACT enabled, only the constraint solve is skipped -- so native collision ran at q_n) ----
  int cap = 256, nc = 0;
  ipcGcon* C = (ipcGcon*) mju_malloc(cap*sizeof(ipcGcon));
  mjtNum* cb = (mjtNum*) mju_malloc((nv > 0 ? cap*nv : 1)*sizeof(mjtNum));   // frozen nv gradient dir per contact
  mjtNum* JA = (mjtNum*) mju_malloc((nv > 0 ? 3*nv : 1)*sizeof(mjtNum));     // detect-time scratch
  mjtNum* JB = (mjtNum*) mju_malloc((nv > 0 ? 3*nv : 1)*sizeof(mjtNum));
  mjtNum* Mb = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  for (int ic=0; ic < d->ncon && nc < cap; ic++) {
    const mjContact* con = d->contact + ic;
    if (con->geom[0] < 0 || con->geom[1] < 0) continue;      // skip flex contacts (handled by the flex path)
    int bA = m->geom_bodyid[con->geom[0]], bB = m->geom_bodyid[con->geom[1]];
    int movA = (m->body_weldid[bA] != 0), movB = (m->body_weldid[bB] != 0);
    if (!movA && !movB) continue;                            // static-static: nothing to do
    ipcGcon* c = &C[nc];
    c->bodyA=bA; c->bodyB=bB; c->movA=movA; c->movB=movB; c->dist0=con->dist;
    mju_copy3(c->n, con->frame);                             // frame[0:3] = unit normal geom[0]->geom[1]
    mju_copy3(c->xA0, con->pos); mju_copy3(c->xB0, con->pos);
    mjtNum rel[3];
    mju_sub3(rel, con->pos, d->xpos+3*bA); mju_mulMatTVec3(c->pA, d->xmat+9*bA, rel);   // world -> body-local anchor
    mju_sub3(rel, con->pos, d->xpos+3*bB); mju_mulMatTVec3(c->pB, d->xmat+9*bB, rel);
    mj_jac(m, d, JA, NULL, con->pos, bA);                    // 3 x nv (zero rows for a weld-to-world body)
    mj_jac(m, d, JB, NULL, con->pos, bB);
    mjtNum* b = cb + nc*nv;                                  // gradient dir b = J_B^T n - J_A^T n (geom1 - geom0)
    for (int i=0; i < nv; i++) { mjtNum s=0; for (int rr=0; rr<3; rr++) s += (JB[rr*nv+i]-JA[rr*nv+i])*c->n[rr]; b[i]=s; }
    mj_solveM(m, d, Mb, b, 1);                               // M^-1 b on the frozen q_n factorization
    mjtNum bMb = mju_dot(b, Mb, nv);
    if (bMb < mjMINVAL) continue;                            // both ends immovable along n -> drop
    c->k = ih2/bMb;                                          // k = m_eff/h^2, m_eff = 1/(b^T M^-1 b)
    nc++;
  }
  // ---- predictor q~, save q_n, Newton over the tangent with a MATRIX-FREE PCG inner solve + line search ----
  // The Hessian H = ih2*M + sum_active k*b b^T is applied matrix-free (mj_mulM for the kinetic block, rank-1 per
  // active contact); preconditioned by the exact inverse of the kinetic block, (ih2*M)^-1 = h^2*M^-1, via h^2*
  // mj_solveM on the frozen q_n factorization. M block-diagonal over trees -> mj_solveM IS the per-tree block
  // solve, so contact-free PCG converges in ~1 iter and contact adds O(n_active) iters. (STEP 1 of the flex unify:
  // rigid path only; the dense Cholesky is gone; the same PCG callbacks generalize to the flex packing later.)
  int cap2 = nc > 0 ? nc : 1;
  mjtNum* qtil = (mjtNum*) mju_malloc((nq > 0 ? nq : 1)*sizeof(mjtNum));
  mjtNum* qn   = (mjtNum*) mju_malloc((nq > 0 ? nq : 1)*sizeof(mjtNum));
  mjtNum* q    = (mjtNum*) mju_malloc((nq > 0 ? nq : 1)*sizeof(mjtNum));
  mjtNum* qtr  = (mjtNum*) mju_malloc((nq > 0 ? nq : 1)*sizeof(mjtNum));
  mjtNum* vp   = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  mjtNum* r    = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  mjtNum* g    = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  mjtNum* dv   = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  mjtNum* Mr   = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));   // M*r (gradient) + ipc_treeE scratch
  mjtNum* pr   = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));   // PCG residual / search / H*p
  mjtNum* pz   = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  mjtNum* pp   = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  mjtNum* pHp  = (mjtNum*) mju_malloc((nv > 0 ? nv : 1)*sizeof(mjtNum));
  int*    acti = (int*)    mju_malloc(cap2*sizeof(int));                   // active-contact indices (gap<0)
  mjtNum* kvec = (mjtNum*) mju_malloc(cap2*sizeof(mjtNum));                // per-contact stiffness (for the ctx)
  for (int c=0; c < nc; c++) kvec[c] = C[c].k;
  for (int i=0; i < nq; i++) qn[i] = d->qpos[i];             // q_n (d->qpos / FK currently at q_n)
  for (int i=0; i < nv; i++) vp[i] = d->qvel[i] + h*d->qacc_smooth[i];
  for (int i=0; i < nq; i++) qtil[i] = qn[i];
  mj_integratePos(m, qtil, vp, h);                           // predictor q~
  for (int i=0; i < nq; i++) q[i] = qn[i];                   // start at q_n
  for (int it=0; it < 32; it++) {
    mju_copy(d->qpos, q, nq); mj_kinematics(m, d);           // load-bearing: FK at trial q so gconGap is live
    mj_differentiatePos(m, r, 1.0, qtil, q);                 // r = d(q~, q)
    mj_mulM(m, d, Mr, r); for (int i=0; i < nv; i++) g[i] = ih2*Mr[i];   // kinetic gradient ih2*M*r
    int na = 0;                                              // freeze the active set + accumulate contact gradient
    for (int cc=0; cc < nc; cc++) {
      mjtNum gap = gconGap(d, &C[cc]);
      if (gap < 0) { const mjtNum* b = cb + cc*nv; mjtNum kg = C[cc].k*gap;
        for (int i=0; i < nv; i++) g[i] += kg*b[i];
        acti[na++] = cc;
      }
    }
    // shared PCG (ipc_solveU, the unified loop): solve H dv = g, H = ih2*M + sum_a k_a b_a b_a^T; rigid ctx ->
    // ih2*M apply via mj_mulM, precond h^2*M^-1 via mj_solveM. dx = -H^-1 g is the Newton step directly. (Mr is
    // free after the gradient -> reused as the ipc_solveU usol scratch.)
    ipcCtx rctx = {0};
    rctx.rigid = 1; rctx.m = m; rctx.d = d; rctx.ih2 = ih2; rctx.h2 = h*h;
    rctx.na = na; rctx.acti = acti; rctx.cb = cb; rctx.kvec = kvec;
    ipc_solveU(dv, g, nv, &rctx, 1e-20, pr, pz, pp, pHp, Mr);
    mjtNum dvn = 0; for (int i=0; i < nv; i++) { mjtNum a = dv[i] < 0 ? -dv[i] : dv[i]; if (a > dvn) dvn = a; }
    if (dvn < 1e-10) break;                                  // step ~ 0 -> converged
    mjtNum E0 = ipc_treeE(m, d, q, qtil, ih2, nv, r, Mr, nc, C), alpha = 1.0; int ok = 0;
    for (int ls=0; ls < 12; ls++) {
      for (int i=0; i < nq; i++) qtr[i] = q[i];
      mj_integratePos(m, qtr, dv, alpha);
      if (ipc_treeE(m, d, qtr, qtil, ih2, nv, r, Mr, nc, C) <= E0 + 1e-12) { ok = 1; break; }
      alpha *= 0.5;
    }
    if (!ok) break;                                          // frozen-Jacobian descent stalled -> stop
    mj_integratePos(m, q, dv, alpha);
  }
  // ---- readback + commit; leave FK consistent at the committed config for downstream ----
  mj_differentiatePos(m, d->qvel, h, qn, q);                 // v_{n+1} = (q_{n+1} (-) q_n)/h
  for (int i=0; i < nq; i++) d->qpos[i] = q[i];              // commit q_{n+1}
  mj_kinematics(m, d); mj_comPos(m, d);                      // FK + cdof at q_{n+1}
  d->time += h;
  mju_free(C); mju_free(cb); mju_free(JA); mju_free(JB); mju_free(Mb);
  mju_free(qtil); mju_free(qn); mju_free(q); mju_free(qtr); mju_free(vp); mju_free(r); mju_free(g); mju_free(dv);
  mju_free(Mr); mju_free(pr); mju_free(pz); mju_free(pp); mju_free(pHp); mju_free(acti); mju_free(kvec);
}

void mj_IPC(const mjModel* m, mjData* d) {
  mjtNum h = m->opt.timestep;
  mjtNum kappa = IPC_KAPPA0;   // placeholder; set to the auto mu = 0.1*max(inertia diag) once mdiag is known
  // all dim-2 flexes participate in the IPC solve (was: only the first). Their vertices are concatenated
  // into the free-point array in flex order; fxadr[k] is the free-point offset of dim-2 flex flist[k].
  int nfd = 0;
  for (int i=0; i < m->nflex; i++) if (m->flex_dim[i] == 2) nfd++;
  if (nfd == 0) { mj_ipcTree(m, d); return; }   // no 2D flex -> per-tree generalized-coordinate core
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

  // FLEX-ONLY point array. The point SoA holds ONLY flex vertices now: npt == nfv. Standalone 3-slide bodies
  // carrying a sphere geom (rigid points) live in a SEPARATE rigid-body list (rbody/rqadr/rdofadr/rmass/
  // rgeom/rrad below) that occupies its own block of the solver vector AFTER the flex DOFs. This keeps the
  // flex fidx assignment (0..nfree_flex-1) and the flex solver arithmetic bit-identical to the unified-array
  // version, while giving the rigid bodies their own kinetic term, state slots, and readback.
  char* isflexvert = (char*) mju_malloc((m->nbody > 0 ? m->nbody : 1)*sizeof(char));
  for (int b=0; b < m->nbody; b++) isflexvert[b] = 0;
  for (int v=0; v < nfv; v++) isflexvert[m->flex_vertbodyid[pt2vg[v]]] = 1;
  int nrigid = 0;                                            // standalone 3-slide sphere bodies (rigid points)
  for (int b=0; b < m->nbody; b++) {
    if (isflexvert[b] || m->body_dofnum[b] != 3 || m->jnt_type[m->body_jntadr[b]] != mjJNT_SLIDE) continue;
    for (int g=m->body_geomadr[b]; g < m->body_geomadr[b]+m->body_geomnum[b]; g++)
      if (m->geom_type[g] == mjGEOM_SPHERE) { nrigid++; break; }
  }
  int npt = nfv;                                             // point array is flex-only
  int* dofadr = (int*) mju_malloc((npt > 0 ? npt : 1)*sizeof(int));
  int* qpadr  = (int*) mju_malloc((npt > 0 ? npt : 1)*sizeof(int));   // qpos address (NOT dof address: differs after
  int* fidx   = (int*) mju_malloc((npt > 0 ? npt : 1)*sizeof(int));   // free/ball joints, which have more qpos than dof)
  mjtNum* mass = (mjtNum*) mju_malloc((npt > 0 ? npt : 1)*sizeof(mjtNum));
  mjtNum* rad  = (mjtNum*) mju_malloc((npt > 0 ? npt : 1)*sizeof(mjtNum));   // per-point radius (flex_radius)
  int* pbody  = (int*) mju_malloc((npt > 0 ? npt : 1)*sizeof(int));          // body id, for the slide-frame rotation R
  int* pgeom  = (int*) mju_malloc((npt > 0 ? npt : 1)*sizeof(int));          // -1 for a flex vertex (no sphere now)
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
  int nfree_flex = nfree;                                     // flex free-DOF count; rigid fidx start here
  // RIGID-BODY LIST: each standalone 3-slide sphere body. It rides the solver vector in a block AFTER the
  // flex DOFs (fidx = nfree_flex + i) and the state vectors in a block AFTER the flex points (state slot
  // nfv + i). Stored: body id, qpos addr, dof addr, point mass (qM diagonal), sphere geom id, radius.
  int* rbody   = (int*) mju_malloc((nrigid > 0 ? nrigid : 1)*sizeof(int));     // body id (for the slide-frame R)
  int* rqadr   = (int*) mju_malloc((nrigid > 0 ? nrigid : 1)*sizeof(int));     // jnt_qposadr
  int* rdofadr = (int*) mju_malloc((nrigid > 0 ? nrigid : 1)*sizeof(int));     // body_dofadr
  mjtNum* rmass = (mjtNum*) mju_malloc((nrigid > 0 ? nrigid : 1)*sizeof(mjtNum));
  int* rgeom   = (int*) mju_malloc((nrigid > 0 ? nrigid : 1)*sizeof(int));     // sphere geom id
  mjtNum* rrad = (mjtNum*) mju_malloc((nrigid > 0 ? nrigid : 1)*sizeof(mjtNum)); // sphere radius
  int nr = 0;
  for (int b=0; b < m->nbody; b++) {
    if (isflexvert[b] || m->body_dofnum[b] != 3 || m->jnt_type[m->body_jntadr[b]] != mjJNT_SLIDE) continue;
    int gsph = -1;
    for (int g=m->body_geomadr[b]; g < m->body_geomadr[b]+m->body_geomnum[b]; g++)
      if (m->geom_type[g] == mjGEOM_SPHERE) { gsph = g; break; }
    if (gsph < 0) continue;
    int da = m->body_dofadr[b];
    rbody[nr] = b; rqadr[nr] = m->jnt_qposadr[m->body_jntadr[b]]; rdofadr[nr] = da;
    rmass[nr] = d->qM[m->M_rowadr[da] + m->M_rownnz[da] - 1];
    rgeom[nr] = gsph; rrad[nr] = m->geom_size[3*gsph];
    nr++;
  }
  mju_free(isflexvert);
  int nstate = nfv + nrigid;                                  // state-vector length (flex points + rigid bodies)
  int N = 3*(nfree_flex + nrigid), Na = (N > 0 ? N : 1);      // solver dim: flex DOFs then rigid DOFs
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
  // bending flaps: read MuJoCo's precomputed quadratic operator Q from flex_bending (b[17e+4i+j]); flat-rest
  // assumption -> drop the curved-reference b[17e+16]. Stencil v[4]={edge0,edge1,flap0,flap1}; skip boundary
  // (v[3]==-1) and concave-rest (trace(Q)<0, rank-1 negative -> keeps each block PSD without a clamp).
  int nbeMax = 0;
  for (int k=0; k < nfd; k++) { int fi = flist[k];
    if (m->flex_dim[fi] == 2 && m->flex_bendingadr[fi] >= 0) nbeMax += m->flex_edgenum[fi]; }
  ipcBend* bends = (ipcBend*) mju_malloc((nbeMax > 0 ? nbeMax : 1)*sizeof(ipcBend));
  int nbe = 0;
  for (int k=0; k < nfd; k++) {
    int fi = flist[k], badr = m->flex_bendingadr[fi];
    if (m->flex_dim[fi] != 2 || badr < 0) continue;
    int eadr = m->flex_edgeadr[fi], enum_k = m->flex_edgenum[fi];
    const mjtNum* b = m->flex_bending + badr;
    for (int e=0; e < enum_k; e++) {
      const int* edge = m->flex_edge + 2*(eadr + e);
      const int* flap = m->flex_edgeflap + 2*(eadr + e);
      int v[4] = { edge[0], edge[1], flap[0], flap[1] };
      if (v[3] == -1) continue;                                  // boundary edge: only one adjacent triangle
      const mjtNum* Qe = b + 17*e;
      if (Qe[0] + Qe[5] + Qe[10] + Qe[15] < 0) continue;         // trace(Q)<0: concave rest -> skip (PSD safe)
      ipcBend* bn = &bends[nbe++];
      for (int i=0; i < 4; i++) { int gp = fxadr[k] + v[i]; bn->vg[i] = gp; bn->fv[i] = fidx[gp]; }
      for (int i=0; i < 16; i++) bn->Q[i] = Qe[i];
      bn->b16 = Qe[16];   // curved-reference (Garg "Cubic Shells"): makes the curved rest the equilibrium
    }
  }
  int amax = npt*64 + 1024;                   // capacity of the active-contact list
  ipcCon* acon = (ipcCon*) mju_malloc(amax*sizeof(ipcCon));
  ipcCC* ccache = (ipcCC*) mju_malloc(amax*sizeof(ipcCC));
  int candmax = npt*192 + 8192;               // capacity of the per-step candidate list (sized for the
                                              // geom-feature-heavy bag-in-bin contact: ~160k at npt~1100)
  ipcCon* cand = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon));
  mjtNum* cgap = (mjtNum*) mju_malloc(candmax*sizeof(mjtNum));   // per-candidate gap at x (try->ccd/E0)
  mjtNum* minc = (mjtNum*) mju_malloc((npt > 0 ? npt : 1)*sizeof(mjtNum));   // per-point min held gap (earliest-collision filter)
  int*    held = (int*)    mju_malloc(candmax*sizeof(int));                  // candidate kept in the inner assembly?
  ipcCon* candLS = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon)); // line-search subset (can activate this step)
  // FLEX persistent active set (the persistent active-set manager): maintained ACROSS outer iterations. Each iter:
  // keep existing pairs with abs(cnt)<=IPC_ASET_AGE, ADD new broad-phase candidates with CCD toi<1-1e-6, dedup by
  // pairHash. The assembled set (grad+Hessian in ipc_try, AND ipc_energy) is THIS persistent set -- bounded by the
  // aging eviction, NOT the per-iter broad-phase. aheld is all-1 so the shared held-gated helpers assemble all of it.
  ipcCon* aset  = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon));   // persistent active pairs
  mjtNum* agap  = (mjtNum*) mju_malloc(candmax*sizeof(mjtNum));   // per-aset-pair maintained gap lower bound
  int*    aheld = (int*)    mju_malloc(candmax*sizeof(int));      // all-1 mask for the held-gated assemble/energy
  ipcCon* amerge = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon));  // merge scratch (new merged set built here)
  int     naset = 0;
  for (int c=0; c < candmax; c++) aheld[c] = 1;
  // Legacy active-set scratch. The current scheme replaces the old per-point gamma carrier with the
  // per-PAIR cnt state machine (g_cntKey/Val store; cnt rides in ipcCon.cnt). gam stays (passed to ipc_spBuild,
  // ignored); appr is filled by ipc_ccd; actc/actpt are now unused but kept allocated to avoid churning the frees.
  mjtNum* gam  = (mjtNum*) mju_malloc(candmax*sizeof(mjtNum));
  int*    actc = (int*)    mju_malloc(candmax*sizeof(int));
  int*    appr = (int*)    mju_malloc(candmax*sizeof(int));
  int*    actpt = (int*)   mju_malloc((npt > 0 ? npt : 1)*sizeof(int));
  for (int c=0; c < candmax; c++) { gam[c] = 1.0; actc[c] = 0; appr[c] = 0; }
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
  // state vectors are sized 3*nstate (flex points 0..nfv-1, then rigid bodies nfv..nfv+nrigid-1) so the rigid
  // kinetic prediction/restart/commit ride the same buffers as the flex points, indexed by their state slot.
  mjtNum* x    = (mjtNum*) mju_malloc(3*nstate*sizeof(mjtNum));
  mjtNum* xfree= (mjtNum*) mju_malloc(3*nstate*sizeof(mjtNum));   // AL two-state: intersection-free output path (paper x[k])
  mjtNum* xtil = (mjtNum*) mju_malloc(3*nstate*sizeof(mjtNum));
  mjtNum* xold = (mjtNum*) mju_malloc(3*nstate*sizeof(mjtNum));
  mjtNum* xn   = (mjtNum*) mju_malloc(3*nstate*sizeof(mjtNum));
  mjtNum* grad = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* dx   = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* mdiag= (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // inertia diagonal (for the H*p apply)
  mjtNum* rcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // CG residual / search-dir / Hp buffers
  mjtNum* zcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* pcg  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* Hpv  = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));
  mjtNum* usol = (mjtNum*) mju_malloc(Na*sizeof(mjtNum));   // PCG u-space solution
  mjtNum* estr = (mjtNum*) mju_malloc((ne > 0 ? 12*ne : 1)*sizeof(mjtNum)); // per-element g_a (9) + Me_a (3)
                                                                           // (full, PSD-projected, this Newton iter)

  // flex/sphere inertial prediction: q~ = q + h*v + h^2*qacc_smooth (point masses have no Coriolis).
  mjtNum* qacc_pred = (mjtNum*) mju_malloc((m->nv > 0 ? m->nv : 1)*sizeof(mjtNum));
  mju_copy(qacc_pred, d->qacc_smooth, m->nv);

  const mjtNum* vx = d->flexvert_xpos;
  for (int v=0; v < npt; v++) {
    // xold position source: flex vertex from flexvert_xpos (the point array is flex-only now)
    for (int c=0; c < 3; c++) xold[3*v+c] = vx[3*pt2vg[v]+c];
    if (fidx[v] >= 0) {
      int da = dofadr[v];
      // qvel/qacc_smooth are in the body's local slide frame; map to WORLD via the body rotation R
      // (= d->xmat). R=I for unrotated bodies (cloth, the balls); non-identity for the rotated bag.
      const mjtNum* R = d->xmat + 9*pbody[v];
      mjtNum vw[3], aw[3];
      mju_mulMatVec3(vw, R, d->qvel + da);
      mju_mulMatVec3(aw, R, qacc_pred + da);
      for (int c=0; c < 3; c++)
        xtil[3*v+c] = xold[3*v+c] + h*vw[c] + h*h*aw[c];
    } else {
      for (int c=0; c < 3; c++) xtil[3*v+c] = xold[3*v+c];   // pinned: fixed
    }
  }
  // RIGID kinetic prediction (state slot nfv+i): reuse the point formula. xold from the body's slide
  // position d->xpos[body] (consistent with the slide DOFs, vs a possibly-offset geom center); predictor
  // xtil = xold + h*v + h^2*qacc_smooth in world frame via R = d->xmat (fixed for a pure-slide body).
  for (int i=0; i < nrigid; i++) {
    int v = nfv + i, da = rdofadr[i];
    for (int c=0; c < 3; c++) xold[3*v+c] = d->xpos[3*rbody[i]+c];
    const mjtNum* R = d->xmat + 9*rbody[i];
    mjtNum vw[3], aw[3];
    mju_mulMatVec3(vw, R, d->qvel + da);
    mju_mulMatVec3(aw, R, qacc_pred + da);
    for (int c=0; c < 3; c++) xtil[3*v+c] = xold[3*v+c] + h*vw[c] + h*h*aw[c];
  }
  for (int i=0; i < 3*nstate; i++) x[i] = xold[i];   // start from last collision-free state (feasibility)

  mjtNum ih2 = 1.0/(h*h);
  for (int i=0; i < N; i++) mdiag[i] = 0;                          // inertia diagonal (fixed per step)
  for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
    int fi = fidx[v]; for (int c=0; c < 3; c++) mdiag[3*fi+c] = mass[v]*ih2;
  }
  // AL contact stiffness: kappa carries a FIXED, auto-set mu = 0.1 * max(inertia diag),
  // matched to the system's Hessian scale (paper Eq.20). No per-step adaptation -> no kappa
  // oscillation; non-penetration is the CCD's job (the AL force is finite). Computed over the FLEX-ONLY
  // mdiag (before the rigid block is written) so the flex AL is bit-identical to the unified-array version.
  {
    mjtNum mmax = 0;
    for (int i=0; i < N; i++) if (mdiag[i] > mmax) mmax = mdiag[i];
    kappa = 0.1*mmax;   // penalty stiffness mu = 0.1 * max(inertia diag), matched to the Hessian scale (paper Eq.20)
  }
  for (int i=0; i < nrigid; i++) {                                // rigid kinetic Hessian: M/h^2 (diagonal)
    int fi = nfree_flex + i; for (int c=0; c < 3; c++) mdiag[3*fi+c] = rmass[i]*ih2;
  }
  {   // AL: per-free-point flex/sphere contact-multiplier warm-start store (persisted across steps).
    if (g_palN != npt) { mju_free(g_pal); g_palN = npt;
      g_pal = (mjtNum*) mju_malloc((size_t)(npt > 0 ? npt : 1)*sizeof(mjtNum));
      for (int i=0; i < npt; i++) g_pal[i] = 0; }            // zero ONLY on resize -> warm-start persists
  }
  (void)kappa;
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
  ipc_cntStepBegin(candmax);   // swap+clear the per-pair cnt store for this step (OLD = last step's final cnts)
  int ncand = ipc_candidates(m, d, x, gv, ge, ngv, nge, r, rad, thresh, threshGeom, maxdisp, xold, xtil, ghat,
                             nfv, npt, fidx, flist, fxadr, nfd, pt2flex, cand, candmax);
  for (int c=0; c < ncand; c++) {   // AL: warm-start each fresh candidate's multiplier from g_pal + cnt from the store
    int vv[4], nvv; ipc_conVerts(&cand[c], vv, &nvv);            // (binding = max over its free-point participants)
    mjtNum s = 0;
    for (int q=0; q < nvv; q++) if (g_pal[vv[q]] > s) s = g_pal[vv[q]];
    cand[c].lam = s; cand[c].cnt = ipc_cntGet(ipc_pairHash(&cand[c])); cand[c].s = 0;
    gam[c] = 1.0;
  }
  // warm start: with no candidate contacts within thresh, the predictor x~ is collision-free (the thresh
  // margin covers the step displacement), so it is a far better feasible initial guess than xold and Newton
  // converges in ~1 iteration instead of ~2 -- halving the cost of contact-free steps.
  if (ncand == 0) for (int i=0; i < 3*nstate; i++) x[i] = xtil[i];
  // sparse Hessian pattern (mesh + candidate-contact couplings) for the IC(0) preconditioner, once/step
  // Rayleigh damping uses the previous-step elongations e_prev = e(xold); xold is fixed over the Newton
  // loop, so cache e_prev once per step. Only needed when damping is on.
  if (nfd > 0 && m->flex_damping[f] > 0)
    for (int t=0; t < ne; t++) { mjtNum dtmp[3][3]; ipc_elemEval(&elems[t], xold, dtmp, elems[t].ep); }
  ipcSparse sp = {0};   // block-Jacobi preconditioner Hessian -- only built/used on the PCG path (!useCRB)
  // mark active inter-flex/sphere candidates (gap already within their per-contact ghat) so the IC0 pattern
  // couples only those (~hundreds) instead of all ~15k candidates (~99% inactive). gap here == iter-0 gap
  // (x is unchanged until the Newton loop); geom contacts (types 2/3/4) add no coupling so they're skipped.
  for (int c=0; c < ncand; c++) {
    if (cand[c].type >= 2 && cand[c].type != 5) continue;
    mjtNum nn[3], cw[4]; int iv[4], nidx;
    cgap[c] = ipc_conGap(&cand[c], m, d, x, gv, ge, r, rad, nn, iv, cw, &nidx, thresh);
  }
  // FLEX: SEED the persistent active set from the iter-0 broad-phase. ipc_addCand already pruned cand to
  // closing-or-distance-active pairs (per-pair closing-bound prune), so every iter-0 candidate is a valid
  // initial active pair (seeding the active set from the first discrete-collision-detection pass).
  // lam/cnt were warm-started above. Aging + the per-iter CCD-toi merge then maintain it (bounded, not holdall).
  naset = (ncand < candmax) ? ncand : candmax;
  for (int c=0; c < naset; c++) { aset[c] = cand[c]; agap[c] = cgap[c]; }
  // block-Jacobi preconditioner: the per-vertex 3x3 diagonal of the maximal Hessian (sp, dim N).
  if (N > 0) ipc_spBuild(&sp, N);   // per-vertex 3x3 block-diagonal pattern (block-Jacobi precond)
  // precompute element -> CSR scatter indices once per step (the pattern is fixed over the Newton loop), so
  // the per-Newton assembly avoids a binary-search ipc_spIdx per matrix entry. -1 marks skipped entries
  // (upper-tri or pinned vertex). Order matches the assembly loop: idx = (i*3+j)*9 + a*3+b.
  int* escat = (N > 0) ? (int*) mju_malloc(ne*81*sizeof(int)) : NULL;
  if (N > 0) for (int t=0; t < ne; t++) {
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
  for (int i=0; i < 3*nstate; i++) xfree[i] = xold[i];   // intersection-free output path (paper x[k]), from feasible xold
  // Single AL Newton loop: inner_cap=1, beta ACCUMULATES to 1 (beta += (1-beta)*ac), terminate when
  // beta >= 1 - IPC_TOI_THRESH (0.9) AND (outer+1 >= min_iter OR newton_converged); no floor, no CFL cap, plain
  // monotone-or-converged line search.
  mjtNum beta = 0.0;
  mjtNum beta_eps = 1e-3;     // (legacy; only referenced by the profiler now)
  int inner_cap = 1;
  // SOFT flex-rigid contact pairs (sphere vs flex triangle / static geom / other sphere), built once per step
  // at xold + a margin. One-sided penalty only (penetrate+recover); never enters the hard set / CCD / line-search
  // feasibility. The gap is recomputed at the current x each Newton iter (ipc_softTry); the PAIR set is fixed here.
  mjtNum softmargin = 0.02;
  int softcap = nrigid*(ne + m->ngeom + nrigid) + 16;
  ipcSoft* soft = (ipcSoft*) mju_malloc((size_t)softcap*sizeof(ipcSoft));
  int nsoft = 0;
  for (int i=0; i < nrigid; i++) {
    const mjtNum* ci = &xold[3*(nfv+i)];
    for (int t=0; t < ne; t++) {                                 // sphere vs flex triangle
      int A=elems[t].vg[0], B=elems[t].vg[1], C=elems[t].vg[2];
      mjtNum cp[3], w[3], dd = ipc_ptTri(ci, &xold[3*A], &xold[3*B], &xold[3*C], cp, w);
      if (dd < rrad[i] + rad[A] + softmargin && nsoft < softcap) {
        soft[nsoft].type=0; soft[nsoft].ri=i; soft[nsoft].idx[0]=A; soft[nsoft].idx[1]=B; soft[nsoft].idx[2]=C; soft[nsoft].gi=-1; nsoft++; }
    }
    for (int g=0; g < m->ngeom; g++) {                           // sphere vs STATIC geom
      if (g == rgeom[i] || (m->geom_contype[g]==0 && m->geom_conaffinity[g]==0)) continue;
      if (m->body_dofnum[m->geom_bodyid[g]] != 0) continue;      // static geoms only
      mjtNum nn[3], dd = ipc_geomDist(m, g, d->geom_xpos+3*g, d->geom_xmat+9*g, ci, nn, rrad[i]+softmargin);
      if (dd < rrad[i] + softmargin && nsoft < softcap) {
        soft[nsoft].type=2; soft[nsoft].ri=i; soft[nsoft].gi=g; nsoft++; }
    }
    for (int j=i+1; j < nrigid; j++) {                           // sphere vs sphere
      const mjtNum* cj = &xold[3*(nfv+j)]; mjtNum dd=0;
      for (int c=0; c < 3; c++) { mjtNum t=ci[c]-cj[c]; dd+=t*t; } dd=mju_sqrt(dd);
      if (dd < rrad[i]+rrad[j]+softmargin && nsoft < softcap) {
        soft[nsoft].type=5; soft[nsoft].ri=i; soft[nsoft].idx[0]=j; soft[nsoft].gi=-1; nsoft++; }
    }
  }
  int outer_cap = 1024;       // newton_max_iter
  mjtNum last_maxdx = 1e30;    // last inner Newton-direction magnitude, exposed to the outer early-out
  int prof_outer=0, prof_inner=0, prof_cb=1, prof_nacon=0;   // MJ_IPC_PROF counters (cb: 1 initial build)
  for (int outer=0; outer < outer_cap && N > 0; outer++) {   // OUTER loop (single AL loop; paper Alg.1)
  prof_outer++;
  g0 = -1;                    // reset the inner-Newton relative stop each outer iteration
  // WORKING SET: the PERSISTENT active set aset/agap/aheld/naset maintained across outer iters by
  // ipc_mergeActiveSet (the persistent active-set manager). aheld is all-1: assemble/energy/slack/lambda
  // run over the ENTIRE persistent set (the aging eviction -- not a per-iter ld0<ghat test -- bounds it).
  ipcCon* wcon = aset;
  mjtNum* wgap = agap;
  int*    wheld = aheld;
  int     wn   = naset;
  // N1 linearize_constraints (at xfree, Eq.10): ld0/ln/lcw/liv this iter -> c(x) is linear in x.
  for (int c=0; c < wn; c++)
    wcon[c].ld0 = ipc_conGap(&wcon[c], m, d, xfree, gv, ge, r, rad,
                             wcon[c].ln, wcon[c].liv, wcon[c].lcw, &wcon[c].lniv, ghat);
  // RESTART the optimizer from the feasible base each outer iter. Our design solves a FREE x and advances
  // a SEPARATE feasible shadow xfree; left to accumulate, x runs ahead of xfree whenever the CCD can't follow (ac->0),
  // |x-xfree| grows, the re-query margin explodes the broad-phase and beta collapses -> the contact-set-explosion
  // NaN. We COMMIT xfree, keeping a single CCD-advanced position, so tying x to the feasible base each iter costs
  // nothing in committed motion and kills the divergence.
  for (int i=0; i < 3*nstate; i++) x[i] = xfree[i];   // restart the optimizer from the feasible base each Newton iter
  (void)minc; (void)appr; (void)gam; (void)actc;
  // N2 update_slack (at x): materialize s[c] and the slack-baked d the assemble (ipc_try) / lambda use.
  ipc_updateSlack(wcon, wn, wheld, x, xfree, rad, ghat, mass, ih2);
  int newton_converged_out = 0;   // carry the last inner iter's newton_converged to the outer termination
  for (int it=0; it < inner_cap && N > 0; it++) {
    prof_inner++; g_nact = 0;   // [PROF] reset active count; last inner iter's value is read in the profiler print
    for (int i=0; i < N; i++) grad[i] = 0;
    if (N > 0) {                                              // IC0 sparse Hessian (per-vertex 3x3 diagonal)
      for (int i=0; i < sp.nnz; i++) sp.val[i] = 0;
      for (int fi=0; fi < N/3; fi++) for (int c=0; c < 3; c++)   // inertia: diagonal
        sp.val[ipc_spIdx(&sp, 3*fi+c, 3*fi+c)] += mdiag[3*fi+c];
    }
    for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
      int fi = fidx[v]; mjtNum mh = mass[v]*ih2;
      for (int c=0; c < 3; c++) grad[3*fi+c] += mh*(x[3*v+c]-xtil[3*v+c]);
    }
    for (int i=0; i < nrigid; i++) {                          // rigid kinetic gradient: M/h^2*(x_r - xtil_r)
      int fi = nfree_flex + i, v = nfv + i; mjtNum mh = rmass[i]*ih2;
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
      for (int a=0; a < 3; a++) {                            // geometric stress part -- PSD-PROJECTED
        int p = ipc_eedge[a][0], q = ipc_eedge[a][1];
        // Keep only TENSILE geometric stiffness (Me>=0). The geometric block is Me[a]*[[I,-I],[-I,I]] over (p,q),
        // which is NEGATIVE-definite when Me<0 (edge compressed below rest -- the deep-bag regime), making the element
        // Hessian indefinite -> the single-Newton-iteration implicit step is non-contractive -> the vz~1 oscillation.
        // Clamping to max(Me,0) keeps the Newton operator SPD (the Gauss-Newton block above is already SPD), restoring
        // implicit-Euler unconditional stability. The FORCE (grad, above) keeps the FULL Me -- forces are unchanged.
        mjtNum Mg = Me[a] > 0.0 ? Me[a] : 0.0;   // structural PSD clamp (Me*Laplacian is PSD iff Me>=0; no EVD needed)
        for (int k=0; k < 3; k++) {
          He[3*p+k][3*p+k] += Mg; He[3*q+k][3*q+k] += Mg;
          He[3*p+k][3*q+k] -= Mg; He[3*q+k][3*p+k] -= Mg; } }
      for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {   // scatter lower-tri into sp (precomputed idx)
        int base = t*81 + (i*3+j)*9;
        for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
          int id = escat[base + a*3+b]; if (id >= 0) sp.val[id] += He[3*i+a][3*j+b]; } }
    }
    // bending: grad += (Q (x) I3) x + b16*frc (curved-reference); add Q's diagonal to sp (block-Jacobi).
    for (int bt=0; bt < nbe; bt++) {
      const ipcBend* bn = &bends[bt]; const mjtNum* Q = bn->Q;
      mjtNum frc[4][3] = {{0}};                              // b16 curved-reference force = grad of b16*det[ed]
      if (bn->b16 != 0) {
        const mjtNum* x0 = x + 3*bn->vg[0]; mjtNum ed[3][3];
        for (int a=0; a < 3; a++) { const mjtNum* xa = x + 3*bn->vg[a+1];
          for (int kc=0; kc < 3; kc++) ed[a][kc] = xa[kc] - x0[kc]; }
        mju_cross(frc[1], ed[1], ed[2]); mju_cross(frc[2], ed[2], ed[0]); mju_cross(frc[3], ed[0], ed[1]);
        for (int kc=0; kc < 3; kc++) frc[0][kc] = -(frc[1][kc] + frc[2][kc] + frc[3][kc]);
      }
      for (int i=0; i < 4; i++) { int fi = bn->fv[i]; if (fi < 0) continue;
        for (int kc=0; kc < 3; kc++) { mjtNum g = bn->b16*frc[i][kc];
          for (int j=0; j < 4; j++) g += Q[4*i+j]*x[3*bn->vg[j]+kc];
          grad[3*fi+kc] += g; }
        mjtNum qii = Q[4*i+i];
        for (int kc=0; kc < 3; kc++) { int id = ipc_spIdx(&sp, 3*fi+kc, 3*fi+kc); if (id >= 0) sp.val[id] += qii; }
      }
    }
    // assemble active contacts. First Newton iter: scan all candidates (initializes cgap exactly). Later
    // iters: only re-test the active set {cgap < ghat} (cgap is a maintained lower bound, so this set
    // contains every candidate that is or could be active). grad/acon/ccache get the active barrier blocks.
    int nacon = 0;
    if (it == 0) {
      for (int c=0; c < wn; c++) if (wheld[c])
        ipc_try(wcon[c], m, d, x, xfree, gv, ge, r, rad, ghat, xold, mass, ih2, fidx, grad, acon, ccache, &nacon, amax, &wgap[c]);
    } else {
      for (int c=0; c < wn; c++) if (wheld[c] && wgap[c] < ghat)
        ipc_try(wcon[c], m, d, x, xfree, gv, ge, r, rad, ghat, xold, mass, ih2, fidx, grad, acon, ccache, &nacon, amax, &wgap[c]);
    }
    for (int s=0; s < nsoft; s++)                               // SOFT flex-rigid penalty (penetration only); shares ccache/nacon
      ipc_softTry(&soft[s], m, d, x, nfv, rrad, rad, rmass, ih2, nfree_flex, fidx, grad, ccache, &nacon, amax);
    if (nacon > prof_nacon) prof_nacon = nacon;
    for (int c=0; c < nacon; c++) {                          // contact GN -> IC0 sparse Hessian
      const ipcCC* cc = &ccache[c];
      for (int p=0; p < cc->nidx; p++) for (int q=0; q < cc->nidx; q++) {
        int fp = cc->f[p], fq = cc->f[q]; if (fp < 0 || fq < 0) continue;
        mjtNum w2 = cc->bdd*cc->cw[p]*cc->cw[q];
        for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
          int R = 3*fp+a, C = 3*fq+b; if (C > R) continue;
          int id = ipc_spIdx(&sp, R, C);              // -1 if this coupling was not in the pattern:
          if (id >= 0) sp.val[id] += w2*cc->n[a]*cc->n[b];   // drop it (degrades preconditioner, never corrupts)
        } }
    }
    // converged when the residual force has dropped well below its initial magnitude. Relative (rather
    // than the old absolute 1e-8) so the test is scale-invariant across scenes/units: the absolute floor
    // depends on mass/h^2, kappa, mesh, so a fixed 1e-8 sat at/below the achievable gradient floor for
    // stiff-contact steps -> Newton ground out useless iterations after it had already converged.
    mjtNum gnorm, gna2 = 0;
    for (int i=0; i < N; i++) gna2 += grad[i]*grad[i];
    gnorm = sqrt(gna2);
    if (g0 < 0) g0 = gnorm;                                   // initial residual (first Newton iteration)
    ipcCtx fctx = {0};
    fctx.rigid = 0; fctx.sp = &sp; fctx.mdiag = mdiag; fctx.nelem = ne; fctx.elems = elems; fctx.estr = estr;
    fctx.nbe = nbe; fctx.bends = bends; fctx.ccache = ccache; fctx.nacon = nacon;
    ipc_solveU(dx, grad, N, &fctx, 1e-8, rcg, zcg, pcg, Hpv, usol);
    mjtNum maxdx = 0;   // max free-vertex displacement of the Newton direction (for the lower-bound decay)
    for (int v=0; v < N/3; v++) { mjtNum n2 = dx[3*v]*dx[3*v]+dx[3*v+1]*dx[3*v+1]+dx[3*v+2]*dx[3*v+2];
                                  if (n2 > maxdx) maxdx = n2; }
    maxdx = sqrt(maxdx);
    last_maxdx = maxdx;   // expose to the outer early-out (system-at-rest test)
    if (getenv("MJ_IPC_PROF")) {   // [DIAG] Newton-decrement sign: gdx=grad.dx>0 => ascent => indefinite H
      mjtNum gdx = 0; for (int i=0; i < N; i++) gdx += grad[i]*dx[i];
      static long ac = 0; if (gdx > 1e-9 && ac++ < 30)
        fprintf(stderr, "[ASCENT] outer=%d gdx=%+.3e maxdx=%.3e -> indefinite H (PSD clamp insufficient)\n", outer, gdx, maxdx);
    }
    // N6 newton_tolerance (flex path, L-infinity dx checker): newton_converged = L-infinity over ALL dx
    // COMPONENTS <= velocity_tol*dt, evaluated on the SOLVED dx BEFORE line search. Used to OR-accept the full step
    // in the line search and to allow termination once beta is feasible. No separate min_iter floor.
    int newton_converged = 0;
    {   // L-infinity dx checker: converged once the largest dx component is below velocity_tol*dt.
      mjtNum maxc = 0;
      for (int i=0; i < N; i++) { mjtNum a = dx[i] < 0 ? -dx[i] : dx[i]; if (a > maxc) maxc = a; }
      newton_converged = (maxc <= IPC_VEL_TOL*h);
      newton_converged_out = newton_converged;
    }
    // line-search subset: a candidate can contribute to the barrier at xn = x + alpha*dx (alpha in
    // [0,cap], cap<=1) only if its gap can drop below ghat. Its gap drop over a full step is bounded
    // by the sum of |dx| over its flex vertices, so keep only c with cgap[c] < ghat + that bound.
    int nls = 0;
    for (int c=0; c < wn; c++) {
      int vv[4], nvv; ipc_conVerts(&wcon[c], vv, &nvv);
      mjtNum lub = 0;
      for (int q=0; q < nvv; q++) { int fq = fidx[vv[q]]; if (fq < 0) continue;
        lub += sqrt(dx[3*fq]*dx[3*fq] + dx[3*fq+1]*dx[3*fq+1] + dx[3*fq+2]*dx[3*fq+2]); }
      if (wheld[c] && wgap[c] < ghat + lub) candLS[nls++] = wcon[c];   // assembled-set parity with the Newton Hessian
    }
    // N7 line search. E0 baseline over the same candidate subset (candLS) as the trials.
    mjtNum E0 = ipc_energy(m, d, npt, ne, elems, nbe, bends, x, xtil, fidx, mass, h, r, rad, ghat,
                           gv, ge, candLS, nls, xold, xfree, nrigid, rmass);
    E0 += ipc_softEnergy(soft, nsoft, m, d, x, nfv, rrad, rad, rmass, ih2);   // soft flex-rigid (parity w/ ipc_softTry)
    mjtNum alpha = 1.0;
    int lsok = 0;
    // Line search: plain monotone decrease + 1e-12 slop, OR newton_converged short-circuit; 8 backtracks /2;
    // ALWAYS accept the final trial (no Armijo-break grind).
    for (int ls=0; ls < 8; ls++) {
      for (int i=0; i < 3*nstate; i++) xn[i] = x[i];
      for (int v=0; v < npt; v++) if (fidx[v] >= 0) { int fi = fidx[v];
        for (int c=0; c < 3; c++) xn[3*v+c] = x[3*v+c] + alpha*dx[3*fi+c]; }
      for (int i=0; i < nrigid; i++) { int fi = nfree_flex + i, v = nfv + i;   // rigid line-search update
        for (int c=0; c < 3; c++) xn[3*v+c] = x[3*v+c] + alpha*dx[3*fi+c]; }
      mjtNum Etr = ipc_energy(m, d, npt, ne, elems, nbe, bends, xn, xtil, fidx, mass, h, r, rad, ghat,
                              gv, ge, candLS, nls, xold, xfree, nrigid, rmass);
      Etr += ipc_softEnergy(soft, nsoft, m, d, xn, nfv, rrad, rad, rmass, ih2);   // soft flex-rigid at the trial
      if (Etr <= E0 + 1e-12 || newton_converged) { lsok = 1; break; }
      alpha *= 0.5;
    }
    for (int i=0; i < 3*nstate; i++) x[i] = xn[i];   // always accept the final trial
    lsok = 1;
    if (getenv("MJ_IPC_PROF2"))   // per-inner-iter: gradient drop / active-set / line-search alpha
      fprintf(stderr, "    o=%d it=%d gnorm/g0=%.3e nacon=%d nls=%d alpha=%.4f lsok=%d nc=%d\n",
              outer, it, g0 > 0 ? gnorm/g0 : 1.0, nacon, nls, alpha, lsok, newton_converged);
    // keep wgap a valid lower bound: every gap can shrink by at most 4 vertices * the max vertex step
    mjtNum dgap = 4.0*alpha*maxdx;
    for (int c=0; c < wn; c++) wgap[c] -= dgap;
  }
  // N8 non-penetration advance (every iter): dual ascent (lambda update + cnt) -> re-query@xfree -> CCD -> advance xfree.
  // Order is FIXED:
  //   lambda update (on the persistent set) -> prepare CCD -> detect trajectory candidates(1.0)
  //   -> alpha = CCD time-of-impact filter(1.0) -> active-set update (MERGE/AGING) -> advance non-penetrate positions(alpha)
  //   -> beta = beta + (1-beta)*alpha.
  // N8a lambda update + cnt on the ASSEMBLED persistent set (aset), then sink lam -> g_pal (warm start).
  ipc_flexLamUpdate(m, d, x, xfree, gv, ge, r, rad, ghat, mass, ih2, aset, naset, aheld, g_pal, npt);
  for (int c=0; c < naset; c++) ipc_cntSet(ipc_pairHash(&aset[c]), aset[c].cnt);   // persist cnt across steps
  // N8b prepare CCD: disp = x - xfree (free-dof layout), base = xfree.
  for (int v=0; v < npt; v++) if (fidx[v] >= 0)
    for (int c=0; c < 3; c++) dx[3*fidx[v]+c] = x[3*v+c] - xfree[3*v+c];
  // N8c detect trajectory candidates(1.0): re-query the broad-phase at xfree over the swept segment xfree->x.
  // FIXED collar (3*ghat, maxdisp=0, d_hat expansion + thickness): the swept segment covers gross motion
  // (no-tunnel), and maxdisp=0 keeps the count bounded regardless of |x-xfree| (the old displacement-scaled
  // collar ballooned to 238k candidates when x flung -> NaN). This is the NEW-candidate source for the merge.
  ncand = ipc_candidates(m, d, xfree, gv, ge, ngv, nge, r, rad, 3*ghat, 3*ghat, 0.0,
                         xfree, x, ghat, nfv, npt, fidx, flist, fxadr, nfd, pt2flex, cand, candmax);
  prof_cb++;
  for (int c=0; c < ncand; c++) { mjtNum nn[3], cw[4]; int idv[4], ni;   // gaps at xfree (for CCD + admission)
    cgap[c] = ipc_conGap(&cand[c], m, d, xfree, gv, ge, r, rad, nn, idv, cw, &ni, ghat); }
  // N8d CCD time-of-impact filter(1.0): CCD over the trajectory candidates -> the advance alpha. appr[c] flags each candidate
  // whose full step closes its gap into the active zone (== its individual CCD time-of-impact < 1).
  mjtNum ac = ipc_ccd(m, d, xfree, dx, gv, ge, r, rad, nfv, fidx, cand, ncand, cgap, pt2flex, appr);
  // N8e update_active_set: MERGE the admitted broad-phase candidates into the persistent set (keep existing with
  // abs(cnt)<=25, add new with toi<1-1e-6, dedup). Admit a candidate iff it is closing this step (appr) OR already
  // distance-active (gap<=0): both are the CCD time-of-impact < 1-1e-6 set. New entries seed lam(g_pal)/cnt(store).
  for (int c=0; c < ncand; c++) actc[c] = (appr[c] || cgap[c] <= 0.0) ? 1 : 0;
  ipc_mergeActiveSet(aset, &naset, cand, ncand, actc, amerge, g_pal, candmax);
  // N8f advance non-penetrate positions(alpha): advance the FLEX xfree, iff alpha > alpha lower bound. beta -> 1.
  if (ac > IPC_ALPHA_LB) for (int i=0; i < 3*npt; i++) xfree[i] = (1.0-ac)*xfree[i] + ac*x[i];
  // RIGID bodies carry NO CCD (they are absent from cand -> not throttled by the flex time-of-impact). Advance
  // their xfree FULLY to the kinetic solution each iter so the sphere free-falls regardless of the flex ac.
  for (int i=0; i < nrigid; i++) { int v = nfv + i; for (int c=0; c < 3; c++) xfree[3*v+c] = x[3*v+c]; }
  beta = beta + (1.0 - beta)*ac;
  // terminate: beta feasible (>= 1 - IPC_TOI_THRESH) AND (newton_iter+1 >= min_iter OR newton_converged).
  if (beta >= 1.0 - IPC_TOI_THRESH && (outer+1 >= IPC_FLEX_MIN_ITER || newton_converged_out)) break;
  (void)last_maxdx; (void)beta_eps;
  }   // OUTER loop close
  if (getenv("MJ_IPC_PROF")) {
    static long pstep = 0;
    fprintf(stderr, "IPCPROF step=%ld outer=%d inner=%d pcg=%ld candbuilds=%d ncand=%d nacon=%d nbe=%d beta=%.2e\n",
            pstep++, prof_outer, prof_inner, g_pcgN, prof_cb, ncand, prof_nacon, nbe, beta);
    fprintf(stderr, "        active(fc>0)=%ld of held=%d\n", g_nact, prof_nacon);
    int th[6] = {0,0,0,0,0,0}; for (int c=0; c < ncand; c++) if (cand[c].type>=0 && cand[c].type<6) th[cand[c].type]++;
    fprintf(stderr, "        cand-by-type: t0(VT)=%d t1(EE)=%d t2(flexvert-geom)=%d t3(flex-geom)=%d t4=%d t5(unused)=%d  sp.nnz=%d\n",
            th[0], th[1], th[2], th[3], th[4], th[5], sp.nnz);
    mjtNum mxf=0, mxx=0, mxt=0;   // [PROF] motion budget: where (if anywhere) is the committed state stuck?
    for (int v=0; v < npt; v++) if (fidx[v] >= 0) { mjtNum df=0,dxx=0,dt=0;
      for (int c=0; c<3; c++){ mjtNum a=xfree[3*v+c]-xold[3*v+c]; df+=a*a; mjtNum b=x[3*v+c]-xold[3*v+c]; dxx+=b*b;
                               mjtNum e=xtil[3*v+c]-xold[3*v+c]; dt+=e*e; }
      if(df>mxf)mxf=df; if(dxx>mxx)mxx=dxx; if(dt>mxt)mxt=dt; }
    fprintf(stderr, "        MOVE maxd(xfree-xold)=%.3e maxd(x-xold)=%.3e maxd(xtil-xold)=%.3e\n",
            sqrt(mxf), sqrt(mxx), sqrt(mxt));
  }
  g_pcgN = 0;   // [PROF] reset per step
  for (int i=0; i < 3*nstate; i++) x[i] = xfree[i];   // commit the intersection-free output (readback uses x)
  for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
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
  // RIGID readback (state slot nfv+i): same world->local R^T displacement map. qpos[qa+c] += dpl, qvel = dpl/h.
  for (int i=0; i < nrigid; i++) {
    int v = nfv + i, da = rdofadr[i], qa = rqadr[i];
    const mjtNum* R = d->xmat + 9*rbody[i];
    mjtNum dpw[3], dpl[3];
    for (int c=0; c < 3; c++) dpw[c] = x[3*v+c] - xold[3*v+c];
    mju_mulMatTVec3(dpl, R, dpw);
    for (int c=0; c < 3; c++) { d->qvel[da+c] = dpl[c]/h; d->qpos[qa+c] += dpl[c]; }
  }
  d->time += h;
  if (N > 0) ipc_spFree(&sp);
  mju_free(escat);
  mju_free(dofadr); mju_free(qpadr); mju_free(fidx); mju_free(mass); mju_free(elems); mju_free(bends);
  mju_free(rad); mju_free(pbody); mju_free(pgeom);
  mju_free(rbody); mju_free(rqadr); mju_free(rdofadr); mju_free(rmass); mju_free(rgeom); mju_free(rrad);
  mju_free(soft);
  mju_free(flist); mju_free(fxadr); mju_free(pt2vg); mju_free(pt2flex);
  mju_free(acon); mju_free(ccache); mju_free(cand); mju_free(cgap); mju_free(candLS); mju_free(minc); mju_free(held);
  mju_free(gam); mju_free(actc); mju_free(appr); mju_free(actpt);
  mju_free(aset); mju_free(agap); mju_free(aheld); mju_free(amerge);
  mju_free(gv); mju_free(ge); mju_free(estr);
  mju_free(x); mju_free(xfree); mju_free(xtil); mju_free(xold); mju_free(xn);
  mju_free(grad); mju_free(dx); mju_free(mdiag); mju_free(qacc_pred);
  mju_free(rcg); mju_free(zcg); mju_free(pcg); mju_free(Hpv); mju_free(usol);
}


// -------------------------------------------------------------------------------------------------
// Thin wrappers exposing the internal IPC kernels to the unit tests (engine_ipc_test.cc). These let
// the barrier and the geometry/contact distance functions be checked directly, without stepping a
// model. Not a supported API; kept here so the kernels themselves stay static.

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
