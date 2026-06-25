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
// a CCD-bounded committed position. The rigid/articulated path uses affine-body dynamics.

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

// Contact STANDOFF: every active contact rests at a small positive gap delta = min(ghc, IPC_DELTACAP). The AL
// multiplier holds the constraint there, so the MaxStepSize CCD always sees a positive gap (nonzero TOI, no lock)
// and the committed trajectory stays STRICTLY intersection-free (no penetration). delta is a small geometric skin:
// it shrinks for thin participants (ghc small -> thin cloth keeps its no-penetration guarantee) and is capped so
// big affine bodies don't carry a fat layer. Same rule for flex and affine contacts.
#define IPC_DELTACAP 0.001   // 1 mm standoff cap
#define IPC_JNTLIMIT_KAPPA 1.0e-4   // AL joint-limit stiffness kj = this*kappaO (kappaO = 1e4*I_rot/h^2). ~10x the
                                     // per-body nominal 0.1*I_rot/h^2: a limit resists the descendant chain's torque
                                     // (effective inertia >> body's own I_rot), so this converges it to ~0 in 1-2
                                     // steps; the AL multiplier adapts, so kj only sets the transient ramp (1e-5 lags ~6deg).
#define IPC_CDAMP_FLEX 0.1   // flex-contact normal dashpot coeff (cde = IPC_CDAMP_FLEX*m/h^2); affine uses 0.1
#define IPC_DUAL_OMEGA 1.0   // AL dual-ascent over-relaxation (lam -= OMEGA*kappa*c): faster ramp -> fewer outer iters
#define IPC_AFFINE_GEOM 1   // 1: full Newton (GN+geometric) affine Hessian; 0: Gauss-Newton only (test)
                            // (GN-only ablation did NOT fix the ball-scene grind/NaN -> the affine geometric term is
                            //  NOT the crucial difference. Reverted.)
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

// AL contact multipliers, per affine FEATURE (key = body*g_calStride + feature), persisted across Newton iters
// AND steps (warm start: contact force ~constant frame-to-frame). Read in the contact gradient/energy; updated
// once per accepted Newton step (lam <- max(0, lam - mu*c)). Pure-affine-vs-geom for now.
static mjtNum* g_cal = NULL; static int g_calN = 0, g_calStride = 0;
// AL contact multipliers for affine-AFFINE capsule pairs (key = ((a*naff+b)*g_caalCap+ca)*g_caalCap+cb), persisted
static mjtNum* g_caal = NULL; static int g_caalN = 0, g_caalCap = 0;
// AL contact multiplier per FREE POINT (flex vertex / rigid sphere): the cross-step warm-start store for the
// ipc_try contacts, whose LIVE multiplier rides in ipcCon.lam. Seeded into cand[].lam at step start, sunk back
// after the step (binding = max over a contact's free-point participants). npt-sized.
static mjtNum* g_pal = NULL; static int g_palN = 0;
// AL joint-LIMIT multipliers, per (model joint, side): 2*jid+0 = lower limit, 2*jid+1 = upper. Persisted across
// Newton iters AND steps (warm start). Updated once per accepted step (lam <- max(0, lam - mu*c)). Size 2*njnt.
static mjtNum* g_jal = NULL; static int g_jalN = 0;
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

// PSD projection (per-element symmetric EVD, clamp negative eigenvalues to 0, reconstruct).
// Small 4x4 symmetric block via cyclic Jacobi rotations (the geometric affine-ortho block is a 4x4 in the 4
// control points, replicated block-diagonally over the 3 coords). In/out: A is overwritten by its PSD projection.
static void ipc_makeSPD4(mjtNum A[4][4]) {
  mjtNum V[4][4];
  for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) V[i][j] = (i == j) ? 1.0 : 0.0;
  for (int sweep=0; sweep < 24; sweep++) {
    mjtNum off = 0;
    for (int p=0; p < 4; p++) for (int q=p+1; q < 4; q++) off += A[p][q]*A[p][q];
    if (off < 1e-30) break;
    for (int p=0; p < 4; p++) for (int q=p+1; q < 4; q++) {
      if (A[p][q]*A[p][q] < 1e-30*(A[p][p]*A[p][p] + A[q][q]*A[q][q]) + 1e-300) continue;
      mjtNum theta = (A[q][q] - A[p][p]) / (2.0*A[p][q]);
      mjtNum t = (theta >= 0 ? 1.0 : -1.0) / (mju_abs(theta) + sqrt(theta*theta + 1.0));
      mjtNum cc = 1.0/sqrt(t*t + 1.0), s = t*cc;
      for (int k=0; k < 4; k++) {                            // A <- J^T A J (rotate rows/cols p,q)
        mjtNum akp = A[k][p], akq = A[k][q];
        A[k][p] = cc*akp - s*akq; A[k][q] = s*akp + cc*akq;
      }
      for (int k=0; k < 4; k++) {
        mjtNum apk = A[p][k], aqk = A[q][k];
        A[p][k] = cc*apk - s*aqk; A[q][k] = s*apk + cc*aqk;
      }
      for (int k=0; k < 4; k++) {                            // accumulate eigenvectors V <- V J
        mjtNum vkp = V[k][p], vkq = V[k][q];
        V[k][p] = cc*vkp - s*vkq; V[k][q] = s*vkp + cc*vkq;
      }
    }
  }
  mjtNum lam[4]; for (int i=0; i < 4; i++) { lam[i] = A[i][i]; if (lam[i] < 0) lam[i] = 0; }   // clamp to PSD
  for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) {     // reconstruct A = V diag(max(lam,0)) V^T
    mjtNum s = 0; for (int k=0; k < 4; k++) s += V[i][k]*lam[k]*V[j][k];
    A[i][j] = s;
  }
}

// affine body inertia (M_cp star block) + orthogonality (rigidity) energy over the 4 control points.
// Returns the energy and, optionally (g/H non-NULL), the 12-gradient and 12x12 Gauss-Newton (PSD) Hessian
// in control-point-local order (point k, coord c -> 3k+c). F is linear in the control points so the energy
// is polynomial -- no polar decomposition in the solve. M_cp is the "star" form (cp0 = com): translation
// mode mass = body mass; off-diagonals couple cp0 to each axis point.
// H is the FULL Hessian (inertia + GN orthogonality + geometric); Hgeo (optional) gets just the geometric
// orthogonality part, so the caller can use H for the matvec (true Newton) but H-Hgeo (PD) for the preconditioner.
static mjtNum ipc_affineGH(const ipcAffine* af, const mjtNum* x, const mjtNum* xtil, mjtNum ih2,
                           mjtNum g[12], mjtNum H[12][12], mjtNum Hgeo[12][12]) {
  mjtNum y[4][3], yt[4][3];
  for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) {
    y[k][c] = x[3*af->cp[k]+c]; yt[k][c] = xtil[3*af->cp[k]+c];
  }
  if (g) for (int i=0; i < 12; i++) { g[i] = 0; if (H) for (int j=0; j < 12; j++) H[i][j] = 0; }
  if (Hgeo) for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) Hgeo[i][j] = 0;
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
  // dF/dcp coefficients G[k][col]: cp_{m+1} contributes Dinv[m][col]; cp0 contributes -sum_m Dinv[m][col].
  mjtNum G[4][3];
  for (int col=0; col < 3; col++) { G[0][col] = -(af->Dinv[col]+af->Dinv[3+col]+af->Dinv[6+col]);
    for (int k=1; k < 4; k++) G[k][col] = af->Dinv[3*(k-1)+col]; }
  mjtNum Mgeo[4][4]; for (int i=0; i < 4; i++) for (int j=0; j < 4; j++) Mgeo[i][j] = 0;   // 4x4 geometric block
  for (int q=0; q < 6; q++) {
    int a = ca[q], b = cb[q];
    E += af->kappaO*w[q]*C[q]*C[q];
    if (g) {
      mjtNum gc = 2.0*af->kappaO*w[q]*C[q], hc = 2.0*af->kappaO*w[q];
      for (int i=0; i < 12; i++) {
        g[i] += gc*Jc[q][i];
        if (H) for (int j=0; j < 12; j++) H[i][j] += hc*Jc[q][i]*Jc[q][j];   // Gauss-Newton part (rank-1 PSD)
      }
      // geometric (curvature) part: gc * d2C_q, with d2C_q[3ki+d][3kj+d] = G[ki][a]G[kj][b]+G[kj][a]G[ki][b]
      // (block-diagonal in the coordinate d). C[q] can be <0 and the a!=b shear blocks are structurally
      // INDEFINITE, so accumulate the full 4x4 Mgeo over all 6 components first, then PSD-project it once
      // (PSD projection of the orthogonality-potential geometric block) before it enters the operator -> SPD Newton direction.
#if IPC_AFFINE_GEOM
      if (H) for (int ki=0; ki < 4; ki++) for (int kj=0; kj < 4; kj++)
        Mgeo[ki][kj] += gc*(G[ki][a]*G[kj][b] + G[kj][a]*G[ki][b]);
#endif
    }
  }
#if IPC_AFFINE_GEOM
  if (g && H) {
    if (!getenv("MJ_IPC_NOSPD")) ipc_makeSPD4(Mgeo);         // clamp neg eigvals to 0 (per-element PSD projection); MJ_IPC_NOSPD=1 ablates
    for (int ki=0; ki < 4; ki++) for (int kj=0; kj < 4; kj++)
      for (int d=0; d < 3; d++) { H[3*ki+d][3*kj+d] += Mgeo[ki][kj]; if (Hgeo) Hgeo[3*ki+d][3*kj+d] += Mgeo[ki][kj]; }
  }
#endif
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
        mjtNum dn = dpw[0]*n[0] + dpw[1]*n[1] + dpw[2]*n[2];
        mjtNum cde = (dn < 0) ? cdH : 0;                       // ONE-SIDED dashpot: damp approach (compression)
        // augmented Lagrangian: force = lambda - mu*(g-delta), c = g - delta (rest at the standoff delta)
        mjtNum lam = g_cal[a*g_calStride + ft];               // (read; updated once per accepted Newton step)
        mjtNum fc = lam - kappa*(g - ipc_off(ghc));           // fc = lambda - mu*c
        if (fc <= 0) continue;                                // inactive: no force (lambda decays in the update)
        mjtNum bd = -fc, bdd = kappa + cde;                   // bd = mu*g - lambda; Hessian = mu (+ dashpot)
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

// AL multiplier update, once per accepted Newton step, at the new x: lambda <- max(0, lambda - mu*c), c = the
// binding (min over geoms) gap of each affine feature. Far features (no geom in range) decay to 0.
static void ipc_affineLamUpdate(const mjModel* m, const mjData* d, const ipcAffine* aff, int naff,
                                const mjtNum* x, mjtNum kappa) {
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a]; mjtNum ghc = af->ghatC;
    for (int ft=0; ft < af->nfeat; ft++) {
      mjtNum pw[3] = {0,0,0};
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) pw[c] += af->cfeat[ft][k]*x[3*af->cp[k]+c];
      mjtNum mn = 1e30;
      for (int gi=0; gi < m->ngeom; gi++) {
        if (m->geom_contype[gi]==0 && m->geom_conaffinity[gi]==0) continue;
        if (!ipc_bodyAnchored(m, m->geom_bodyid[gi])) continue;
        mjtNum nn[3], g = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, pw, nn, ghc+af->cfrad[ft])
                         - af->cfrad[ft];
        if (g < mn) mn = g;
      }
      mjtNum* lam = &g_cal[a*g_calStride + ft];
      if (mn < ghc) { mjtNum nl = *lam - IPC_DUAL_OMEGA*kappa*(mn - ipc_off(ghc)); *lam = nl > 0 ? nl : 0; }   // binding gap, rest at delta
      else *lam = 0;                                                            // out of range -> decay
    }
  }
}

// AL joint-LIMIT multiplier update, once per accepted Newton step at the converged angle theta = qpos + thRO:
// lam_side <- max(0, lam_side - OMEGA*kj*c_side); c_hi = hi-theta, c_lo = theta-lo (both >= 0 within range).
static void ipc_jntLimitLamUpdate(const mjModel* m, const mjData* d, const ipcAffine* aff, int naff,
                                  const mjtNum* thRO) {
  for (int a=0; a < naff; a++) { const ipcAffine* af = &aff[a]; if (!af->isHinge) continue;
    mjtNum kj = IPC_JNTLIMIT_KAPPA*af->kappaO;
    for (int jl=0; jl < af->njnt; jl++) { int jid = af->ja + jl;
      if (!m->jnt_limited[jid]) continue;
      mjtNum theta = d->qpos[af->qadr+jl] + thRO[3*a+jl];
      mjtNum lo = m->jnt_range[2*jid], hi = m->jnt_range[2*jid+1];
      mjtNum* lhi = &g_jal[2*jid+1]; mjtNum nhi = *lhi - IPC_DUAL_OMEGA*kj*(hi - theta); *lhi = nhi > 0 ? nhi : 0;
      mjtNum* llo = &g_jal[2*jid+0]; mjtNum nlo = *llo - IPC_DUAL_OMEGA*kj*(theta - lo); *llo = nlo > 0 ? nlo : 0;
    }
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

// flat warm-start index for the affine-affine capsule-pair AL multiplier g_caal (pairs enumerated a<b)
static inline int ipc_aaIdx(int a, int b, int ca, int cb, int naff) {
  return ((a*naff + b)*g_caalCap + ca)*g_caalCap + cb;
}

// affine-affine contact: gradient -> grad (both bodies), GN block -> ccache (8 cps), per-body diagonal -> affHc.
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
      mjtNum dn = 0;                                              // relative approach speed along n
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++)
        dn += (cwA[k]*(x[3*afA->cp[k]+c]-xold[3*afA->cp[k]+c]) + cwB[k]*(x[3*afB->cp[k]+c]-xold[3*afB->cp[k]+c]))*n[c];
      mjtNum cde = (dn < 0) ? ipc_fmin(afA->cdampH, afB->cdampH) : 0;
      // AL: force = lambda - mu*(g-delta), c = g - delta (rest at the standoff delta)
      mjtNum lam = g_caal[ipc_aaIdx(a, b, ca, cb, naff)];
      mjtNum fc = lam - kappa*(g - ipc_off(ghc)); if (fc <= 0) continue;   // inactive: no force (multiplier decays)
      mjtNum bd = -fc, bdd = kappa + cde;                        // bd = mu*g - lambda; Hessian = mu (+ dashpot)
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
      mjtNum dn = 0;
      for (int k=0; k < 4; k++) for (int c=0; c < 3; c++)
        dn += (cwA[k]*(x[3*afA->cp[k]+c]-xold[3*afA->cp[k]+c]) + cwB[k]*(x[3*afB->cp[k]+c]-xold[3*afB->cp[k]+c]))*n[c];
      mjtNum cde = (dn < 0) ? ipc_fmin(afA->cdampH, afB->cdampH) : 0;
      // AL merit (1/2mu)(max(0,lam-mu*g)^2 - lam^2): C1
      mjtNum lam = g_caal[ipc_aaIdx(a, b, ca, cb, naff)];
      mjtNum fc = lam - kappa*(g - ipc_off(ghc)); if (fc < 0) fc = 0;
      E += (fc*fc - lam*lam)/(2*kappa) + (fc > 0 && dn < 0 ? 0.5*cde*dn*dn : 0);
    }
  }
  return E;
}

// AL multiplier update for affine-affine capsule pairs, once per accepted Newton step at the new x:
// g_caal[pair] <- max(0, lambda - mu*g). Pairs out of range (or filtered) decay to 0. Mirrors ipc_affineLamUpdate.
static void ipc_affineAffineLamUpdate(const mjModel* m, const ipcAffine* aff, int naff, const mjtNum* x, mjtNum kappa) {
  for (int a=0; a < naff; a++) for (int b=a+1; b < naff; b++) {
    const ipcAffine *afA = &aff[a], *afB = &aff[b];
    for (int ca=0; ca < afA->ncap; ca++) for (int cb=0; cb < afB->ncap; cb++) {
      mjtNum* lam = &g_caal[ipc_aaIdx(a, b, ca, cb, naff)];
      if (!ipc_affineAACanCollide(m, afA->body, afA->capgeom[ca], afB->body, afB->capgeom[cb])) { *lam = 0; continue; }
      mjtNum n[3], cwA[4], cwB[4];
      mjtNum ghc = ipc_fmin(afA->ghatC, afB->ghatC), g = ipc_affineAAGap(afA, ca, afB, cb, x, n, cwA, cwB);
      if (g < ghc) { mjtNum nl = *lam - IPC_DUAL_OMEGA*kappa*(g - ipc_off(ghc)); *lam = nl > 0 ? nl : 0; }
      else *lam = 0;
    }
  }
}

// conservative additive CCD for affine-affine pairs: cap alpha so no pair closes more than ~90% of its gap.
static mjtNum ipc_affineAffineCCD(const mjModel* m, const ipcAffine* aff, int naff, const mjtNum* x, const mjtNum* dx) {
  mjtNum cap = 1.0;
  for (int a=0; a < naff; a++) for (int b=a+1; b < naff; b++) {
    const ipcAffine *afA = &aff[a], *afB = &aff[b];
    for (int ca=0; ca < afA->ncap; ca++) for (int cb=0; cb < afB->ncap; cb++) {
      if (!ipc_affineAACanCollide(m, afA->body, afA->capgeom[ca], afB->body, afB->capgeom[cb])) continue;
      mjtNum n[3], cwA[4], cwB[4];
      mjtNum g = ipc_affineAAGap(afA, ca, afB, cb, x, n, cwA, cwB);   // true gap; rest at standoff -> CCD has room
      if (g <= 0) continue;                                       // true gap past -delta: the AL recovers it
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
          mjtNum dn = dpw[0]*n[0] + dpw[1]*n[1] + dpw[2]*n[2];
          // AL merit (1/2mu)(max(0,lam-mu*g)^2-lam^2): C1, inactive value = -lam^2/2mu
          mjtNum lam = g_cal[a*g_calStride + ft];
          mjtNum fc = lam - kappa*(g - ipc_off(ghc)); if (fc < 0) fc = 0;
          E += (fc*fc - lam*lam)/(2*kappa) + (fc > 0 && dn < 0 ? 0.5*cdH*dn*dn : 0);
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
        mjtNum n[3], g = ipc_geomDist(m, gi, d->geom_xpos+3*gi, d->geom_xmat+9*gi, pw, n, 1e30)
                         - af->cfrad[ft];   // true gap; rest sits at standoff delta so the CCD has room (no lock)
        if (g <= 0) continue;     // true gap past -delta: the AL recovers it (CCD can't)
        mjtNum rate = dw[0]*n[0] + dw[1]*n[1] + dw[2]*n[2];       // d(gap)/d(alpha)
        if (rate < 0) { mjtNum amx = 0.9*g/(-rate); if (amx < cap) cap = amx; }
      }
    }
  }
  return cap;
}

// joint-passive (spring/damper/limit) x-space Hessian: rank-1 h*(c.x)^2 per hinge angle, c = dtheta/dx over the
// body's (and parent's) control-point DOFs. Applied matrix-free in ipc_applyH; the Nb reduction then projects it.
typedef struct { int ndof; int dof[24]; mjtNum c[24], h; } ipcJpH;

static void ipc_applyH(const mjtNum* p, mjtNum* Hp, int N, const mjtNum* mdiag,
                       int nelem, const ipcElem* elems, const mjtNum* estr,
                       int nbe, const ipcBend* bends,
                       int njp, const ipcJpH* jps,
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
  for (int jt=0; jt < njp; jt++) {                        // joint passive: rank-1 h*(c.p) per hinge angle (matrix-free)
    const ipcJpH* jp = &jps[jt]; mjtNum dp = 0;
    for (int i=0; i < jp->ndof; i++) dp += jp->c[i]*p[jp->dof[i]];
    mjtNum hd = jp->h*dp;
    for (int i=0; i < jp->ndof; i++) Hp[jp->dof[i]] += jp->c[i]*hd;
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
  if (Nna > 0) ipc_jacobiApply(sp, z, r);
  if (Nz > 0)  ipc_crbApply(aff, naff, D, A, K, r + Nna, z + Nna, s, dcp);
}

// reduced matvec Hp = (T^T H T) p:  expand p->dxm via T, apply maximal H (ipc_applyH), reduce via T^T.
static void ipc_uMatvec(const mjtNum* p, mjtNum* Hp, int N, int Nna, int Nz,
                        const ipcAffine* aff, int naff, const mjtNum* mdiag, int ne, const ipcElem* elems,
                        int nbe, const ipcBend* bends,
                        int njp, const ipcJpH* jps,
                        const mjtNum* estr, const ipcCC* ccache, int nacon, const mjtNum* affH,
                        mjtNum* dxm, mjtNum* Hpm, mjtNum* adj) {
  if (naff == 0) {   // pure flex: no reduction, apply the maximal matrix-free Hessian directly (== old ipc_pcg)
    ipc_applyH(p, Hp, N, mdiag, ne, elems, estr, nbe, bends, njp, jps, ccache, nacon, 0, aff, affH); return;
  }
  for (int i=0; i < Nna; i++) dxm[i] = p[i];                    // expand: na identity
  if (naff > 0) {                                               // affine: forward tree map (dcp in Hpm scratch)
    ipc_crbForward(aff, naff, p + Nna, Hpm);
    for (int a=0; a < naff; a++) for (int i=0; i < 12; i++) dxm[Nna+12*a+i] = Hpm[12*a+i];
  }
  ipc_applyH(dxm, Hpm, N, mdiag, ne, elems, estr, nbe, bends, njp, jps, ccache, nacon, naff, aff, affH);
  for (int i=0; i < Nna; i++) Hp[i] = Hpm[i];                   // reduce: na identity
  if (naff > 0) ipc_crbAdjoint(aff, naff, Hpm + Nna, Hp + Nna, adj);   // affine: adjoint tree map
}

// unified preconditioned CG. RHS r_u = [-grad_na ; gz] (gz = the affine reduced gradient RHS from
// ipc_reduceGrad). bjf = prebuilt per-tree factors. Writes the maximal Newton direction dx[N].
static int ipc_solveU(mjtNum* dx, const mjtNum* grad, int N, int Nna, int Nz,
                      const ipcAffine* aff, int naff, const ipcSparse* sp, const mjtNum* mdiag,
                      int ne, const ipcElem* elems, int nbe, const ipcBend* bends, int njp, const ipcJpH* jps,
                      const mjtNum* estr, const ipcCC* ccache, int nacon,
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
    ipc_uMatvec(p, Hp, N, Nna, Nz, aff, naff, mdiag, ne, elems, nbe, bends, njp, jps, estr, ccache, nacon, affH, dxm, Hpm, adj);
    mjtNum pHp = 0; for (int i=0; i < Nu; i++) pHp += p[i]*Hp[i];
    if (pHp <= 1e-30) break;
    mjtNum alpha = rz/pHp, rr = 0;
    for (int i=0; i < Nu; i++) { usol[i] += alpha*p[i]; r[i] -= alpha*Hp[i]; rr += r[i]*r[i]; }
    if (rr < 1e-8*r0) break;
    g_pcgN += 1;   // [PROF] total PCG iterations this step
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

static void ipc_hingeTheta(const ipcAffine* aff, int naff, const mjtNum* bquat,
                           const mjtNum* xold, const mjtNum* x, mjtNum* qoA, mjtNum* qnA, mjtNum* thOut);
static mjtNum ipc_energy(const mjModel* m, const mjData* d, int nfv, int nelem, const ipcElem* elems,
                         int nbe, const ipcBend* bends,
                         const mjtNum* x, const mjtNum* xtil, const int* fidx, const mjtNum* mass,
                         mjtNum h, mjtNum r, const mjtNum* rad, mjtNum ghat, mjtNum kappa,
                         const mjtNum* gv, const mjtNum* ge, const ipcCon* acon, int nacon,
                         const ipcAffine* aff, int naff, const mjtNum* xold, const mjtNum* xfree,
                         mjtNum* jqoA, mjtNum* jqnA, mjtNum* jthRO) {
  mjtNum E = 0, ih2 = 1.0/(h*h);
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum mh = mass[v]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  for (int a=0; a < naff; a++) E += ipc_affineGH(&aff[a], x, xtil, ih2, NULL, NULL, NULL);
  // implicit joint passive energy (spring + damping + range-limit barrier) on the hinge angles at THIS x, so the
  // line search SEES them and refuses to step a joint past its range. Mirrors the reduced-gradient/Hessian terms.
  if (naff > 0 && jqoA) {
    ipc_hingeTheta(aff, naff, d->xquat, xold, x, jqoA, jqnA, jthRO);
    int nojp = getenv("MJ_IPC_NOJP") != NULL;
    for (int a=0; a < naff; a++) { const ipcAffine* af = &aff[a]; if (!af->isHinge) continue;
      for (int jl=0; jl < af->njnt; jl++) {
        int jid = af->ja + jl;
        mjtNum dth = jthRO[3*a+jl], theta = d->qpos[af->qadr+jl] + dth;
        mjtNum k = m->jnt_stiffness[jid], cc = m->dof_damping[af->vadr+jl], ref = m->qpos_spring[af->qadr+jl];
        if (nojp) { k = 0; cc = 0; }
        E += 0.5*k*(theta-ref)*(theta-ref) + 0.5*(cc/h)*dth*dth;
        if (m->jnt_limited[jid] && !nojp) {   // AL limit merit E = 0.5*kj*min(0, c - lam/kj)^2 per side (c >= 0 desired)
          mjtNum lo = m->jnt_range[2*jid], hi = m->jnt_range[2*jid+1], kj = IPC_JNTLIMIT_KAPPA*af->kappaO;
          if (kj > 0) {
            mjtNum chi = (hi - theta) - g_jal[2*jid+1]/kj; if (chi < 0) E += 0.5*kj*chi*chi;   // upper limit
            mjtNum clo = (theta - lo) - g_jal[2*jid+0]/kj; if (clo < 0) E += 0.5*kj*clo*clo;   // lower limit
          }
        }
      }
    }
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
  E += ipc_affineContactEnergy(m, d, aff, naff, x, xold, kappa);
  E += ipc_affineAffineEnergy(m, aff, naff, x, xold, kappa);
  return E;
}


static inline mjtNum ipc_fmin(mjtNum a, mjtNum b) { return a < b ? a : b; }

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
      ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, dfrom, dto, ghat, cand, &nc, candmax);
    }
  }

  // sphere-sphere (type 5): all pairs of rigid points -- nsph is tiny so the O(nsph^2) scan is cheap
  for (int p=nfv; p < npt; p++) for (int q=p+1; q < npt; q++) {
    ipcCon con = {5, {p, q, 0, 0}, -1};
    ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, dfrom, dto, ghat, cand, &nc, candmax);
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
        ipc_addCand(con, m, d, x, gv, ge, r, rad, thp, dfrom, dto, ghat, cand, &nc, candmax); }
    }
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
  int nv = N/3, nnz = 6*nv;   // N == Nna == 3*nfree; 6 lower-tri entries per 3x3 vertex block
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
// potential (inertia + flex edge-stretch energy + affine-body dynamics) with penetration-free contact by a
// barrier-free AUGMENTED-LAGRANGIAN method (paper arXiv 2512.12151) -- the contact multiplier carries the force at a
// fixed low stiffness (no log barrier, no kappa adaptation, no TOI-lock); the inner optimizer is solved freely and
// the committed output is a conservative-CCD blend from the last intersection-free state. Covers flex self-contact
// (vertex-triangle + edge-edge), flex-vs-geom, affine-vs-geom, and affine-affine. Falls back to Euler if no 2D flex.
// contact-stiffness floor: the AL's penalty stiffness mu is auto-set to 0.1*max(inertia diag) each step
// (matched to the system Hessian scale, paper Eq.20); this is only the pre-mdiag init placeholder.
#define IPC_KAPPA0   1000.0

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
// Hinge-angle readback: per affine hinge body, the njnt joint-angle increments th[] from the body's rotation
// (control points xold->x), projected onto its hinge axes by truncated-SVD pseudo-inverse (gimbal-lock-robust).
// Used both per Newton iter (implicit joint spring/damp/limit needs the CURRENT angle theta = qpos + th) and
// at the final readback. thOut[3*a + j] = the j-th hinge increment of affine body a (0 for free/non-hinge).
static void ipc_hingeTheta(const ipcAffine* aff, int naff, const mjtNum* bquat,
                           const mjtNum* xold, const mjtNum* x, mjtNum* qoA, mjtNum* qnA, mjtNum* thOut) {
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a];
    mjtNum cpo[4][3], cpn[4][3];
    for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) { cpo[k][c] = xold[3*af->cp[k]+c]; cpn[k][c] = x[3*af->cp[k]+c]; }
    ipc_affineQuat(af, cpo, bquat+4*af->body, qoA+4*a);
    ipc_affineQuat(af, cpn, bquat+4*af->body, qnA+4*a);
  }
  for (int a=0; a < naff; a++) {
    const ipcAffine* af = &aff[a];
    thOut[3*a] = thOut[3*a+1] = thOut[3*a+2] = 0;
    if (!af->isHinge) continue;
    int k = af->njnt;
    mjtNum wrel[3]; ipc_twistVec(qnA+4*a, qoA+4*a, wrel);
    if (af->jparent >= 0) { mjtNum wp[3]; ipc_twistVec(qnA+4*af->jparent, qoA+4*af->jparent, wp);
      for (int c=0; c < 3; c++) wrel[c] -= wp[c]; }
    mjtNum AtA[9] = {0}, Atb[3] = {0}, th[3] = {0};
    for (int i=0; i < k; i++) {
      for (int j=0; j < k; j++) { mjtNum s = 0; for (int c=0; c < 3; c++) s += af->jaxis[i][c]*af->jaxis[j][c]; AtA[3*i+j] = s; }
      mjtNum s = 0; for (int c=0; c < 3; c++) s += af->jaxis[i][c]*wrel[c]; Atb[i] = s;
    }
    if (k == 1) { th[0] = Atb[0]/AtA[0]; }
    else {
      mjtNum eval[3], evec[9], q4[4];
      mju_eig3(eval, evec, q4, AtA);
      mjtNum tol = 1e-2 * (eval[0] > 0 ? eval[0] : 1.0);
      for (int j=0; j < 3; j++) {
        if (eval[j] <= tol) continue;
        mjtNum vAtb = evec[j]*Atb[0] + evec[3+j]*Atb[1] + evec[6+j]*Atb[2];
        mjtNum cf = vAtb/eval[j];
        th[0] += cf*evec[j]; th[1] += cf*evec[3+j]; th[2] += cf*evec[6+j];
      }
    }
    thOut[3*a] = th[0]; thOut[3*a+1] = th[1]; thOut[3*a+2] = th[2];
  }
}

// single-body hinge angle theta_jl at free-point positions x (relative to xold), for the joint-passive FD Jacobian
// dtheta/dx. Mirrors ipc_hingeTheta's per-body axis projection but for ONE body (+ its parent), so each FD
// perturbation is O(1). Returns the jl-th component of the axis-projected relative twist.
static mjtNum ipc_hingeThetaOne(const ipcAffine* aff, int naff, int a, int jl,
                                const mjtNum* bquat, const mjtNum* xold, const mjtNum* x) {
  const ipcAffine* af = &aff[a];
  mjtNum cpo[4][3], cpn[4][3], qo[4], qn[4], wrel[3];
  for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) { cpo[k][c] = xold[3*af->cp[k]+c]; cpn[k][c] = x[3*af->cp[k]+c]; }
  ipc_affineQuat(af, cpo, bquat+4*af->body, qo);
  ipc_affineQuat(af, cpn, bquat+4*af->body, qn);
  ipc_twistVec(qn, qo, wrel);
  if (af->jparent >= 0) { const ipcAffine* afp = &aff[af->jparent];
    mjtNum cpo2[4][3], cpn2[4][3], qo2[4], qn2[4], wp[3];
    for (int k=0; k < 4; k++) for (int c=0; c < 3; c++) { cpo2[k][c] = xold[3*afp->cp[k]+c]; cpn2[k][c] = x[3*afp->cp[k]+c]; }
    ipc_affineQuat(afp, cpo2, bquat+4*afp->body, qo2);
    ipc_affineQuat(afp, cpn2, bquat+4*afp->body, qn2);
    ipc_twistVec(qn2, qo2, wp);
    for (int c=0; c < 3; c++) wrel[c] -= wp[c];
  }
  int k = af->njnt; mjtNum AtA[9] = {0}, Atb[3] = {0}, th[3] = {0};
  for (int i=0; i < k; i++) {
    for (int j=0; j < k; j++) { mjtNum s = 0; for (int c=0; c < 3; c++) s += af->jaxis[i][c]*af->jaxis[j][c]; AtA[3*i+j] = s; }
    mjtNum s = 0; for (int c=0; c < 3; c++) s += af->jaxis[i][c]*wrel[c]; Atb[i] = s;
  }
  if (k == 1) { th[0] = Atb[0]/AtA[0]; }
  else {
    mjtNum eval[3], evec[9], q4[4];
    mju_eig3(eval, evec, q4, AtA);
    mjtNum tol = 1e-2 * (eval[0] > 0 ? eval[0] : 1.0);
    for (int j=0; j < 3; j++) {
      if (eval[j] <= tol) continue;
      mjtNum vAtb = evec[j]*Atb[0] + evec[3+j]*Atb[1] + evec[6+j]*Atb[2];
      mjtNum cf = vAtb/eval[j];
      th[0] += cf*evec[j]; th[1] += cf*evec[3+j]; th[2] += cf*evec[6+j];
    }
  }
  return th[jl];
}

void mj_IPC(const mjModel* m, mjData* d) {
  mjtNum h = m->opt.timestep;
  mjtNum kappa = IPC_KAPPA0;   // placeholder; set to the auto mu = 0.1*max(inertia diag) once mdiag is known
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
    // penalty path needs ~3x more normal damping to suppress the spring's parametric/impact resonance (a
    // narrow stiffness+excitation match that blew up otherwise); auto-set, no per-scene tuning.
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
  mjtNum* x    = (mjtNum*) mju_malloc(3*npt*sizeof(mjtNum));
  mjtNum* xfree= (mjtNum*) mju_malloc(3*npt*sizeof(mjtNum));   // AL two-state: intersection-free output path (paper x[k])
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
  mjtNum* affH = (mjtNum*) mju_malloc((naff > 0 ? 144*naff : 1)*sizeof(mjtNum)); // per-affine 12x12 Hessian (full)
  mjtNum* affHgeo = (mjtNum*) mju_malloc((naff > 0 ? 144*naff : 1)*sizeof(mjtNum)); // geom part (subtracted for precond)
  mjtNum* estr = (mjtNum*) mju_malloc((ne > 0 ? 12*ne : 1)*sizeof(mjtNum)); // per-element g_a (9) + Me_a (3)
                                                                           // (full, PSD-projected, this Newton iter)

  // Change A (affine impact fix): gravity-only smooth accel for the AFFINE inertial prediction. The affine-body
  // predictor is gravity+external only (q~ = q + h*v + h^2*(g+f_ext)); our d->qacc_smooth carries qfrc_bias's
  // velocity-coupled Coriolis/centrifugal term, which at ground impact feeds a spurious readback velocity back
  // through C(q,v)v and flings the body (the affine analog of the flex elastic double-count). Recompute the bias
  // at ZERO velocity (zero qvel AND cvel -- mj_rne reads both) to isolate gravity g(q), then
  // qacc_pred = M^-1 (qfrc_passive + qfrc_applied + qfrc_actuator - g(q)). naff==0 (flex/balls) is unchanged.
  mjtNum* qacc_pred = (mjtNum*) mju_malloc((m->nv > 0 ? m->nv : 1)*sizeof(mjtNum));
  if (naff > 0) {
    mjtNum* gfrc  = (mjtNum*) mju_malloc((m->nv > 0 ? m->nv : 1)*sizeof(mjtNum));
    mjtNum* vsave = (mjtNum*) mju_malloc((m->nv > 0 ? m->nv : 1)*sizeof(mjtNum));
    mjtNum* cvsav = (mjtNum*) mju_malloc(6*m->nbody*sizeof(mjtNum));
    mju_copy(vsave, d->qvel, m->nv);
    mju_copy(cvsav, d->cvel, 6*m->nbody);
    mju_zero(d->qvel, m->nv);
    mju_zero(d->cvel, 6*m->nbody);
    mj_rne(m, d, 0, gfrc);                       // gravity-only generalized force g(q) (no Coriolis)
    mju_copy(d->qvel, vsave, m->nv);
    mju_copy(d->cvel, cvsav, 6*m->nbody);
    for (int i=0; i < m->nv; i++)
      gfrc[i] = d->qfrc_applied[i] + d->qfrc_actuator[i] - gfrc[i];   // gravity+external only; joint springs/dampers are now IMPLICIT (below)
    mj_solveM(m, d, qacc_pred, gfrc, 1);         // qacc_pred = M^-1 (qfrc_app - g(q)) = gravity-only accel
    mju_free(gfrc); mju_free(vsave); mju_free(cvsav);
  } else {
    mju_copy(qacc_pred, d->qacc_smooth, m->nv);  // flex/balls path unchanged (point masses have no Coriolis)
  }

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
      mju_mulMatVec3(aw, R, qacc_pred + da);
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
    for (int i=0; i < m->nv; i++) qvt[i] = d->qvel[i] + h*qacc_pred[i];
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
  // AL contact stiffness: kappa carries a FIXED, auto-set mu = 0.1 * max(inertia diag) (flex/free mdiag +
  // affine Mcp4*ih2), matched to the system's Hessian scale (paper Eq.20). No per-step adaptation -> no kappa
  // oscillation; non-penetration is the CCD's job (the AL force is finite).
  {
    mjtNum mmax = 0;
    for (int i=0; i < N; i++) if (mdiag[i] > mmax) mmax = mdiag[i];
    for (int a=0; a < naff; a++) for (int k=0; k < 4; k++) { mjtNum md = aff[a].Mcp4[k][k]*ih2;
      if (md > mmax) mmax = md; }
    kappa = 0.1*mmax;   // penalty stiffness mu = 0.1 * max(inertia diag), matched to the Hessian scale (paper Eq.20)
  }
  {   // AL: persistent per-feature contact multiplier (warm-started across steps). Stride = max nfeat.
    int stride = 1; for (int a=0; a < naff; a++) if (aff[a].nfeat > stride) stride = aff[a].nfeat;
    if (g_calN != naff*stride) { mju_free(g_cal); g_calN = naff*stride; g_calStride = stride;
      g_cal = (mjtNum*) mju_malloc((size_t)(g_calN > 0 ? g_calN : 1)*sizeof(mjtNum));
      for (int i=0; i < g_calN; i++) g_cal[i] = 0; }
    if (g_palN != npt) { mju_free(g_pal); g_palN = npt;       // per-free-point flex/sphere warm-start store
      g_pal = (mjtNum*) mju_malloc((size_t)(npt > 0 ? npt : 1)*sizeof(mjtNum));
      for (int i=0; i < npt; i++) g_pal[i] = 0; }            // zero ONLY on resize -> warm-start persists
    int aacap = 1; for (int a=0; a < naff; a++) if (aff[a].ncap > aacap) aacap = aff[a].ncap;
    g_caalCap = aacap;                                       // affine-affine per-capsule-pair multiplier store
    int aaneed = naff*naff*aacap*aacap;
    if (g_caalN != aaneed) { mju_free(g_caal); g_caalN = aaneed;
      g_caal = (mjtNum*) mju_malloc((size_t)(aaneed > 0 ? aaneed : 1)*sizeof(mjtNum));
      for (int i=0; i < aaneed; i++) g_caal[i] = 0; }
    if (g_jalN != 2*m->njnt) { mju_free(g_jal); g_jalN = 2*m->njnt;   // per-(joint,side) AL joint-limit multiplier
      g_jal = (mjtNum*) mju_malloc((size_t)(g_jalN > 0 ? g_jalN : 1)*sizeof(mjtNum));
      for (int i=0; i < g_jalN; i++) g_jal[i] = 0; }                  // zero only on resize -> warm-start persists
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
  // converges in ~1 iteration instead of ~2 -- halving the cost of contact-free steps. NOT for affine bodies:
  // their contacts aren't in ncand, so x~ (the unconstrained rigid step) may penetrate -> start from xold.
  if (ncand == 0 && naff == 0) for (int i=0; i < 3*npt; i++) x[i] = xtil[i];
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
  if (naff == 0) {
    naset = (ncand < candmax) ? ncand : candmax;
    for (int c=0; c < naset; c++) { aset[c] = cand[c]; agap[c] = cgap[c]; }
  }
  // UNIFIED solve: one PCG in reduced coords u = [non-affine maximal DOF [0,Nna) ; affine reduced DOF (Nz)].
  // The affine control points are the last 12*naff DOF of N (contiguous, see setup), so Nna = N - 12*naff is
  // exactly the flex+free-point block. The IC(0) sparse Hessian sp covers ONLY that non-affine block; the
  // affine block is preconditioned by its per-tree reduced-Hessian factor. (Both old paths -- the jointed
  // CRB direct solve and the flex PCG -- are special cases of ipc_solveU below.)
  int Nna = N - 12*naff;
  if (Nna > 0) ipc_spBuild(&sp, Nna);   // per-vertex 3x3 block-diagonal pattern (block-Jacobi precond)
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
                     *crbP = NULL, *crbD = NULL, *crbA = NULL, *crbK = NULL, *crbS = NULL, *crbDcp = NULL,
                     *thRO = NULL, *qoA_ro = NULL, *qnA_ro = NULL;
  ipcJpH* jps = NULL; int njp = 0;   // joint-passive x-space rank-1 Hessian terms (rebuilt each Newton iter)
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
    jps = (ipcJpH*) mju_malloc((size_t)(naff > 0 ? 3*naff : 1)*sizeof(ipcJpH));  // joint passive: <=3 hinge angles/body
    thRO    = (mjtNum*) mju_malloc((size_t)3*naff*sizeof(mjtNum));   // per-iter hinge-angle increments (readback)
    qoA_ro  = (mjtNum*) mju_malloc((size_t)4*naff*sizeof(mjtNum));   // readback quat scratch (xold pose)
    qnA_ro  = (mjtNum*) mju_malloc((size_t)4*naff*sizeof(mjtNum));   // readback quat scratch (current x)
  }
  for (int i=0; i < 3*npt; i++) xfree[i] = xold[i];   // intersection-free output path (paper x[k]), from feasible xold
  // Unified AL Newton loop for BOTH paths (was: affine ran a nested outer(24)/inner(40) grind with
  // product-to-0 beta + Armijo-break, leaving contact/joint forces off-equilibrium -> injected energy at impact).
  // ONE single Newton loop (inner_cap=1), beta ACCUMULATES to 1 (beta += (1-beta)*ac), terminate when
  // beta >= 1 - IPC_TOI_THRESH (0.9) AND (outer+1 >= min_iter OR newton_converged); no floor, no CFL cap, plain
  // monotone-or-converged line search. The affine ASSEMBLY (CRB reduction, ortho/kappaO Hessian, affine contact +
  // joints) is unchanged -- only the loop CONTROL is unified across paths.
  mjtNum beta = 0.0;
  mjtNum beta_eps = 1e-3;     // (legacy; only referenced by the profiler now)
  int inner_cap = 1;
  int outer_cap = 1024;       // newton_max_iter
  mjtNum last_maxdx = 1e30;    // last inner Newton-direction magnitude, exposed to the outer early-out
  int prof_outer=0, prof_inner=0, prof_cb=1, prof_nacon=0;   // MJ_IPC_PROF counters (cb: 1 initial build)
  for (int outer=0; outer < outer_cap && N > 0; outer++) {   // OUTER loop (single AL loop for flex; paper Alg.1)
  prof_outer++;
  g0 = -1;                    // reset the inner-Newton relative stop each outer iteration
  // WORKING SET selection. AFFINE (naff>0): the per-iter broad-phase cand/cgap/held/ncand (UNCHANGED -- the affine
  // path is byte-for-byte identical). FLEX (naff==0): the PERSISTENT active set aset/agap/aheld/naset maintained
  // across outer iters by ipc_mergeActiveSet (the persistent active-set manager). aheld is all-1: assemble/energy/
  // slack/lambda run over the ENTIRE persistent set (the aging eviction -- not a per-iter ld0<ghat test -- bounds it).
  ipcCon* wcon = (naff == 0) ? aset  : cand;
  mjtNum* wgap = (naff == 0) ? agap  : cgap;
  int*    wheld = (naff == 0) ? aheld : held;
  int     wn   = (naff == 0) ? naset : ncand;
  // N1 linearize_constraints (at xfree, Eq.10): ld0/ln/lcw/liv this iter -> c(x) is linear in x.
  for (int c=0; c < wn; c++)
    wcon[c].ld0 = ipc_conGap(&wcon[c], m, d, xfree, gv, ge, r, rad,
                             wcon[c].ln, wcon[c].liv, wcon[c].lcw, &wcon[c].lniv, ghat);
  // RESTART the optimizer from the feasible base each outer iter (flex path). Our design solves a FREE x and advances
  // a SEPARATE feasible shadow xfree; left to accumulate, x runs ahead of xfree whenever the CCD can't follow (ac->0),
  // |x-xfree| grows, the re-query margin 4*maxstep (~line 2667) explodes the broad-phase (ncand 238k) and beta
  // collapses -> the contact-set-explosion NaN. We COMMIT xfree (line ~2724), keeping a single CCD-advanced
  // position, so tying x to the feasible base each iter costs nothing in committed motion and kills the divergence.
  for (int i=0; i < 3*npt; i++) x[i] = xfree[i];   // restart the optimizer from the feasible base each Newton iter
  (void)minc; (void)appr; (void)gam; (void)actc;
  // AFFINE held mask: hold distance-active candidates (ld0<ghat). FLEX: aheld is all-1 (the whole persistent set
  // is assembled; the merge/aging is the active-set manager).
  if (naff > 0) {
    for (int c=0; c < wn; c++) wheld[c] = (wcon[c].ld0 < ghat) ? 1 : 0;
  }
  // N2 update_slack (at x): materialize s[c] and the slack-baked d the assemble (ipc_try) / lambda use. Runs for
  // both paths so the shared flex/sphere contact bookkeeping is consistent (affine-specific contacts are separate).
  ipc_updateSlack(wcon, wn, wheld, x, xfree, rad, ghat, mass, ih2);
  mjtNum gprev = 1e30; int stall = 0;   // inner-Newton stagnation guard (affine path)
  int newton_converged_out = 0;   // (flex) carry the last inner iter's newton_converged to the outer termination
  for (int it=0; it < inner_cap && N > 0; it++) {
    prof_inner++; g_nact = 0;   // [PROF] reset active count; last inner iter's value is read in the profiler print
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
    // affine bodies: inertia + FULL orthogonality Hessian (GN + geometric) -> affH (the matrix-free matvec =
    // true Newton operator). The geometric part is also saved (affHgeo) so the PRECONDITIONER can use the PD
    // Gauss-Newton block (affH - affHgeo); the full block in the preconditioner would lose definiteness and
    // cost CG iterations. affH is NOT scattered into the IC0 sparse Hessian (the affine block is the per-tree factor).
    for (int a=0; a < naff; a++) {
      mjtNum g12[12], H12[12][12], H12geo[12][12];
      ipc_affineGH(&aff[a], x, xtil, ih2, g12, H12, H12geo);
      mjtNum* Hb = affH + 144*a; mjtNum* Hg = affHgeo + 144*a;
      for (int i=0; i < 12; i++) for (int j=0; j < 12; j++) { Hb[12*i+j] = H12[i][j]; Hg[12*i+j] = H12geo[i][j]; }
      for (int i=0; i < 12; i++) grad[aff[a].sdof[i/3] + (i%3)] += g12[i];
    }
    // (hinge joints are eliminated exactly by the reduced coordinates -- the per-body cp gradient/Hessian
    // above are reduced into the free DOFs z inside ipc_solveU; nothing joint-related is added here.)
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
    if (naff > 0) for (int i=0; i < 144*naff; i++) crbHc[i] = 0;   // per-body contact Hessian (affine precond)
    ipc_affineContact(m, d, aff, naff, x, xold, kappa, grad, ccache, &nacon, amax, naff > 0 ? crbHc : NULL);
    int nacon_ground = nacon;   // [DIAG] contacts from body-vs-static-geom (incl. ground)
    ipc_affineAffineContact(m, aff, naff, x, xold, kappa, grad, ccache, &nacon, amax, naff > 0 ? crbHc : NULL);  // self-collision
    if (getenv("MJ_IPC_CSPLIT")) fprintf(stderr, "    CSPLIT ground=%d inter-affine=%d\n", nacon_ground, nacon - nacon_ground);
    if (nacon > prof_nacon) prof_nacon = nacon;
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
      for (int a=0; a < naff; a++) for (int i=0; i < 144; i++)   // BLOCK-JACOBI precond: FULL per-body 12x12 Hessian
        crbHp[144*a+i] = affH[144*a+i] + crbHc[144*a+i];          // (inertia + ortho_GN + PSD-projected geom + contact).
      // The geom block is already PSD-clamped in ipc_affineGH (ipc_makeSPD4), so keeping it makes crbHp MATCH the
      // matvec (affH) -- was affH - affHgeo (dropped geom, a pre-PSD-projection relic), which mismatched the operator and
      // crawled the kappaO PCG past its 200 cap. Matching it converges the affine PCG in a few iters (the
      // per-body 12x12 block-Jacobi preconditioner), so the single-Newton loop converges.
      (void)affHgeo;
      // joint passive (spring + damper + AL limit) in x-space: theta is a nonlinear function of the free points,
      // so assemble dE/dtheta * dtheta/dx into the MAXIMAL gradient (grad) and h*(dtheta/dx)(dtheta/dx)^T as a
      // matrix-free rank-1 term per angle (jps, applied in ipc_applyH); the Nb reduction (ipc_reduceGrad below +
      // the CRB factor) then projects BOTH correctly, including the body-body parent coupling. (Was assembled into
      // the reduced slot z[zadr+jl] assuming z==theta -- FALSE: Nb is an arbitrary null-space basis, so dtheta/dz is
      // spread across columns; that mis-projection softened + mis-aimed the force. See review wm4stxw2z.)
      njp = 0;
      int nojp = getenv("MJ_IPC_NOJP") != NULL;   // [ABLATE] disable spring+damp+limit
      mjtNum thmax = 0;
      ipc_hingeTheta(aff, naff, d->xquat, xold, x, qoA_ro, qnA_ro, thRO);
      if (!nojp) for (int a=0; a < naff; a++) { const ipcAffine* af = &aff[a]; if (!af->isHinge) continue;
        for (int jl=0; jl < af->njnt; jl++) {
          int jid = af->ja + jl;
          mjtNum dth = thRO[3*a+jl], theta = d->qpos[af->qadr+jl] + dth;
          if (theta > thmax) thmax = theta; else if (-theta > thmax) thmax = -theta;
          mjtNum ks = m->jnt_stiffness[jid], cd = m->dof_damping[af->vadr+jl], ref = m->qpos_spring[af->qadr+jl];
          mjtNum dEdth = ks*(theta - ref) + (cd/h)*dth, h2 = ks + cd/h;   // spring + damper: dE/dtheta, d2E/dtheta2
          if (m->jnt_limited[jid]) {                                       // AL limit: dE/dtheta += fc, curvature += kj
            mjtNum lo = m->jnt_range[2*jid], hi = m->jnt_range[2*jid+1], kj = IPC_JNTLIMIT_KAPPA*af->kappaO;
            if (kj > 0) {
              mjtNum fhi = g_jal[2*jid+1] - kj*(hi - theta); if (fhi > 0) { dEdth += fhi; h2 += kj; }   // upper
              mjtNum flo = g_jal[2*jid+0] - kj*(theta - lo); if (flo > 0) { dEdth -= flo; h2 += kj; }   // lower
            }
          }
          if (dEdth == 0 && h2 == 0) continue;
          ipcJpH* jp = &jps[njp]; jp->ndof = 0; jp->h = h2;            // c = dtheta/dx (central FD) over body (+ parent) cps
          const mjtNum eps = 1e-6;   // central diff: O(eps^2), no cancellation (theta ~ O(1); fwd eps=1e-7 lost ~7 digits)
          for (int side=0; side < (af->jparent >= 0 ? 2 : 1); side++) {
            const ipcAffine* ab = side == 0 ? af : &aff[af->jparent];
            for (int c4=0; c4 < 4; c4++) { if (ab->sdof[c4] < 0) continue;   // pinned cp: no solver DOF, no force
              for (int c=0; c < 3; c++) {
                int pdof = 3*ab->cp[c4] + c;     // POINT index: x / ipc_hingeTheta are point-indexed (full)
                int sdof = ab->sdof[c4] + c;     // SOLVER index: grad / Hp / dx are sdof-indexed (compacted by pins)
                mjtNum sv = x[pdof];
                x[pdof] = sv + eps; mjtNum thp = ipc_hingeThetaOne(aff, naff, a, jl, d->xquat, xold, x);
                x[pdof] = sv - eps; mjtNum thm = ipc_hingeThetaOne(aff, naff, a, jl, d->xquat, xold, x);
                x[pdof] = sv;
                mjtNum cv = (thp - thm)/(2.0*eps);
                if (cv != 0 && jp->ndof < 24) { jp->dof[jp->ndof] = sdof; jp->c[jp->ndof] = cv; jp->ndof++; }
              }
            }
          }
          for (int i=0; i < jp->ndof; i++) grad[jp->dof[i]] += dEdth * jp->c[i];   // x-space gradient
          njp++;
        }
      }
      if (getenv("MJ_IPC_JT")) fprintf(stderr, "  JNT thmax=%.3f rad (%.0f deg) njp=%d nojp=%d\n", thmax, thmax*57.3, njp, nojp);
      gaff = ipc_reduceGrad(aff, naff, grad + Nna, crbGz, crbAdj);   // reduce grad (now incl. joint passive) -> crbGz
    }
    gnorm = sqrt(gna2 + gaff*gaff);
    if (g0 < 0) g0 = gnorm;                                   // initial residual (first Newton iteration)
    // AFFINE-only inner-Newton early stops (stagnation + relative residual). The FLEX path has inner_cap=1 and
    // ALWAYS takes its single Newton step, so these are skipped for naff==0.
    if (naff > 0) {
      if (gnorm > 0.9*gprev) { if (++stall >= 2) break; } else stall = 0;   // stagnation -> hand to the outer loop
      gprev = gnorm;
      mjtNum ctol = 1e-6;   // AL multiplier converges only linearly; a tighter 1e-7 just wastes inner iterations
      if (gnorm < ctol*g0 + 1e-9) break;
    }
    // ONE solver: per-tree affine preconditioner factors (from inertia+ortho+contact), then the unified PCG.
    if (naff > 0) ipc_crbFactor(aff, naff, crbHp, crbP, crbD, crbA, crbK);
    ipc_solveU(dx, grad, N, Nna, Nz, aff, naff, &sp, mdiag, ne, elems, nbe, bends, njp, jps, estr, ccache, nacon,
               affH, crbD, crbA, crbK, crbS, crbDcp, crbGz, rcg, zcg, pcg, Hpv, usol, dxm, Hpm, crbAdj);
    mjtNum maxdx = 0;   // max free-vertex displacement of the Newton direction (for the lower-bound decay)
    for (int v=0; v < N/3; v++) { mjtNum n2 = dx[3*v]*dx[3*v]+dx[3*v+1]*dx[3*v+1]+dx[3*v+2]*dx[3*v+2];
                                  if (n2 > maxdx) maxdx = n2; }
    maxdx = sqrt(maxdx);
    last_maxdx = maxdx;   // expose to the outer early-out (system-at-rest test)
    if (naff == 0 && getenv("MJ_IPC_PROF")) {   // [DIAG] Newton-decrement sign: gdx=grad.dx>0 => ascent => indefinite H
      mjtNum gdx = 0; for (int i=0; i < N; i++) gdx += grad[i]*dx[i];
      static long ac = 0; if (gdx > 1e-9 && ac++ < 30)
        fprintf(stderr, "[ASCENT] outer=%d gdx=%+.3e maxdx=%.3e -> indefinite H (PSD clamp insufficient)\n", outer, gdx, maxdx);
    }
    // N6 newton_tolerance (flex path, L-infinity dx checker): newton_converged = L-infinity over ALL dx
    // COMPONENTS <= velocity_tol*dt, evaluated on the SOLVED dx BEFORE line search. Used to OR-accept the full step
    // in the line search and to allow termination once beta is feasible. No separate min_iter floor.
    int newton_converged = 0;
    {   // L-infinity dx checker: L-infinity over dx (both paths now). The kappaO tail drives affine dx->0,
        // which short-circuits the line search (like flex) rather than triggering an early break.
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
    mjtNum E0 = ipc_energy(m, d, npt, ne, elems, nbe, bends, x, xtil, fidx, mass, h, r, rad, ghat, kappa,
                           gv, ge, candLS, nls, aff, naff, xold, xfree, qoA_ro, qnA_ro, thRO);
    mjtNum alpha = 1.0;
    int lsok = 0;
    // Line search for BOTH paths: plain monotone decrease + 1e-12 slop,
    // OR newton_converged short-circuit; 8 backtracks /2; ALWAYS accept the final trial (NO Armijo-break grind --
    // that off-equilibrium rejection was the affine energy injector). The reduced-coordinate dx is already mapped
    // to the control points (CRB forward), so the same step/accept logic applies to affine.
    for (int ls=0; ls < 8; ls++) {
      for (int i=0; i < 3*npt; i++) xn[i] = x[i];
      for (int v=0; v < npt; v++) if (fidx[v] >= 0) { int fi = fidx[v];
        for (int c=0; c < 3; c++) xn[3*v+c] = x[3*v+c] + alpha*dx[3*fi+c]; }
      mjtNum Etr = ipc_energy(m, d, npt, ne, elems, nbe, bends, xn, xtil, fidx, mass, h, r, rad, ghat, kappa,
                              gv, ge, candLS, nls, aff, naff, xold, xfree, qoA_ro, qnA_ro, thRO);
      if (Etr <= E0 + 1e-12 || newton_converged) { lsok = 1; break; }
      alpha *= 0.5;
    }
    for (int i=0; i < 3*npt; i++) x[i] = xn[i];   // always accept the final trial
    lsok = 1;
    if (getenv("MJ_IPC_PROF2"))   // per-inner-iter: gradient drop / active-set / line-search alpha
      fprintf(stderr, "    o=%d it=%d gnorm/g0=%.3e nacon=%d nls=%d alpha=%.4f lsok=%d nc=%d\n",
              outer, it, g0 > 0 ? gnorm/g0 : 1.0, nacon, nls, alpha, lsok, newton_converged);
    // keep wgap a valid lower bound: every gap can shrink by at most 4 vertices * the max vertex step
    mjtNum dgap = 4.0*alpha*maxdx;
    for (int c=0; c < wn; c++) wgap[c] -= dgap;
  }
  // N8 non-penetration advance (every iter): dual ascent (lambda update + cnt) -> re-query@xfree -> CCD -> advance xfree.
  if (naff == 0) {
    // FLEX: non-penetration advance. Order is FIXED:
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
    // N8f advance non-penetrate positions(alpha): advance ONLY xfree, iff alpha > alpha lower bound. beta -> 1.
    if (ac > IPC_ALPHA_LB) for (int i=0; i < 3*npt; i++) xfree[i] = (1.0-ac)*xfree[i] + ac*x[i];
    beta = beta + (1.0 - beta)*ac;
    // terminate: beta feasible (>= 1 - IPC_TOI_THRESH) AND (newton_iter+1 >= min_iter OR newton_converged).
    if (beta >= 1.0 - IPC_TOI_THRESH && (outer+1 >= IPC_FLEX_MIN_ITER || newton_converged_out)) break;
    (void)last_maxdx;
  } else {
    // AFFINE: dual ascent -> re-query cand -> CCD -> product-to-0 beta.
    ipc_affineLamUpdate(m, d, aff, naff, x, kappa);   // dual ascent at the converged optimizer x (affine, Alg.2 tail)
    ipc_affineAffineLamUpdate(m, aff, naff, x, kappa); // dual ascent for affine-affine capsule pairs
    if (naff > 0) { ipc_hingeTheta(aff, naff, d->xquat, xold, x, qoA_ro, qnA_ro, thRO);   // joint angle at converged x
                    ipc_jntLimitLamUpdate(m, d, aff, naff, thRO); }   // AL dual ascent for joint limits
    // N8a update_lambda + cnt (at x): two-branch AL update with the cnt state machine, then sink lam->g_pal.
    ipc_flexLamUpdate(m, d, x, xfree, gv, ge, r, rad, ghat, mass, ih2, cand, ncand, held, g_pal, npt);
    for (int c=0; c < ncand; c++) ipc_cntSet(ipc_pairHash(&cand[c]), cand[c].cnt);   // persist cnt to the store
    // N8c re-query candidates at xfree with a margin covering the actual inner displacement, so EVERY pair the
    // committed CCD step could cross is in the sweep (RE-QUERY EVERY iter -- needed for moving inter-flex contacts).
    mjtNum maxstep = 0;
    for (int v=0; v < npt; v++) if (fidx[v] >= 0) { mjtNum dd = 0;
      for (int k=0; k < 3; k++) { mjtNum t = x[3*v+k]-xfree[3*v+k]; dd += t*t; } if (dd > maxstep) maxstep = dd; }
    maxstep = mju_sqrt(maxstep); (void)maxstep;
    ncand = ipc_candidates(m, d, xfree, gv, ge, ngv, nge, r, rad, 3*ghat, 3*ghat, 0.0,
                           xfree, x, ghat, nfv, npt, fidx, flist, fxadr, nfd, pt2flex, cand, candmax);
    prof_cb++;
    for (int c=0; c < ncand; c++) {   // re-seed lambda from g_pal + re-hydrate cnt from the per-pair store EVERY iter
      int vv[4], nvv; ipc_conVerts(&cand[c], vv, &nvv);
      mjtNum s = 0; for (int q=0; q < nvv; q++) if (g_pal[vv[q]] > s) s = g_pal[vv[q]];
      cand[c].lam = s; cand[c].cnt = ipc_cntGet(ipc_pairHash(&cand[c])); cand[c].s = 0; gam[c] = 1.0;
    }
    // N8b/N8d/N8f prepare CCD (disp = x-xfree, base = xfree) + CCD time-of-impact filter -> alpha, advance ONLY xfree.
    for (int v=0; v < npt; v++) if (fidx[v] >= 0)
      for (int c=0; c < 3; c++) dx[3*fidx[v]+c] = x[3*v+c] - xfree[3*v+c];   // displacement xfree->x (free-dof layout)
    for (int c=0; c < ncand; c++) { mjtNum nn[3], cw[4]; int idv[4], ni;     // candidate gaps at the feasible xfree
      cgap[c] = ipc_conGap(&cand[c], m, d, xfree, gv, ge, r, rad, nn, idv, cw, &ni, ghat); }
    mjtNum ac = ipc_ccd(m, d, xfree, dx, gv, ge, r, rad, nfv, fidx, cand, ncand, cgap, pt2flex, appr);
    { mjtNum a2 = ipc_affineContactCCD(m, d, aff, naff, xfree, dx); if (a2 < ac) ac = a2; }
    { mjtNum a2 = ipc_affineAffineCCD(m, aff, naff, xfree, dx);     if (a2 < ac) ac = a2; }
    for (int i=0; i < 3*npt; i++) xfree[i] = (1.0-ac)*xfree[i] + ac*x[i];
    // beta: ACCUMULATE toward 1 + feasible-AND-converged termination, like flex --
    // was beta*=(1-ac) (shrinks to 0) + beta<beta_eps break, which never reached a clean feasible step.
    beta = beta + (1.0 - beta)*ac;
    if (beta >= 1.0 - IPC_TOI_THRESH && (outer+1 >= IPC_FLEX_MIN_ITER || newton_converged_out)) break;
    (void)last_maxdx; (void)beta_eps;
  }
  }   // OUTER loop close
  if (getenv("MJ_IPC_PROF")) {
    static long pstep = 0;
    fprintf(stderr, "IPCPROF step=%ld outer=%d inner=%d pcg=%ld candbuilds=%d ncand=%d naff=%d nacon=%d nbe=%d beta=%.2e\n",
            pstep++, prof_outer, prof_inner, g_pcgN, prof_cb, ncand, naff, prof_nacon, nbe, beta);
    fprintf(stderr, "        active(fc>0)=%ld of held=%d\n", g_nact, prof_nacon);
    int th[6] = {0,0,0,0,0,0}; for (int c=0; c < ncand; c++) if (cand[c].type>=0 && cand[c].type<6) th[cand[c].type]++;
    fprintf(stderr, "        cand-by-type: t0(sph-flex)=%d t1(EE)=%d t2(pt-geom)=%d t3(flex-geom)=%d t4=%d t5(sph-sph)=%d  sp.nnz=%d\n",
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
  for (int i=0; i < 3*npt; i++) x[i] = xfree[i];   // commit the intersection-free output (readback uses x)
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
      else {
        // k=2/3: truncated-SVD pseudo-inverse of the axis Gram matrix AtA. At gimbal lock the stacked hinge
        // axes become near-coplanar, AtA -> singular, and the naive inverse amplifies the small committed-
        // rotation noise into a huge angle increment (det(AtA)~3e-5 -> th~10rad -> qvel~5000). Dropping the
        // near-null eigendirection gives the minimum-norm angle that reproduces the well-determined rotation.
        // Well-conditioned joints are unchanged (all eigenvalues survive the threshold).
        mjtNum eval[3], evec[9], q4[4];
        mju_eig3(eval, evec, q4, AtA);                  // eval DECREASING; evec columns = eigenvectors (row-major)
        mjtNum tol = 1e-2 * (eval[0] > 0 ? eval[0] : 1.0);
        th[0] = th[1] = th[2] = 0;
        for (int j=0; j < 3; j++) {
          if (eval[j] <= tol) continue;                 // skip gimbal-lock null direction
          mjtNum vAtb = evec[j]*Atb[0] + evec[3+j]*Atb[1] + evec[6+j]*Atb[2];
          mjtNum cf = vAtb/eval[j];
          th[0] += cf*evec[j]; th[1] += cf*evec[3+j]; th[2] += cf*evec[6+j];
        }
      }
      if (getenv("MJ_IPC_HINGE")) {
        mjtNum mth = 0; for (int i=0; i < k; i++) { mjtNum a2 = th[i] < 0 ? -th[i] : th[i]; if (a2 > mth) mth = a2; }
        if (mth/h > 50.0) {
          mjtNum det = (k==1) ? AtA[0] : (k==2) ? (AtA[0]*AtA[4]-AtA[1]*AtA[3]) :
            (AtA[0]*(AtA[4]*AtA[8]-AtA[5]*AtA[7])-AtA[1]*(AtA[3]*AtA[8]-AtA[5]*AtA[6])+AtA[2]*(AtA[3]*AtA[7]-AtA[4]*AtA[6]));
          fprintf(stderr, "HINGE a=%d body=%d k=%d det(AtA)=%.3e |wrel|=%.3e max|th|=%.3e qvel=%.1f\n",
                  a, af->body, k, det, mju_norm3(wrel), mth, mth/h);
        }
      }
      for (int i=0; i < k; i++) { d->qpos[af->qadr+i] += th[i]; d->qvel[af->vadr+i] = th[i]/h; }
    }
  }
  mju_free(qoA); mju_free(qnA);
  d->time += h;
  if (naff > 0) { mju_free(crbGz); mju_free(crbAdj); mju_free(crbHp); mju_free(crbHc);
                mju_free(crbP); mju_free(crbD); mju_free(crbA); mju_free(crbK); mju_free(crbS); mju_free(crbDcp);
                mju_free(thRO); mju_free(qoA_ro); mju_free(qnA_ro); mju_free(jps); }
  if (Nna > 0) ipc_spFree(&sp);
  mju_free(escat);
  mju_free(dofadr); mju_free(qpadr); mju_free(fidx); mju_free(mass); mju_free(elems); mju_free(bends);
  mju_free(rad); mju_free(pbody); mju_free(pgeom); mju_free(pcp); mju_free(aff);
  mju_free(xpos_pred); mju_free(xquat_pred); mju_free(isAff); mju_free(bidx);
  mju_free(flist); mju_free(fxadr); mju_free(pt2vg); mju_free(pt2flex);
  mju_free(acon); mju_free(ccache); mju_free(cand); mju_free(cgap); mju_free(candLS); mju_free(minc); mju_free(held);
  mju_free(gam); mju_free(actc); mju_free(appr); mju_free(actpt);
  mju_free(aset); mju_free(agap); mju_free(aheld); mju_free(amerge);
  mju_free(gv); mju_free(ge); mju_free(estr); mju_free(affH); mju_free(affHgeo);
  mju_free(x); mju_free(xfree); mju_free(xtil); mju_free(xold); mju_free(xn);
  mju_free(grad); mju_free(dx); mju_free(mdiag); mju_free(qacc_pred);
  mju_free(rcg); mju_free(zcg); mju_free(pcg); mju_free(Hpv); mju_free(usol); mju_free(dxm); mju_free(Hpm);
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
