// Copyright 2026 DeepMind Technologies Limited
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
#include <stdio.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#include "engine/engine_core_smooth.h"
#include "engine/engine_derivative.h"
#include "engine/engine_memory.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_support.h"
#include "engine/engine_thread.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"

//---------------------------------- closest-point primitives -------------------------------------

// closest points between segments p1q1 and p2q2: parameters s, t in [0, 1] (Ericson 5.1.9)
static void segSegST(const mjtNum p1[3], const mjtNum q1[3], const mjtNum p2[3],
                     const mjtNum q2[3], mjtNum* s, mjtNum* t) {
  mjtNum d1[3], d2[3], r[3];
  mju_sub3(d1, q1, p1);
  mju_sub3(d2, q2, p2);
  mju_sub3(r, p1, p2);
  mjtNum a = mju_dot3(d1, d1);
  mjtNum e = mju_dot3(d2, d2);
  mjtNum f = mju_dot3(d2, r);

  if (a < mjMINVAL && e < mjMINVAL) {
    *s = *t = 0;
    return;
  }
  if (a < mjMINVAL) {
    *s = 0;
    *t = f/e;
  } else {
    mjtNum c = mju_dot3(d1, r);
    if (e < mjMINVAL) {
      *t = 0;
      *s = -c/a;
    } else {
      mjtNum b = mju_dot3(d1, d2);
      mjtNum denom = a*e - b*b;
      *s = denom > mjMINVAL ? (b*f - c*e)/denom : 0;
      *s = *s < 0 ? 0 : (*s > 1 ? 1 : *s);
      *t = (b*(*s) + f)/e;
      if (*t < 0) {
        *t = 0;
        *s = -c/a;
      } else if (*t > 1) {
        *t = 1;
        *s = (b - c)/a;
      }
    }
  }
  *s = *s < 0 ? 0 : (*s > 1 ? 1 : *s);
  *t = *t < 0 ? 0 : (*t > 1 ? 1 : *t);
}


// closest point on triangle abc to point p: barycentric coordinates (Ericson 5.1.5)
static void ptTriBary(const mjtNum p[3], const mjtNum a[3], const mjtNum b[3],
                      const mjtNum c[3], mjtNum bary[3]) {
  mjtNum ab[3], ac[3], ap[3], bp[3], cp[3];
  mju_sub3(ab, b, a);
  mju_sub3(ac, c, a);
  mju_sub3(ap, p, a);
  mjtNum d1 = mju_dot3(ab, ap);
  mjtNum d2 = mju_dot3(ac, ap);
  if (d1 <= 0 && d2 <= 0) {
    bary[0] = 1; bary[1] = 0; bary[2] = 0;
    return;
  }

  mju_sub3(bp, p, b);
  mjtNum d3 = mju_dot3(ab, bp);
  mjtNum d4 = mju_dot3(ac, bp);
  if (d3 >= 0 && d4 <= d3) {
    bary[0] = 0; bary[1] = 1; bary[2] = 0;
    return;
  }

  mjtNum vc = d1*d4 - d3*d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    mjtNum v = d1/(d1 - d3);
    bary[0] = 1 - v; bary[1] = v; bary[2] = 0;
    return;
  }

  mju_sub3(cp, p, c);
  mjtNum d5 = mju_dot3(ab, cp);
  mjtNum d6 = mju_dot3(ac, cp);
  if (d6 >= 0 && d5 <= d6) {
    bary[0] = 0; bary[1] = 0; bary[2] = 1;
    return;
  }

  mjtNum vb = d5*d2 - d1*d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    mjtNum w = d2/(d2 - d6);
    bary[0] = 1 - w; bary[1] = 0; bary[2] = w;
    return;
  }

  mjtNum va = d3*d6 - d5*d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    mjtNum w = (d4 - d3)/((d4 - d3) + (d5 - d6));
    bary[0] = 0; bary[1] = 1 - w; bary[2] = w;
    return;
  }

  mjtNum denom = 1/(va + vb + vc);
  mjtNum v = vb*denom, w = vc*denom;
  bary[0] = 1 - v - w; bary[1] = v; bary[2] = w;
}


// segment pq crosses the interior of triangle abc (Moller-Trumbore on a segment)
static int segTriCross(const mjtNum p[3], const mjtNum q[3], const mjtNum a[3],
                       const mjtNum b[3], const mjtNum c[3]) {
  mjtNum e1[3], e2[3], dir[3], h[3], s[3], qv[3];
  mju_sub3(e1, b, a);
  mju_sub3(e2, c, a);
  mju_sub3(dir, q, p);
  mju_cross(h, dir, e2);
  mjtNum det = mju_dot3(e1, h);
  if (fabs(det) < mjMINVAL) {
    return 0;
  }
  mjtNum inv = 1/det;
  mju_sub3(s, p, a);
  mjtNum u = mju_dot3(s, h)*inv;
  if (u < 0 || u > 1) {
    return 0;
  }
  mju_cross(qv, s, e1);
  mjtNum v = mju_dot3(dir, qv)*inv;
  if (v < 0 || u + v > 1) {
    return 0;
  }
  mjtNum t = mju_dot3(e2, qv)*inv;
  return t >= 0 && t <= 1;
}


//---------------------------------- element distance ---------------------------------------------

// candidate closest-feature pair: squared distance and barycentric witness on each element
typedef struct {
  mjtNum dist2;
  mjtNum bary1[3];
  mjtNum bary2[3];
} mjIPCFeature;

// evaluate a candidate given witness barycentrics, keep it if closer
static void considerFeature(mjIPCFeature* best, const mjtNum* vert1, int nv1,
                            const mjtNum* vert2, int nv2, const mjtNum bary1[3],
                            const mjtNum bary2[3]) {
  mjtNum w1[3] = {0, 0, 0}, w2[3] = {0, 0, 0};
  for (int i=0; i < nv1; i++) {
    mju_addToScl3(w1, vert1 + 3*i, bary1[i]);
  }
  for (int i=0; i < nv2; i++) {
    mju_addToScl3(w2, vert2 + 3*i, bary2[i]);
  }
  mjtNum diff[3];
  mju_sub3(diff, w1, w2);
  mjtNum dist2 = mju_dot3(diff, diff);
  if (dist2 < best->dist2) {
    best->dist2 = dist2;
    mju_copy3(best->bary1, bary1);
    mju_copy3(best->bary2, bary2);
  }
}

// segment-segment candidate
static void candSegSeg(mjIPCFeature* best, const mjtNum* vert1, int i0, int i1,
                       const mjtNum* vert2, int j0, int j1, int nv1, int nv2) {
  mjtNum s, t;
  segSegST(vert1 + 3*i0, vert1 + 3*i1, vert2 + 3*j0, vert2 + 3*j1, &s, &t);
  mjtNum bary1[3] = {0, 0, 0}, bary2[3] = {0, 0, 0};
  bary1[i0] = 1 - s;
  bary1[i1] = s;
  bary2[j0] = 1 - t;
  bary2[j1] = t;
  considerFeature(best, vert1, nv1, vert2, nv2, bary1, bary2);
}

// vertex i of one side against the triangle on the other side; tri_is_1 selects which of
// the original sides holds the triangle; vert1/vert2/nv1/nv2 keep the caller's order
static void candVertTri(mjIPCFeature* best, const mjtNum* vert1, int nv1, const mjtNum* vert2,
                        int nv2, int i, int tri_is_1) {
  mjtNum bary[3];
  mjtNum bp[3] = {0, 0, 0};
  bp[i] = 1;
  if (tri_is_1) {
    ptTriBary(vert2 + 3*i, vert1, vert1 + 3, vert1 + 6, bary);
    considerFeature(best, vert1, nv1, vert2, nv2, bary, bp);
  } else {
    ptTriBary(vert1 + 3*i, vert2, vert2 + 3, vert2 + 6, bary);
    considerFeature(best, vert1, nv1, vert2, nv2, bp, bary);
  }
}

// distance between two flex elements with witness points and gradient, see header
mjtNum mj_ipcElemDistance(const mjtNum* vert1, int dim1, const mjtNum* vert2, int dim2,
                          mjtNum witness[6], mjtNum* grad) {
  int nv1 = dim1 + 1, nv2 = dim2 + 1;
  static const int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
  mjIPCFeature best;
  best.dist2 = mjMAXVAL;
  mju_zero3(best.bary1);
  mju_zero3(best.bary2);

  // crossing tests: a segment piercing a triangle interior has distance zero
  int cross = 0;
  if (dim1 == 1 && dim2 == 2) {
    cross = segTriCross(vert1, vert1 + 3, vert2, vert2 + 3, vert2 + 6);
  } else if (dim1 == 2 && dim2 == 1) {
    cross = segTriCross(vert2, vert2 + 3, vert1, vert1 + 3, vert1 + 6);
  } else if (dim1 == 2 && dim2 == 2) {
    for (int k=0; k < 3 && !cross; k++) {
      cross = segTriCross(vert1 + 3*edges[k][0], vert1 + 3*edges[k][1],
                          vert2, vert2 + 3, vert2 + 6) ||
              segTriCross(vert2 + 3*edges[k][0], vert2 + 3*edges[k][1],
                          vert1, vert1 + 3, vert1 + 6);
    }
  }
  if (cross) {
    if (witness) {
      mju_zero(witness, 6);
    }
    if (grad) {
      mju_zero(grad, 3*(nv1 + nv2));
    }
    return 0;
  }

  // enumerate closest-feature candidates by dimension pair
  if (dim1 == 1 && dim2 == 1) {
    candSegSeg(&best, vert1, 0, 1, vert2, 0, 1, nv1, nv2);
  } else if (dim1 == 1 && dim2 == 2) {
    for (int k=0; k < 3; k++) {
      candSegSeg(&best, vert1, 0, 1, vert2, edges[k][0], edges[k][1], nv1, nv2);
    }
    candVertTri(&best, vert1, nv1, vert2, nv2, 0, /*tri_is_1=*/0);
    candVertTri(&best, vert1, nv1, vert2, nv2, 1, /*tri_is_1=*/0);
  } else if (dim1 == 2 && dim2 == 1) {
    for (int k=0; k < 3; k++) {
      candSegSeg(&best, vert1, edges[k][0], edges[k][1], vert2, 0, 1, nv1, nv2);
    }
    candVertTri(&best, vert1, nv1, vert2, nv2, 0, /*tri_is_1=*/1);
    candVertTri(&best, vert1, nv1, vert2, nv2, 1, /*tri_is_1=*/1);
  } else if (dim1 == 2 && dim2 == 2) {
    for (int k1=0; k1 < 3; k1++) {
      for (int k2=0; k2 < 3; k2++) {
        candSegSeg(&best, vert1, edges[k1][0], edges[k1][1],
                   vert2, edges[k2][0], edges[k2][1], nv1, nv2);
      }
    }
    for (int i=0; i < 3; i++) {
      candVertTri(&best, vert1, nv1, vert2, nv2, i, /*tri_is_1=*/0);
      candVertTri(&best, vert1, nv1, vert2, nv2, i, /*tri_is_1=*/1);
    }
  } else {
    mjERROR("unsupported element dimensions %d and %d", dim1, dim2);
  }

  // witness points from barycentrics
  mjtNum w1[3] = {0, 0, 0}, w2[3] = {0, 0, 0};
  for (int i=0; i < nv1; i++) {
    mju_addToScl3(w1, vert1 + 3*i, best.bary1[i]);
  }
  for (int i=0; i < nv2; i++) {
    mju_addToScl3(w2, vert2 + 3*i, best.bary2[i]);
  }
  if (witness) {
    mju_copy3(witness, w1);
    mju_copy3(witness + 3, w2);
  }

  mjtNum dist = mju_sqrt(best.dist2);

  // gradient: d = |w1 - w2|, dd/dvert1_i = bary1_i * n, dd/dvert2_j = -bary2_j * n
  if (grad) {
    mju_zero(grad, 3*(nv1 + nv2));
    if (dist > mjMINVAL) {
      mjtNum n[3];
      mju_sub3(n, w1, w2);
      mju_scl3(n, n, 1/dist);
      for (int i=0; i < nv1; i++) {
        mju_scl3(grad + 3*i, n, best.bary1[i]);
      }
      for (int i=0; i < nv2; i++) {
        mju_scl3(grad + 3*(nv1 + i), n, -best.bary2[i]);
      }
    }
  }
  return dist;
}


//---------------------------------- barrier ------------------------------------------------------

// IPC barrier with thickness offset, see header
mjtNum mj_ipcBarrier(mjtNum d, mjtNum xi, mjtNum dhat, mjtNum* db, mjtNum* ddb) {
  mjtNum s = d - xi;

  // inactive region
  if (s >= dhat) {
    if (db) *db = 0;
    if (ddb) *ddb = 0;
    return 0;
  }

  // infeasible region: clamp to a large finite value (callers must keep s > 0)
  if (s < mjMINVAL) {
    if (db) *db = -mjMAXVAL;
    if (ddb) *ddb = mjMAXVAL;
    return mjMAXVAL;
  }

  // b(s) = -(s - dhat)^2 * log(s/dhat)
  mjtNum sd = s - dhat;
  mjtNum logs = mju_log(s/dhat);
  if (db) {
    *db = -2*sd*logs - sd*sd/s;
  }
  if (ddb) {
    *ddb = -2*logs - 4*sd/s + sd*sd/(s*s);
  }
  return -sd*sd*logs;
}


#define mjIPC_HANDOFF    0.1    // force-taper depth of the extended barrier (fraction of
                                // dhat): pairs penetrating deeper feel zero force and
                                // constant energy, leaving them to the legacy fallback
#define mjIPC_SFLOOR     0.01   // barrier extension point (fraction of dhat)

// barrier extended below the floor gap: the outward force tapers linearly to zero at
// depth mjIPC_HANDOFF*dhat, and the energy plateaus beyond. this keeps the energy
// continuous and monotone over the whole penetration range: marginally penetrating
// pairs are ejected (the legacy contact force vanishes at zero depth, so without the
// extension the handoff deadlocks exactly at the contact distance), deeply penetrating
// pairs are force-neutral for the solve (the legacy fallback owns them), and there is
// no energy cliff at any ownership boundary for the line search to veto
static mjtNum extBarrier(mjtNum dist, mjtNum xi, mjtNum dhat, mjtNum* db, mjtNum* ddb) {
  mjtNum sfloor = mjIPC_SFLOOR*dhat;
  mjtNum s = dist - xi;
  if (s >= sfloor) {
    return mj_ipcBarrier(dist, xi, dhat, db, ddb);
  }
  if (ddb) {
    *ddb = 0;
  }
  mjtNum db0;
  mjtNum b0 = mj_ipcBarrier(xi + sfloor, xi, dhat, &db0, NULL);
  mjtNum L = (mjIPC_SFLOOR + mjIPC_HANDOFF)*dhat;
  if (s <= sfloor - L) {
    if (db) {
      *db = 0;
    }
    return b0 - db0*L/2;
  }
  mjtNum u = (sfloor - s)/L;
  if (db) {
    *db = db0*(1 - u);
  }
  return b0 - db0*L*(u - u*u/2);
}


#define mjIPC_STRETCHK   0.0    // DISABLED pending the variational stretch energy.
                                // three formulations were tried and falsified: no term
                                // parks draped sliding (tug-of-war with equality rows);
                                // an xhat-anchored deviation metric is non-conservative
                                // and pumps energy into parked contacts; rest-length
                                // springs usurp the equality rows on loaded assets
                                // (slow, drifting, and they break friction grasps).
                                // stretch must be owned by the variational stage with
                                // the equality rows retired for solvable flexes

// stretch deviation metric: an edge whose length is constrained upstream (edge
// equality, solved by the convex solver and already absorbed into xhat) is measured
// against the length it has AT xhat, not against its rest length. the energy and its
// gradient vanish at x = xhat, preserving the projection identity, while corrections
// that stretch the material relative to the upstream answer are penalized: without
// this term the barrier projection is blind to inextensibility and parks draped
// sliding configurations in a tug-of-war with the equality rows
typedef struct {
  int s1, s2;            // moving-vertex slots; s2 < 0 means the second end is fixed
  mjtNum fixedpos[3];    // position of the fixed end (valid when s2 < 0)
  mjtNum k;              // metric stiffness
  mjtNum Lhat;           // edge length at xhat
} mjIPCEdge;

// energy of the stretch deviation metric at packed positions x
static mjtNum stretchEnergy(const mjIPCEdge* edges, int nedge, const mjtNum* x) {
  mjtNum E = 0;
  for (int e=0; e < nedge; e++) {
    mjtNum dvec[3];
    const mjtNum* p2 = edges[e].s2 >= 0 ? x + 3*edges[e].s2 : edges[e].fixedpos;
    mju_sub3(dvec, x + 3*edges[e].s1, p2);
    mjtNum c = mju_norm3(dvec) - edges[e].Lhat;
    E += 0.5*edges[e].k*c*c;
  }
  return E;
}


//---------------------------------- broadphase ---------------------------------------------------

// flex participates in IPC collision (as moving or fixed geometry): segment/triangle
// elements; rigid and interpolated flexes collide as prescribed obstacles
static int ipcEligible(const mjModel* m, int f) {
  return m->flex_dim[f] == 1 || m->flex_dim[f] == 2;
}

// flex vertices are solved as DOFs: direct vertex representation only (flex_interp == 0,
// the standard cloth case); rigid and interpolated flexes are never solved
static int ipcSolvable(const mjModel* m, int f) {
  return ipcEligible(m, f) && !m->flex_interp[f] && !m->flex_rigid[f];
}

// global vertex ids of element e of flex f, returns count
static int elemVerts(const mjModel* m, int f, int e, int vid[3]) {
  int dim = m->flex_dim[f];
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim + 1);
  for (int i=0; i <= dim; i++) {
    vid[i] = m->flex_vertadr[f] + edata[i];
  }
  return dim + 1;
}

// two elements share a vertex (only possible within the same flex)
static int shareVertex(const mjModel* m, int f1, int e1, int f2, int e2) {
  if (f1 != f2) {
    return 0;
  }
  int v1[3], v2[3];
  int n1 = elemVerts(m, f1, e1, v1);
  int n2 = elemVerts(m, f2, e2, v2);
  for (int i=0; i < n1; i++) {
    for (int j=0; j < n2; j++) {
      if (v1[i] == v2[j]) {
        return 1;
      }
    }
  }
  return 0;
}

// pair of flexes can produce IPC contacts, see header
int mj_ipcPairEligible(const mjModel* m, int f1, int f2) {
  if (!ipcEligible(m, f1) || !ipcEligible(m, f2)) {
    return 0;
  }
  if (f1 == f2) {
    return m->flex_selfcollide[f1] != mjFLEXSELF_NONE;
  }
  return (m->flex_contype[f1] & m->flex_conaffinity[f2]) ||
         (m->flex_contype[f2] & m->flex_conaffinity[f1]);
}

// spatial-hash broadphase over flex elements, see header
int mj_ipcBroadphase(const mjModel* m, mjData* d, const mjtNum* vertpos,
                     const mjtNum* vertpos2, mjtNum margin, mjIPCPair* pairs, int npairmax) {
  int nflex = m->nflex;
  const mjtNum* pos = vertpos ? vertpos : d->flexvert_xpos;

  // count eligible elements
  int nelem = 0;
  for (int f=0; f < nflex; f++) {
    if (ipcEligible(m, f)) {
      nelem += m->flex_elemnum[f];
    }
  }
  if (!nelem) {
    return 0;
  }

  mj_markStack(d);

  // element table: flex id, element id, AABB
  int* eflex = mjSTACKALLOC(d, nelem, int);
  int* eid = mjSTACKALLOC(d, nelem, int);
  mjtNum* aabb = mjSTACKALLOC(d, 6*nelem, mjtNum);
  int ne = 0;
  mjtNum cell = mjMINVAL;
  for (int f=0; f < nflex; f++) {
    if (!ipcEligible(m, f)) {
      continue;
    }
    mjtNum inflate = m->flex_radius[f] + margin;
    for (int e=0; e < m->flex_elemnum[f]; e++) {
      int vid[3];
      int nv = elemVerts(m, f, e, vid);
      mjtNum* bb = aabb + 6*ne;
      mjtNum sbb[6];  // static (unswept) box: sets the grid resolution
      mju_copy3(bb, pos + 3*vid[0]);
      mju_copy3(bb + 3, pos + 3*vid[0]);
      mju_copy3(sbb, pos + 3*vid[0]);
      mju_copy3(sbb + 3, pos + 3*vid[0]);
      for (int i=0; i < nv; i++) {
        for (int set=0; set < (vertpos2 ? 2 : 1); set++) {
          const mjtNum* p = set ? vertpos2 + 3*vid[i] : pos + 3*vid[i];
          for (int k=0; k < 3; k++) {
            bb[k] = p[k] < bb[k] ? p[k] : bb[k];
            bb[3 + k] = p[k] > bb[3 + k] ? p[k] : bb[3 + k];
            if (!set) {
              sbb[k] = p[k] < sbb[k] ? p[k] : sbb[k];
              sbb[3 + k] = p[k] > sbb[3 + k] ? p[k] : sbb[3 + k];
            }
          }
        }
      }
      for (int k=0; k < 3; k++) {
        bb[k] -= inflate;
        bb[3 + k] += inflate;
        // grid resolution from static extents: a single fast element must not coarsen
        // the grid for everyone (swept boxes simply span more cells)
        mjtNum ext = sbb[3 + k] - sbb[k] + 2*inflate;
        cell = ext > cell ? ext : cell;
      }
      eflex[ne] = f;
      eid[ne] = e;
      ne++;
    }
  }

  // hash table: heads + linked entries; static boxes span at most 8 cells, swept boxes
  // as many as their motion covers (counted exactly in a first pass)
  int tabsize = 1;
  while (tabsize < 4*ne) {
    tabsize *= 2;
  }
  int* head = mjSTACKALLOC(d, tabsize, int);
  for (int i=0; i < tabsize; i++) {
    head[i] = -1;
  }

  int nentrymax = 0;
  for (int i=0; i < ne; i++) {
    const mjtNum* bb = aabb + 6*i;
    int span = 1;
    for (int k=0; k < 3; k++) {
      span *= (int)floor(bb[3 + k]/cell) - (int)floor(bb[k]/cell) + 1;
    }
    nentrymax += span;
  }
  int* next = mjSTACKALLOC(d, nentrymax, int);
  int* entry = mjSTACKALLOC(d, nentrymax, int);

  #define IPC_CELLHASH(ix, iy, iz) \
    ((unsigned int)((ix)*73856093 ^ (iy)*19349663 ^ (iz)*83492791) & (tabsize - 1))

  int nentry = 0;
  for (int i=0; i < ne; i++) {
    const mjtNum* bb = aabb + 6*i;
    int lo[3], hi[3];
    for (int k=0; k < 3; k++) {
      lo[k] = (int)floor(bb[k]/cell);
      hi[k] = (int)floor(bb[3 + k]/cell);
    }
    for (int ix=lo[0]; ix <= hi[0]; ix++) {
      for (int iy=lo[1]; iy <= hi[1]; iy++) {
        for (int iz=lo[2]; iz <= hi[2]; iz++) {
          unsigned int h = IPC_CELLHASH(ix, iy, iz);
          entry[nentry] = i;
          next[nentry] = head[h];
          head[h] = nentry;
          nentry++;
        }
      }
    }
  }

  // gather pairs: for each element walk its cells, mark-dedup neighbors
  mjtByte* seen = mjSTACKALLOC(d, ne, mjtByte);
  int* found = mjSTACKALLOC(d, ne, int);
  for (int i=0; i < ne; i++) {
    seen[i] = 0;
  }

  int npair = 0;
  for (int i=0; i < ne; i++) {
    const mjtNum* bb = aabb + 6*i;
    int lo[3], hi[3];
    for (int k=0; k < 3; k++) {
      lo[k] = (int)floor(bb[k]/cell);
      hi[k] = (int)floor(bb[3 + k]/cell);
    }
    int nfound = 0;
    for (int ix=lo[0]; ix <= hi[0]; ix++) {
      for (int iy=lo[1]; iy <= hi[1]; iy++) {
        for (int iz=lo[2]; iz <= hi[2]; iz++) {
          for (int it=head[IPC_CELLHASH(ix, iy, iz)]; it >= 0; it = next[it]) {
            int j = entry[it];
            if (j <= i || seen[j]) {
              continue;
            }
            seen[j] = 1;
            found[nfound++] = j;

            // flex-level eligibility: one comparison that skips the AABB work (e.g.
            // same-flex neighborhoods of a flex with self-collision disabled)
            if (!mj_ipcPairEligible(m, eflex[i], eflex[j])) {
              continue;
            }

            // AABB overlap test
            const mjtNum* bj = aabb + 6*j;
            if (bb[3] < bj[0] || bj[3] < bb[0] || bb[4] < bj[1] || bj[4] < bb[1] ||
                bb[5] < bj[2] || bj[5] < bb[2]) {
              continue;
            }

            // element-level adjacency filter
            if (shareVertex(m, eflex[i], eid[i], eflex[j], eid[j])) {
              continue;
            }


            if (npair < npairmax) {
              pairs[npair].f1 = eflex[i];
              pairs[npair].e1 = eid[i];
              pairs[npair].f2 = eflex[j];
              pairs[npair].e2 = eid[j];
            }
            npair++;
          }
        }
      }
    }
    for (int k=0; k < nfound; k++) {
      seen[found[k]] = 0;
    }
  }

  #undef IPC_CELLHASH

  mj_freeStack(d);
  return npair;
}


//---------------------------------- feasibility --------------------------------------------------

// pack element vertex positions
static void packElem(const mjModel* m, const mjtNum* pos, int f, int e, mjtNum* vert) {
  int vid[3];
  int nv = elemVerts(m, f, e, vid);
  for (int i=0; i < nv; i++) {
    mju_copy3(vert + 3*i, pos + 3*vid[i]);
  }
}





// check flex-flex feasibility, see header
int mj_ipcFeasible(const mjModel* m, mjData* d, mjIPCPair* badpair) {
  mj_markStack(d);

  // generous capacity: feasibility uses zero margin, so overlaps are near-touching pairs
  int npairmax = 64*m->nflexelem + 1024;
  mjIPCPair* pairs = mjSTACKALLOC(d, npairmax, mjIPCPair);
  int npair = mj_ipcBroadphase(m, d, NULL, NULL, 0, pairs, npairmax);
  if (npair > npairmax) {
    npair = npairmax;
  }

  int feasible = 1;
  for (int i=0; i < npair; i++) {
    mjtNum vert1[9], vert2[9];
    int f1 = pairs[i].f1, f2 = pairs[i].f2;
    packElem(m, d->flexvert_xpos, f1, pairs[i].e1, vert1);
    packElem(m, d->flexvert_xpos, f2, pairs[i].e2, vert2);
    mjtNum dist = mj_ipcElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
                                     NULL, NULL);
    if (dist <= m->flex_radius[f1] + m->flex_radius[f2]) {
      feasible = 0;
      if (badpair) {
        *badpair = pairs[i];
      }
      break;
    }
  }

  mj_freeStack(d);
  return feasible;
}


// distance below which the legacy soft-contact fallback owns a flex-flex pair: the
// extended barrier's force plateaus there, see header
mjtNum mj_ipcLegacyDistance(const mjModel* m, int f1, int f2) {
  return m->flex_radius[f1] + m->flex_radius[f2];
}


//---------------------------------- implicit elastic model ---------------------------------------

// quadratic model of the implicit elastic solve, mirroring flexInterp_cgsolve exactly:
//   A = (M - h*qDeriv) - (h^2 + h*damping)*K_interp + (h^2 + h*damping)*K_bend
//   rhs = (M - h*qDeriv)*qacc + h*K_interp*qvel_pre - h*K_bend*qvel_pre
// where qacc is the uncorrected acceleration (the engine CG is gated off under the flag)
// and qvel_pre = qvel - h*qacc; the minimizer of E = 0.5*D'A*D - h^2*rhs'D over
// D = x - x_prev - h*qvel_pre reproduces the CG-corrected step
typedef struct {
  int mode;             // 0: inertia only (explicit integrators), 1: quadratic elastic model
  mjData* d;
  mjtNum h;
  mjtNum* scat;         // nv scratch: scatter packed vectors
  mjtNum* Av;           // nv scratch: matvec result
  mjtNum* tmp;          // nv scratch
  mjtNum* rhs;          // nv: precomputed right-hand side
  mjtNum* Krot;         // interp stiffness cache
} mjIPCElastic;

// any flex needs implicit stiffness treatment (mirror of the engine-internal check)
static int hasImplicitStiffness(const mjModel* m) {
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_rigid[f]) {
      continue;
    }
    if (m->flex_interp[f] && m->flex_edgeequality[f] != 3 &&
        m->flex_stiffness[m->flex_stiffnessadr[f]] != 0) {
      return 1;
    }
    if (!m->flex_interp[f] && m->flex_dim[f] == 2 && m->flex_bendingadr[f] >= 0) {
      return 1;
    }
  }
  return 0;
}

// res_nv = A * vec_nv (see model above)
static void elasticAmul(const mjModel* m, mjIPCElastic* el, mjtNum* res, const mjtNum* vec) {
  int nv = m->nv;
  mjtNum h = el->h;
  mju_mulMatVecSparse(res, el->d->qDeriv, vec, nv, m->D_rownnz, m->D_rowadr,
                      m->D_colind, NULL);
  mju_mulSymVecSparse(el->tmp, el->d->M, vec, nv, m->M_rownnz, m->M_rowadr, m->M_colind);
  mju_addScl(res, el->tmp, res, -h, nv);
  mjd_flexInterp_mul(m, el->d, res, vec, -(h*h), -h, el->Krot);
  mjd_flexBend_mul(m, el->d, res, vec, h*h, h);
}

// scatter packed (3 per moving vertex) into an nv vector, zeroing the rest
static void scatterPacked(mjtNum* nvvec, int nv, const mjtNum* packed, int nmv,
                          const int* dofadr) {
  mju_zero(nvvec, nv);
  for (int k=0; k < 3*nmv; k++) {
    nvvec[dofadr[k]] = packed[k];
  }
}

// smooth (non-barrier) energy and optionally its packed gradient
static mjtNum smoothEnergy(const mjModel* m, mjIPCElastic* el, const mjtNum* x,
                           const mjtNum* xref, const mjtNum* xhat, const mjtNum* mass,
                           int nmv, const int* dofadr, mjtNum* grad) {
  // explicit mode: E = 0.5*sum m*|x - xhat|^2
  if (!el->mode) {
    mjtNum E = 0;
    for (int k=0; k < nmv; k++) {
      mjtNum diff[3];
      mju_sub3(diff, x + 3*k, xhat + 3*k);
      E += 0.5*mass[k]*mju_dot3(diff, diff);
      if (grad) {
        mju_scl3(grad + 3*k, diff, mass[k]);
      }
    }
    return E;
  }

  // elastic mode: E = 0.5*D'A*D - h^2*rhs'D with D = x - xref (xref = xprev + h*qvel_pre)
  int nv = m->nv;
  mjtNum h2 = el->h*el->h;
  for (int k=0; k < 3*nmv; k++) {
    el->tmp[k] = x[k] - xref[k];  // packed D, reusing tmp head
  }
  scatterPacked(el->scat, nv, el->tmp, nmv, dofadr);
  elasticAmul(m, el, el->Av, el->scat);
  mjtNum E = 0;
  for (int k=0; k < 3*nmv; k++) {
    mjtNum AD = el->Av[dofadr[k]];
    mjtNum D = x[k] - xref[k];
    E += 0.5*D*AD - h2*el->rhs[dofadr[k]]*D;
    if (grad) {
      grad[k] = AD - h2*el->rhs[dofadr[k]];
    }
  }
  return E;
}


//---------------------------------- barrier projection solve -------------------------------------

// solver constants
#define mjIPC_NEWTONITER 20     // maximum Newton iterations
#define mjIPC_CGITER     100    // maximum inner PCG iterations
#define mjIPC_LSITER     20     // maximum line-search backtracks
#define mjIPC_DMARGIN    0.1    // CCD filter preserves this fraction of the current gap

// per-active-pair cached barrier data for Hessian-vector products
typedef struct {
  int nvert;            // total vertices (both sides)
  int slot[6];          // moving-vertex slot per vertex, -1 if fixed
  mjtNum graddist[18];  // distance gradient wrt vertices
  mjtNum coef;          // h^2 * kappa * ddb (Gauss-Newton weight)
} mjIPCActive;

// gather pair vertices: global ids, count
static int pairVerts(const mjModel* m, const mjIPCPair* p, int gid[6]) {
  int v1[3], v2[3];
  int n1 = elemVerts(m, p->f1, p->e1, v1);
  int n2 = elemVerts(m, p->f2, p->e2, v2);
  for (int i=0; i < n1; i++) {
    gid[i] = v1[i];
  }
  for (int i=0; i < n2; i++) {
    gid[n1 + i] = v2[i];
  }
  return n1 + n2;
}

// conservative max step fraction for one pair along displacement dx (per moving vertex slot)
static mjtNum caFilterPair(const mjModel* m, const mjIPCPair* p, const mjtNum* vertpos,
                           const mjtNum* dx, const int* vertslot) {
  int gid[6];
  int nv = pairVerts(m, p, gid);
  int dim1 = m->flex_dim[p->f1], dim2 = m->flex_dim[p->f2];
  int nv1 = dim1 + 1;
  mjtNum xi = m->flex_radius[p->f1] + m->flex_radius[p->f2];

  // per-vertex displacements over the full step; the closing-speed bound is the maximum
  // pairwise RELATIVE displacement (witness points are convex combinations of vertices,
  // so their relative velocity is bounded by the largest vertex-pair difference); the
  // absolute-sum bound would stall co-moving pairs, the common case in coherent cloth
  mjtNum disp[18];
  for (int i=0; i < nv; i++) {
    int slot = vertslot[gid[i]];
    if (slot >= 0) {
      mju_copy3(disp + 3*i, dx + 3*slot);
    } else {
      mju_zero3(disp + 3*i);
    }
  }
  mjtNum vmax = 0;
  for (int i=0; i < nv1; i++) {
    for (int j=nv1; j < nv; j++) {
      mjtNum rel[3];
      mju_sub3(rel, disp + 3*i, disp + 3*j);
      mjtNum mag = mju_norm3(rel);
      vmax = mag > vmax ? mag : vmax;
    }
  }
  if (vmax < mjMINVAL) {
    return 1;
  }

  // conservative advancement on per-vertex linear motion; the filter permits consuming
  // a fraction of the current gap, so feasible pairs never return zero (a stuck filter
  // would freeze the solve)
  mjtNum vert[18];
  mjtNum keep = 0;
  mjtNum floor = xi;
  mjtNum t = 0;
  for (int iter=0; iter < 64; iter++) {
    for (int i=0; i < nv; i++) {
      const mjtNum* base = vertpos + 3*gid[i];
      mju_addScl3(vert + 3*i, base, disp + 3*i, t);
    }
    mjtNum dist = mj_ipcElemDistance(vert, dim1, vert + 3*nv1, dim2, NULL, NULL);
    if (iter == 0) {
      if (dist <= xi) {
        // already past the surface: the surface invariant is lost, but the crossing
        // invariant must survive any load and step size, so these pairs keep a
        // fraction of their remaining centerline distance instead of abstaining
        // (at large timesteps the old abstention let loaded pairs cross a face in
        // a single step, faster than the soft-contact fallback could push back)
        if (dist < 0.01*xi) {
          return 1;  // effectively pierced: nothing left to protect reliably
        }
        // absolute floor: a proportional keep decays geometrically under sustained
        // load until it drowns in floating-point noise and the pair grinds through;
        // parking hard-pressed pairs at a small fixed fraction of the thickness is
        // the invariant working, not a failure
        floor = 0;
        keep = mjIPC_DMARGIN*dist > 0.05*xi ? mjIPC_DMARGIN*dist : 0.05*xi;
      } else {
        keep = mjIPC_DMARGIN*(dist - xi);
      }
    }
    mjtNum gap = dist - floor - keep;
    if (gap <= 0) {
      return t;
    }
    t += gap/vmax;
    if (t >= 1) {
      return 1;
    }
  }
  return t;
}

// barrier energy over a shortlist of pairs (callers precheck with aabbGap, so no
// per-pair rejection is needed here); deeply penetrating pairs contribute their
// constant plateau energy, keeping line-search comparisons consistent
static mjtNum shortlistEnergy(const mjModel* m, const mjIPCPair* pairs, int npair,
                              const mjtNum* vertpos, mjtNum h2kappa) {
  mjtNum E = 0;
  for (int i=0; i < npair; i++) {
    mjtNum vert1[9], vert2[9];
    int f1 = pairs[i].f1, f2 = pairs[i].f2;
    packElem(m, vertpos, f1, pairs[i].e1, vert1);
    packElem(m, vertpos, f2, pairs[i].e2, vert2);
    mjtNum xi = m->flex_radius[f1] + m->flex_radius[f2];
    mjtNum dist = mj_ipcElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
                                     NULL, NULL);
    E += h2kappa*extBarrier(dist, xi, 0.1*xi, NULL, NULL);
  }
  return E;
}

// shared context for the threaded per-pair passes of the barrier solve; per-task
// outputs are chunked so workers never write shared memory (reduced after dispatch)
typedef struct {
  // shared inputs
  const mjIPCPair* pairs;
  int npair;
  const mjtNum* vertpos;
  const mjtNum* dx;
  const int* vertslot;
  mjtNum* gapeval;
  mjtNum* accateval;
  mjtNum accmotion;
  mjtNum h2kappa;
  mjtNum mu;
  int nmv;
  int ntask;

  // gradient pass: per-task barrier gradient/diagonal buffers and active chunks
  mjtNum* gradbuf;       // ntask x 3*nmv, zeroed before dispatch
  mjtNum* diagbuf;       // ntask x 3*nmv, zeroed before dispatch
  mjIPCActive* active;   // task t fills [t*activechunk, t*activechunk + nactive_t[t])
  int activechunk;
  int* nactive_t;
  mjtNum* E0_t;

  // filter pass: per-task step fractions and shortlist chunks
  mjIPCPair* shortlist;  // task t fills [t*shortchunk, t*shortchunk + nshort_t[t])
  int shortchunk;
  int* nshort_t;
  mjtNum* alpha_t;

  // line-search pass: per-task energy partial sums over the compacted shortlist
  int nshort;
  mjtNum* E_t;
} mjIPCCtx;

// slice bounds for task `t` of `ntask` over `n` items
static void taskSlice(int n, int ntask, int t, int* lo, int* hi) {
  int span = (n + ntask - 1)/ntask;
  *lo = t*span;
  *hi = (t + 1)*span < n ? (t + 1)*span : n;
}

// gradient pass task: barrier energy, gradient, Jacobi diagonal, active-set chunk
static void ipcGradTask(const mjModel* m, mjData* d, void* arg, int thread_id, int task_id) {
  mjIPCCtx* c = arg;
  int lo, hi;
  taskSlice(c->npair, c->ntask, task_id, &lo, &hi);
  mjtNum* grad = c->gradbuf + task_id*3*c->nmv;
  mjtNum* diag = c->diagbuf + task_id*3*c->nmv;
  mjIPCActive* active = c->active + task_id*c->activechunk;
  int nactive = 0;
  mjtNum E0 = 0;

  for (int i=lo; i < hi; i++) {
    int f1 = c->pairs[i].f1, f2 = c->pairs[i].f2;
    mjtNum xi = m->flex_radius[f1] + m->flex_radius[f2];
    mjtNum dhat = 0.1*xi;

    // budget gate: the gap shrinks at most twice the motion accumulated since the
    // last exact evaluation, so a sufficient remainder means zero energy and force
    if (c->gapeval[i] - 2*(c->accmotion - c->accateval[i]) > dhat) {
      continue;
    }

    mjtNum vert1[9], vert2[9], graddist[18];
    int dim1 = m->flex_dim[f1], dim2 = m->flex_dim[f2];
    packElem(m, c->vertpos, f1, c->pairs[i].e1, vert1);
    packElem(m, c->vertpos, f2, c->pairs[i].e2, vert2);
    mjtNum dist = mj_ipcElemDistance(vert1, dim1, vert2, dim2, NULL, graddist);
    c->gapeval[i] = dist - xi;
    c->accateval[i] = c->accmotion;

    // the energy at the current iterate doubles as the line-search reference E0
    mjtNum db, ddb;
    mjtNum b = extBarrier(dist, xi, dhat, &db, &ddb);
    E0 += c->h2kappa*b;

    // no force: outside the band, or so deeply penetrating that the extended barrier
    // has plateaued (the legacy soft-contact fallback owns those)
    if (b == 0 || db == 0) {
      continue;
    }

    if (nactive >= c->activechunk) {
      fprintf(stderr, "IPC SATURATED: nactive=%d activechunk=%d\n", nactive, c->activechunk);
      continue;
    }

    // accumulate gradient and Jacobi diagonal; cache for Hessian products
    mjIPCActive* a = active + nactive++;
    int gid[6];
    a->nvert = pairVerts(m, c->pairs + i, gid);
    a->coef = c->h2kappa*(ddb > 0 ? ddb : 0);
    for (int vi=0; vi < a->nvert; vi++) {
      a->slot[vi] = c->vertslot[gid[vi]];
      mju_copy3(a->graddist + 3*vi, graddist + 3*vi);
      if (a->slot[vi] >= 0) {
        for (int cc=0; cc < 3; cc++) {
          grad[3*a->slot[vi] + cc] += c->h2kappa*db*graddist[3*vi + cc];
          diag[3*a->slot[vi] + cc] += a->coef*graddist[3*vi + cc]*graddist[3*vi + cc];
        }
      }
    }
  }
  c->nactive_t[task_id] = nactive;
  c->E0_t[task_id] = E0;
}

// CCD filter task: per-pair conservative advancement and shortlist chunk
static void ipcFilterTask(const mjModel* m, mjData* d, void* arg, int thread_id, int task_id) {
  mjIPCCtx* c = arg;
  int lo, hi;
  taskSlice(c->npair, c->ntask, task_id, &lo, &hi);
  mjIPCPair* shortlist = c->shortlist + task_id*c->shortchunk;
  int nshort = 0;
  mjtNum alpha = 1;

  for (int i=lo; i < hi; i++) {
    mjtNum remaining = c->gapeval[i] - 2*(c->accmotion - c->accateval[i]);
    mjtNum dhat = 0.1*(m->flex_radius[c->pairs[i].f1] + m->flex_radius[c->pairs[i].f2]);
    if (remaining > dhat + 2*c->mu) {
      continue;
    }

    // refine with the pair's own motion bound: the global bound is set by the single
    // fastest vertex anywhere, which would open the gate for the whole contact region
    int gid[6];
    int nvp = pairVerts(m, c->pairs + i, gid);
    mjtNum mup2 = 0;
    for (int v=0; v < nvp; v++) {
      int slot = c->vertslot[gid[v]];
      if (slot >= 0) {
        mjtNum n2 = mju_dot3(c->dx + 3*slot, c->dx + 3*slot);
        mup2 = n2 > mup2 ? n2 : mup2;
      }
    }
    mjtNum mup = mju_sqrt(mup2);
    if (remaining > dhat + 2*mup) {
      continue;
    }
    if (nshort >= c->shortchunk) {
      fprintf(stderr, "IPC SATURATED: nshort=%d shortchunk=%d\n", nshort, c->shortchunk);
      continue;
    }
    shortlist[nshort++] = c->pairs[i];
    if (remaining <= 2*mup) {
      mjtNum ai = caFilterPair(m, c->pairs + i, c->vertpos, c->dx, c->vertslot);
      alpha = ai < alpha ? ai : alpha;
    }
  }
  c->nshort_t[task_id] = nshort;
  c->alpha_t[task_id] = alpha;
}

// line-search energy task: barrier sum over a slice of the compacted shortlist
static void ipcEnergyTask(const mjModel* m, mjData* d, void* arg, int thread_id, int task_id) {
  mjIPCCtx* c = arg;
  int lo, hi;
  taskSlice(c->nshort, c->ntask, task_id, &lo, &hi);
  c->E_t[task_id] = shortlistEnergy(m, c->shortlist + lo, hi - lo, c->vertpos, c->h2kappa);
}


// barrier projection of the integrated flex state, see header
void mj_flexIPC(const mjModel* m, mjData* d) {

  int nflexvert = m->nflexvert;
  mjtNum h = m->opt.timestep;

  // implicit elastic stage: active under implicit integrators with flex stiffness and no
  // sleep filtering (the same gating the deleted post-hoc CG used); independent of the
  // barrier stage, which requires solvable vertex DOFs
  int elastic = (m->opt.integrator == mjINT_IMPLICIT ||
                 m->opt.integrator == mjINT_IMPLICITFAST) &&
                !(mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv) && hasImplicitStiffness(m);

  // quick exit: nothing to do
  int any = 0;
  for (int f=0; f < m->nflex; f++) {
    if (ipcSolvable(m, f)) {
      any = 1;
      break;
    }
  }
  if (!any && !elastic) {
    return;
  }

  mj_markStack(d);

  // implicit elastic model context
  mjIPCElastic el;
  el.mode = elastic;
  el.d = d;
  el.h = h;
  el.scat = el.Av = el.tmp = el.rhs = el.Krot = NULL;
  if (el.mode) {
    int nv = m->nv;
    el.scat = mjSTACKALLOC(d, nv, mjtNum);
    el.Av = mjSTACKALLOC(d, nv, mjtNum);
    el.tmp = mjSTACKALLOC(d, nv, mjtNum);
    el.rhs = mjSTACKALLOC(d, nv, mjtNum);
    if (m->nflexstiffness) {
      el.Krot = mjSTACKALLOC(d, m->nflexstiffness, mjtNum);
      mju_zero(el.Krot, m->nflexstiffness);
      mjd_flexInterp_cacheKrot(m, d, el.Krot);
    }

    // pre-integration velocity: the integrator advanced qvel by h*qacc
    mjtNum* vpre = mjSTACKALLOC(d, nv, mjtNum);
    mju_addScl(vpre, d->qvel, d->qacc, -h, nv);

    // rhs = qfrc_smooth + qfrc_constraint + h*K_interp*vpre - h*K_bend*vpre, exactly as
    // the engine CG builds it
    mju_add(el.rhs, d->qfrc_smooth, d->qfrc_constraint, nv);
    mjd_flexInterp_mul(m, d, el.rhs, vpre, h, 0, el.Krot);
    mjd_flexBend_mul(m, d, el.rhs, vpre, -h, 0);

    // stage 1: solve A*qacc_corr = rhs over the full nv (the variational smooth solve,
    // identical system to the deleted post-hoc CG), preconditioned by the engine's own
    // factorization; this covers all flex DOF layouts including interpolated flexes on
    // arbitrary joints
    mjtNum* qcorr = mjSTACKALLOC(d, nv, mjtNum);
    mjtNum* cr = mjSTACKALLOC(d, nv, mjtNum);
    mjtNum* cz = mjSTACKALLOC(d, nv, mjtNum);
    mjtNum* cp = mjSTACKALLOC(d, nv, mjtNum);
    mjtNum* cAp = mjSTACKALLOC(d, nv, mjtNum);
    mju_copy(qcorr, d->qacc, nv);
    elasticAmul(m, &el, cAp, qcorr);
    mju_sub(cr, el.rhs, cAp, nv);
    mjtNum rnorm = mju_dot(cr, cr, nv);
    mjtNum ctol = 1e-10*mju_dot(el.rhs, el.rhs, nv);
    if (rnorm >= ctol && rnorm >= mjMINVAL) {
      // preconditioner: the factored standard system (M - h*qDeriv)
      #define IPC_PRECOND(z_out, r_in)         if (m->opt.integrator == mjINT_IMPLICIT) {           mju_solveLUSparse(z_out, d->qLU, r_in, nv, m->D_rownnz, m->D_rowadr,                             m->D_diag, m->D_colind, NULL);         } else {           mju_copy(z_out, r_in, nv);           mj_solveLD(z_out, d->qH, d->qHDiagInv, nv, 1, m->M_rownnz, m->M_rowadr,                      m->M_colind, NULL);         }
      IPC_PRECOND(cz, cr);
      mju_copy(cp, cz, nv);
      mjtNum crz = mju_dot(cr, cz, nv);
      for (int cg=0; cg < 50; cg++) {
        elasticAmul(m, &el, cAp, cp);
        mjtNum pAp = mju_dot(cp, cAp, nv);
        if (mju_abs(pAp) < mjMINVAL) {
          break;
        }
        mjtNum calpha = crz/pAp;
        mju_addToScl(qcorr, cp, calpha, nv);
        mju_addToScl(cr, cAp, -calpha, nv);
        rnorm = mju_dot(cr, cr, nv);
        if (rnorm < ctol || rnorm < mjMINVAL) {
          break;
        }
        IPC_PRECOND(cz, cr);
        mjtNum crznew = mju_dot(cr, cz, nv);
        mju_addScl(cp, cz, cp, crznew/(crz > mjMINVAL ? crz : mjMINVAL), nv);
        crz = crznew;
      }
      #undef IPC_PRECOND
    }

    // apply the correction exactly: undo the position advance, correct velocity and
    // acceleration, redo the advance (valid for quaternion DOFs via mj_integratePos)
    mj_integratePos(m, d->qpos, d->qvel, -h);
    for (int i=0; i < nv; i++) {
      mjtNum dv = h*(qcorr[i] - d->qacc[i]);
      d->qvel[i] += dv;
      d->qacc[i] = qcorr[i];
    }
    mj_integratePos(m, d->qpos, d->qvel, h);

    // stage 2 (barrier projection) uses the inertia metric; the elastic correction is
    // now part of the integrated target gathered below
    el.mode = 0;
  }


  // barrier stage requires solvable vertices and is gated by the flexipc flag (the
  // elastic stage above always runs: it replaces the former post-hoc CG)
  if (!any || mjDISABLED(mjDSBL_FLEXIPC)) {
    mj_freeStack(d);
    return;
  }

  // identify moving vertices: bodies with exactly 3 axis-aligned slide joints; all other
  // vertices (pinned or unsupported layouts) are treated as fixed at their pre-step position
  int* vertslot = mjSTACKALLOC(d, nflexvert, int);
  int* slotgv = mjSTACKALLOC(d, nflexvert, int);
  int* qposadr = mjSTACKALLOC(d, 3*nflexvert, int);
  int* dofadr = mjSTACKALLOC(d, 3*nflexvert, int);
  int nmv = 0;
  for (int gv=0; gv < nflexvert; gv++) {
    vertslot[gv] = -1;
  }
  for (int f=0; f < m->nflex; f++) {
    if (!ipcSolvable(m, f)) {
      continue;
    }
    for (int v=0; v < m->flex_vertnum[f]; v++) {
      int gv = m->flex_vertadr[f] + v;
      int bid = m->flex_vertbodyid[gv];
      if (bid <= 0 || m->body_jntnum[bid] != 3) {
        continue;
      }
      int qadr[3] = {-1, -1, -1}, dadr[3] = {-1, -1, -1};
      int ok = 1;
      for (int j=0; j < 3; j++) {
        int jid = m->body_jntadr[bid] + j;
        if (m->jnt_type[jid] != mjJNT_SLIDE) {
          ok = 0;
          break;
        }
        const mjtNum* ax = m->jnt_axis + 3*jid;
        int comp = ax[0] > 0.5 ? 0 : (ax[1] > 0.5 ? 1 : (ax[2] > 0.5 ? 2 : -1));
        if (comp < 0 || qadr[comp] >= 0) {
          ok = 0;
          break;
        }
        qadr[comp] = m->jnt_qposadr[jid];
        dadr[comp] = m->jnt_dofadr[jid];
      }
      if (!ok) {
        continue;
      }
      vertslot[gv] = nmv;
      slotgv[nmv] = gv;
      for (int k=0; k < 3; k++) {
        qposadr[3*nmv + k] = qadr[k];
        dofadr[3*nmv + k] = dadr[k];
      }
      nmv++;
    }
  }
  if (!nmv) {
    mj_freeStack(d);
    return;
  }

  // states: x_prev from pre-step kinematics, xhat from the integrated qpos
  mjtNum* x = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* xprev = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* xhat = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* mass = mjSTACKALLOC(d, nmv, mjtNum);
  mjtNum masssum = 0, ximin = mjMAXVAL;
  for (int k=0; k < nmv; k++) {
    int gv = slotgv[k];
    mju_copy3(xprev + 3*k, d->flexvert_xpos + 3*gv);
    // slide qpos is a displacement from the body reference, not a world position: build
    // the world target from the pre-step position plus the integrated step h*qvel, which
    // is exactly how Euler and the implicit integrators advance qpos
    for (int c=0; c < 3; c++) {
      xhat[3*k + c] = xprev[3*k + c] + h*d->qvel[dofadr[3*k + c]];
    }
    mass[k] = m->body_mass[m->flex_vertbodyid[gv]];
    masssum += mass[k];
  }
  mju_copy(x, xprev, 3*nmv);
  for (int f=0; f < m->nflex; f++) {
    if (ipcEligible(m, f) && 2*m->flex_radius[f] < ximin) {
      ximin = 2*m->flex_radius[f];
    }
  }

  // gather the stretch deviation metric: edges of flexes with an active edge-equality
  // constraint, measured against their length at xhat
  int nedgemax = 0;
  for (int f=0; f < m->nflex; f++) {
    nedgemax += m->flex_edgenum[f];
  }
  mjIPCEdge* edges = mjSTACKALLOC(d, nedgemax > 0 ? nedgemax : 1, mjIPCEdge);
  int nedge = 0;
  for (int i=0; mjIPC_STRETCHK > 0 && i < m->neq; i++) {
    if (m->eq_type[i] != mjEQ_FLEX || !d->eq_active[i]) {
      continue;
    }
    int f = m->eq_obj1id[i];
    if (!ipcSolvable(m, f)) {
      continue;
    }
    for (int e=0; e < m->flex_edgenum[f]; e++) {
      const int* ev = m->flex_edge + 2*(m->flex_edgeadr[f] + e);
      int gv1 = m->flex_vertadr[f] + ev[0];
      int gv2 = m->flex_vertadr[f] + ev[1];
      int s1 = vertslot[gv1], s2 = vertslot[gv2];
      if (s1 < 0 && s2 < 0) {
        continue;
      }
      if (s1 < 0) {  // keep the moving end first
        int tmp = s1; s1 = s2; s2 = tmp;
        int tg = gv1; gv1 = gv2; gv2 = tg;
      }
      mjIPCEdge* ed = edges + nedge++;
      ed->s1 = s1;
      ed->s2 = s2;
      mjtNum p2[3];
      if (s2 >= 0) {
        mju_copy3(p2, xhat + 3*s2);
        mju_zero3(ed->fixedpos);
      } else {
        mju_copy3(p2, d->flexvert_xpos + 3*gv2);
        mju_copy3(ed->fixedpos, p2);
      }
      // anchor to the rest length, not to the length at xhat: a per-step re-anchored
      // reference is a non-conservative moving-target potential that pumps energy into
      // parked contacts (sustained velocity churn against the equality rows)
      ed->Lhat = m->flexedge_length0[m->flex_edgeadr[f] + e];
      mjtNum m2 = s2 >= 0 ? mass[s2] : mass[s1];
      ed->k = mjIPC_STRETCHK*0.5*(mass[s1] + m2);
    }
  }

  // barrier scale: heuristic stiffness (adaptive kappa is a follow-up); the activation
  // band dhat is a small fraction of the combined thickness: the resting gap under load
  // settles inside the band, so a wide band reads as objects hovering at a distance
  mjtNum kappa = 2.0*(masssum/nmv)/(h*ximin);
  mjtNum h2kappa = h*h*kappa;

  // full vertex position buffers for distance evaluation (fixed verts stay at pre-step)
  mjtNum* vertpos = mjSTACKALLOC(d, 3*nflexvert, mjtNum);
  mjtNum* vertswept = mjSTACKALLOC(d, 3*nflexvert, mjtNum);
  mju_copy(vertpos, d->flexvert_xpos, 3*nflexvert);
  mju_copy(vertswept, d->flexvert_xpos, 3*nflexvert);

  // work vectors
  mjtNum* grad = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* dx = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* r = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* z = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* cgp = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* Ap = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* diag = mjSTACKALLOC(d, 3*nmv, mjtNum);
  int npairmax = 64*m->nflexelem + 1024;
  mjIPCPair* pairs = mjSTACKALLOC(d, npairmax, mjIPCPair);
  mjIPCPair* shortlist = mjSTACKALLOC(d, npairmax, mjIPCPair);

  // the active set holds only pairs inside the activation band, a small fraction of
  // the broadphase candidates (the band is a sliver of the broadphase margin)
  int nactivemax = 16*m->nflexelem + 256;
  mjIPCActive* active = mjSTACKALLOC(d, nactivemax, mjIPCActive);

  // distance budgets: gapeval[i] is the exact surface gap of pair i at its last
  // evaluation and accateval[i] the motion accumulator at that moment; the pair's gap
  // can have shrunk by at most twice the motion accumulated since, so most pairs are
  // skipped with one subtraction instead of a distance kernel
  mjtNum* gapeval = mjSTACKALLOC(d, npairmax, mjtNum);
  mjtNum* accateval = mjSTACKALLOC(d, npairmax, mjtNum);
  mjtNum* xbp = mjSTACKALLOC(d, 3*nmv, mjtNum);

  // per-task buffers for the threaded passes (workers cannot use the mjData stack)
  int ntask = mju_numThread(d);
  mjIPCCtx ctx;
  ctx.vertslot = vertslot;
  ctx.gapeval = gapeval;
  ctx.accateval = accateval;
  ctx.h2kappa = h2kappa;
  ctx.nmv = nmv;
  ctx.ntask = ntask;
  ctx.gradbuf = mjSTACKALLOC(d, ntask*3*nmv, mjtNum);
  ctx.diagbuf = mjSTACKALLOC(d, ntask*3*nmv, mjtNum);
  ctx.activechunk = nactivemax/ntask;
  ctx.shortchunk = npairmax/ntask;
  ctx.shortlist = shortlist;
  ctx.nactive_t = mjSTACKALLOC(d, ntask, int);
  ctx.nshort_t = mjSTACKALLOC(d, ntask, int);
  ctx.E0_t = mjSTACKALLOC(d, ntask, mjtNum);
  ctx.alpha_t = mjSTACKALLOC(d, ntask, mjtNum);
  ctx.E_t = mjSTACKALLOC(d, ntask, mjtNum);

  // Newton loop
#ifdef mjIPC_DEBUG
  int debug_newton = 0, debug_active = 0, debug_filter = 0, debug_nshort = 0, debug_nca = 0;
#endif
  // broadphase margin: the activation band of the largest flex
  mjtNum maxdhat = 0;
  for (int f=0; f < m->nflex; f++) {
    if (ipcEligible(m, f) && 0.2*m->flex_radius[f] > maxdhat) {
      maxdhat = 0.2*m->flex_radius[f];
    }
  }

  // one envelope broadphase covers the whole solve: boxes span (x, xhat) plus a slack
  // margin; iterates deviating beyond half the slack trigger a re-run (rare), so pairs
  // outside the candidate set provably stay separated through every iterate
  mjtNum slack = 4*maxdhat;
  mjtNum accmotion = 0;
  int need_bp = 1;
  int npair = 0;
  for (int newton=0; newton < mjIPC_NEWTONITER; newton++) {
#ifdef mjIPC_DEBUG
    debug_newton++;
#endif
    // refresh vertex buffer with current iterate
    for (int k=0; k < nmv; k++) {
      mju_copy3(vertpos + 3*slotgv[k], x + 3*k);
    }

    if (need_bp) {
      mju_copy(xbp, x, 3*nmv);
      for (int k=0; k < nmv; k++) {
        mju_copy3(vertswept + 3*slotgv[k], xhat + 3*k);
      }
      npair = mj_ipcBroadphase(m, d, vertpos, vertswept, maxdhat + slack, pairs, npairmax);
      if (npair > npairmax) {
        fprintf(stderr, "IPC SATURATED: npair=%d npairmax=%d\n", npair, npairmax);
        npair = npairmax;
      }
      for (int i=0; i < npair; i++) {
        gapeval[i] = -mjMAXVAL;  // unknown: evaluated on first use
        accateval[i] = accmotion;
      }
      need_bp = 0;
    }

    // assemble smooth gradient (inertia or elastic quadratic) and cache barrier data
    smoothEnergy(m, &el, x, xhat, xhat, mass, nmv, dofadr, grad);
    for (int k=0; k < nmv; k++) {
      for (int c=0; c < 3; c++) {
        diag[3*k + c] = mass[k];
      }
    }
    // stretch deviation metric: gradient, Jacobi diagonal, and the E0 contribution
    mjtNum E0stretch = 0;
    for (int e=0; e < nedge; e++) {
      mjtNum dvec[3];
      const mjtNum* p2 = edges[e].s2 >= 0 ? x + 3*edges[e].s2 : edges[e].fixedpos;
      mju_sub3(dvec, x + 3*edges[e].s1, p2);
      mjtNum L = mju_norm3(dvec);
      if (L < mjMINVAL) {
        continue;
      }
      mjtNum c = L - edges[e].Lhat;
      E0stretch += 0.5*edges[e].k*c*c;
      mjtNum u[3];
      mju_scl3(u, dvec, 1/L);
      for (int cc=0; cc < 3; cc++) {
        mjtNum g = edges[e].k*c*u[cc];
        mjtNum dd = edges[e].k*u[cc]*u[cc];
        grad[3*edges[e].s1 + cc] += g;
        diag[3*edges[e].s1 + cc] += dd;
        if (edges[e].s2 >= 0) {
          grad[3*edges[e].s2 + cc] -= g;
          diag[3*edges[e].s2 + cc] += dd;
        }
      }
    }

    // gradient pass: threaded over candidate pairs, per-task chunks reduced here
    mju_zero(ctx.gradbuf, ntask*3*nmv);
    mju_zero(ctx.diagbuf, ntask*3*nmv);
    ctx.pairs = pairs;
    ctx.npair = npair;
    ctx.vertpos = vertpos;
    ctx.accmotion = accmotion;
    ctx.active = active;
    mju_dispatch(m, d, ipcGradTask, &ctx, ntask);

    mjtNum E0barrier = 0;
    int nactive = 0;
    for (int t=0; t < ntask; t++) {
      E0barrier += ctx.E0_t[t];
      for (int k=0; k < 3*nmv; k++) {
        grad[k] += ctx.gradbuf[t*3*nmv + k];
        diag[k] += ctx.diagbuf[t*3*nmv + k];
      }
      // compact the active chunks in task order
      if (t > 0 && ctx.nactive_t[t] > 0) {
        memmove(active + nactive, active + t*ctx.activechunk,
                ctx.nactive_t[t]*sizeof(mjIPCActive));
      }
      nactive += ctx.nactive_t[t];
    }

    // PCG solve: (M + sum coef*graddist*graddist') dx = -grad, Jacobi preconditioner
    mju_zero(dx, 3*nmv);
    mju_scl(r, grad, -1, 3*nmv);
    for (int k=0; k < 3*nmv; k++) {
      z[k] = r[k]/diag[k];
    }
    mju_copy(cgp, z, 3*nmv);
    mjtNum rz = mju_dot(r, z, 3*nmv);
    mjtNum rz0 = rz;
    for (int cg=0; cg < mjIPC_CGITER && rz > 1e-12*rz0; cg++) {
      // Ap = (smooth Hessian)*p + sum coef*(graddist'p)*graddist
      if (el.mode) {
        scatterPacked(el.scat, m->nv, cgp, nmv, dofadr);
        elasticAmul(m, &el, el.Av, el.scat);
        for (int k=0; k < 3*nmv; k++) {
          Ap[k] = el.Av[dofadr[k]];
        }
      } else {
        for (int k=0; k < nmv; k++) {
          for (int c=0; c < 3; c++) {
            Ap[3*k + c] = mass[k]*cgp[3*k + c];
          }
        }
      }
      for (int i=0; i < nactive; i++) {
        const mjIPCActive* a = active + i;
        mjtNum gp = 0;
        for (int vi=0; vi < a->nvert; vi++) {
          if (a->slot[vi] >= 0) {
            gp += mju_dot3(a->graddist + 3*vi, cgp + 3*a->slot[vi]);
          }
        }
        gp *= a->coef;
        for (int vi=0; vi < a->nvert; vi++) {
          if (a->slot[vi] >= 0) {
            mju_addToScl3(Ap + 3*a->slot[vi], a->graddist + 3*vi, gp);
          }
        }
      }
      for (int e=0; e < nedge; e++) {
        mjtNum dvec[3];
        const mjtNum* p2 = edges[e].s2 >= 0 ? x + 3*edges[e].s2 : edges[e].fixedpos;
        mju_sub3(dvec, x + 3*edges[e].s1, p2);
        mjtNum L = mju_norm3(dvec);
        if (L < mjMINVAL) {
          continue;
        }
        mjtNum u[3];
        mju_scl3(u, dvec, 1/L);
        mjtNum gp = mju_dot3(u, cgp + 3*edges[e].s1);
        if (edges[e].s2 >= 0) {
          gp -= mju_dot3(u, cgp + 3*edges[e].s2);
        }
        gp *= edges[e].k;
        mju_addToScl3(Ap + 3*edges[e].s1, u, gp);
        if (edges[e].s2 >= 0) {
          mju_addToScl3(Ap + 3*edges[e].s2, u, -gp);
        }
      }
      mjtNum pAp = mju_dot(cgp, Ap, 3*nmv);
      if (pAp < mjMINVAL) {
        break;
      }
      mjtNum alpha = rz/pAp;
      mju_addToScl(dx, cgp, alpha, 3*nmv);
      mju_addToScl(r, Ap, -alpha, 3*nmv);
      for (int k=0; k < 3*nmv; k++) {
        z[k] = r[k]/diag[k];
      }
      mjtNum rznew = mju_dot(r, z, 3*nmv);
      mju_addScl(cgp, z, cgp, rznew/(rz > mjMINVAL ? rz : mjMINVAL), 3*nmv);
      rz = rznew;
    }

    // inertial step scale: motion toward the unconstrained target is never clamped
    mjtNum stepnorm0 = 0;
    for (int k=0; k < nmv; k++) {
      mjtNum dxh[3];
      mju_sub3(dxh, xhat + 3*k, x + 3*k);
      mjtNum n2 = mju_dot3(dxh, dxh);
      stepnorm0 = n2 > stepnorm0 ? n2 : stepnorm0;
    }
    stepnorm0 = mju_sqrt(stepnorm0);

    // converged: negligible step
    mjtNum stepnorm = 0;
    for (int k=0; k < 3*nmv; k++) {
      stepnorm = fabs(dx[k]) > stepnorm ? fabs(dx[k]) : stepnorm;
    }
    if (stepnorm < 1e-10) {
      break;
    }

    // per-iteration motion bound: the largest vertex displacement norm; a pair's gap
    // shrinks by at most twice that during this step (per-axis bounds are NOT valid
    // here: a vertex moves up to sqrt(3) times its largest axis component)
    mjtNum mu = 0;
    for (int k=0; k < nmv; k++) {
      mjtNum n2 = mju_dot3(dx + 3*k, dx + 3*k);
      mu = n2 > mu ? n2 : mu;
    }
    mu = mju_sqrt(mu);

    // trust region: near-contact pairs produce enormous barrier gradients that the
    // Jacobi preconditioner cannot temper, proposing steps of many activation bands
    // that the CCD filter then crushes to nothing; clamp the step instead and let the
    // Newton spend its iterations making accepted progress
    mjtNum mumax = 4*maxdhat > stepnorm0 ? 4*maxdhat : stepnorm0;
    if (mu > mumax) {
      mju_scl(dx, dx, mumax/mu, 3*nmv);
      mu = mumax;
    }

#ifdef mjIPC_DEBUG
    debug_filter = npair > debug_filter ? npair : debug_filter;
    debug_active = nactive > debug_active ? nactive : debug_active;
#endif
    // CCD filter and line-search shortlist from the budgets: pairs that cannot reach
    // the band during this step contribute no energy at any trial point, pairs that
    // cannot touch need no conservative advancement; threaded over candidate pairs
    ctx.dx = dx;
    ctx.mu = mu;
    mju_dispatch(m, d, ipcFilterTask, &ctx, ntask);

    mjtNum alpha = 1;
    int nshort = 0;
    for (int t=0; t < ntask; t++) {
      alpha = ctx.alpha_t[t] < alpha ? ctx.alpha_t[t] : alpha;
      if (t > 0 && ctx.nshort_t[t] > 0) {
        memmove(shortlist + nshort, shortlist + t*ctx.shortchunk,
                ctx.nshort_t[t]*sizeof(mjIPCPair));
      }
      nshort += ctx.nshort_t[t];
    }
    ctx.nshort = nshort;

    // backtracking line search; the barrier part of E0 was accumulated alongside the
    // gradient (same pairs, same positions), the trials sum over the swept shortlist
    mjtNum E0 = smoothEnergy(m, &el, x, xhat, xhat, mass, nmv, dofadr, NULL) +
                E0barrier + E0stretch;

    mjtNum gdx = mju_dot(grad, dx, 3*nmv);
    mjtNum* xtry = z;  // reuse work vector
    int accepted = 0;
    for (int ls=0; ls < mjIPC_LSITER; ls++) {
      mju_addScl(xtry, x, dx, alpha, 3*nmv);
      for (int k=0; k < nmv; k++) {
        mju_copy3(vertpos + 3*slotgv[k], xtry + 3*k);
      }
      mju_dispatch(m, d, ipcEnergyTask, &ctx, ntask);
      mjtNum E = smoothEnergy(m, &el, xtry, xhat, xhat, mass, nmv, dofadr, NULL) +
                 stretchEnergy(edges, nedge, xtry);
      for (int t=0; t < ntask; t++) {
        E += ctx.E_t[t];
      }
      // Armijo decrease with a noise floor relative to the energy itself: both sums
      // are nonnegative, so E0 is the correct rounding scale (an absolute floor would
      // dwarf the energies of settled scenes and stall the solver mid-motion)
      if (E <= E0 + 1e-4*alpha*gdx + 1e-12*fabs(E0) + mjMINVAL) {
        mju_copy(x, xtry, 3*nmv);
        // converged (2) when the decrease is within rounding of the energy
        accepted = (E0 - E > 1e-11*fabs(E0) + mjMINVAL) ? 1 : 2;
        break;
      }
      alpha *= 0.5;
    }
    if (accepted != 1) {
      // converged (no meaningful energy progress) or stuck: restore buffer and stop
      for (int k=0; k < nmv; k++) {
        mju_copy3(vertpos + 3*slotgv[k], x + 3*k);
      }
      break;
    }

    // account the accepted motion against the budgets, and re-run the broadphase if
    // any vertex left the candidate envelope by more than half the slack
    accmotion += alpha*mu;
    mjtNum dev = 0;
    for (int k=0; k < 3*nmv; k++) {
      mjtNum lo = xbp[k] < xhat[k] ? xbp[k] : xhat[k];
      mjtNum hi = xbp[k] > xhat[k] ? xbp[k] : xhat[k];
      mjtNum dk = x[k] < lo ? lo - x[k] : (x[k] > hi ? x[k] - hi : 0);
      dev = dk > dev ? dk : dev;
    }
    if (dev > slack/2) {
      need_bp = 1;
    }
  }

#ifdef mjIPC_DEBUG
  fprintf(stderr, "ipc: newton=%d active=%d filter=%d nshort=%d nca=%d\n",
          debug_newton, debug_active, debug_filter, debug_nshort, debug_nca);
#endif

  // write back: shift qpos by the correction (qpos is a displacement, not a position)
  // and set the consistent velocity
  for (int k=0; k < nmv; k++) {
    for (int c=0; c < 3; c++) {
      d->qpos[qposadr[3*k + c]] += x[3*k + c] - xhat[3*k + c];
      d->qvel[dofadr[3*k + c]] = (x[3*k + c] - xprev[3*k + c])/h;
    }
  }

  mj_freeStack(d);
}
