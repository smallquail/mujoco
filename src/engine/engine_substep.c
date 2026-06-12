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

#include "engine/engine_substep.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
} mjSubFeature;

// evaluate a candidate given witness barycentrics, keep it if closer
static void considerFeature(mjSubFeature* best, const mjtNum* vert1, int nv1,
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
static void candSegSeg(mjSubFeature* best, const mjtNum* vert1, int i0, int i1,
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
static void candVertTri(mjSubFeature* best, const mjtNum* vert1, int nv1, const mjtNum* vert2,
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
mjtNum mj_substepElemDistance(const mjtNum* vert1, int dim1, const mjtNum* vert2, int dim2,
                          mjtNum witness[6], mjtNum* grad) {
  int nv1 = dim1 + 1, nv2 = dim2 + 1;
  static const int edges[3][2] = {{0, 1}, {1, 2}, {2, 0}};
  mjSubFeature best;
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


//---------------------------------- broadphase ---------------------------------------------------

// flex participates in substep collision (as moving or fixed geometry): segment/triangle
// elements; rigid and interpolated flexes collide as prescribed obstacles
static int substepEligible(const mjModel* m, int f) {
  return m->flex_dim[f] == 1 || m->flex_dim[f] == 2;
}

// flex vertices are solved as DOFs: direct vertex representation only (flex_interp == 0,
// the standard cloth case); rigid and interpolated flexes are never solved
static int substepSolvable(const mjModel* m, int f) {
  return substepEligible(m, f) && !m->flex_interp[f] && !m->flex_rigid[f];
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

// pair of flexes is owned by the substep solver, see header
int mj_substepPairEligible(const mjModel* m, int f1, int f2) {
  if (!substepEligible(m, f1) || !substepEligible(m, f2)) {
    return 0;
  }
  if (f1 == f2) {
    return m->flex_selfcollide[f1] != mjFLEXSELF_NONE;
  }
  return (m->flex_contype[f1] & m->flex_conaffinity[f2]) ||
         (m->flex_contype[f2] & m->flex_conaffinity[f1]);
}

// spatial-hash broadphase over flex elements, see header
int mj_substepBroadphase(const mjModel* m, mjData* d, const mjtNum* vertpos,
                     const mjtNum* vertpos2, mjtNum margin, mjSubstepPair* pairs, int npairmax) {
  int nflex = m->nflex;
  const mjtNum* pos = vertpos ? vertpos : d->flexvert_xpos;

  // count eligible elements
  int nelem = 0;
  for (int f=0; f < nflex; f++) {
    if (substepEligible(m, f)) {
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
    if (!substepEligible(m, f)) {
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

  #define SUB_CELLHASH(ix, iy, iz) \
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
          unsigned int h = SUB_CELLHASH(ix, iy, iz);
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
          for (int it=head[SUB_CELLHASH(ix, iy, iz)]; it >= 0; it = next[it]) {
            int j = entry[it];
            if (j <= i || seen[j]) {
              continue;
            }
            seen[j] = 1;
            found[nfound++] = j;

            // flex-level eligibility: one comparison that skips the AABB work (e.g.
            // same-flex neighborhoods of a flex with self-collision disabled)
            if (!mj_substepPairEligible(m, eflex[i], eflex[j])) {
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

  #undef SUB_CELLHASH

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
int mj_substepFeasible(const mjModel* m, mjData* d, mjSubstepPair* badpair) {
  mj_markStack(d);

  // generous capacity: feasibility uses zero margin, so overlaps are near-touching pairs
  int npairmax = 64*m->nflexelem + 1024;
  mjSubstepPair* pairs = mjSTACKALLOC(d, npairmax, mjSubstepPair);
  int npair = mj_substepBroadphase(m, d, NULL, NULL, 0, pairs, npairmax);
  if (npair > npairmax) {
    npair = npairmax;
  }

  int feasible = 1;
  for (int i=0; i < npair; i++) {
    mjtNum vert1[9], vert2[9];
    int f1 = pairs[i].f1, f2 = pairs[i].f2;
    packElem(m, d->flexvert_xpos, f1, pairs[i].e1, vert1);
    packElem(m, d->flexvert_xpos, f2, pairs[i].e2, vert2);
    mjtNum dist = mj_substepElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
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


// ownership boundary for the legacy soft-contact fallback, see header. the substep
// solver owns every pair it can move (penalty pressure acts at all depths, so there
// is no deep-penetration regime to hand off); double ownership would put two
// pressure sources on one pair, the split-authority failure mode
mjtNum mj_substepLegacyDistance(const mjModel* m, int f1, int f2) {
  // the substep estimator injects forces pre-integration and is integrator-agnostic;
  // only RK4 is excluded (it re-runs the forward pipeline per stage, which would
  // re-trigger the estimator with inconsistent intermediate states)
  if (m->opt.integrator == mjINT_RK4) {
    return mjMAXVAL;
  }
  if (substepSolvable(m, f1) || substepSolvable(m, f2)) {
    return -mjMAXVAL;  // never legacy
  }
  return mjMAXVAL;  // no movable side: always legacy
}


//---------------------------------- implicit elastic model ---------------------------------------



// CCD filter constant
#define mjSUB_DMARGIN    0.1    // CCD filter preserves this fraction of the current gap

// gather pair vertices: global ids, count
static int pairVerts(const mjModel* m, const mjSubstepPair* p, int gid[6]) {
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
static mjtNum caFilterPair(const mjModel* m, const mjSubstepPair* p, const mjtNum* vertpos,
                           const mjtNum* dx, const int* vertslot, mjtNum abstain) {
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
    mjtNum dist = mj_substepElemDistance(vert, dim1, vert + 3*nv1, dim2, NULL, NULL);
    if (iter == 0) {
      if (dist <= xi) {
        // already past the surface: the surface invariant is lost, but the crossing
        // invariant must survive any load and step size, so these pairs keep a
        // fraction of their remaining centerline distance instead of abstaining
        // (at large timesteps the old abstention let loaded pairs cross a face in
        // a single step, faster than the soft-contact fallback could push back).
        // the abstention threshold is the caller's: the substep loop gives up at a
        // fraction of xi (its state is discarded, deep pairs only burn iterations),
        // the projection only at exact piercing (it is the last line of defense)
        if (dist < abstain) {
          return 1;  // effectively pierced: nothing left to protect reliably
        }
        // proportional keep: the pair retains a fraction of its remaining centerline
        // distance each step, decaying geometrically but never crossing. an absolute
        // floor would be robust to sustained loads at large timesteps, but parked
        // pairs throttle the whole scene through the global step fraction (per-island
        // line search is the prerequisite for that, see the large-timestep notes)
        floor = 0;
        keep = mjSUB_DMARGIN*dist;
      } else {
        keep = mjSUB_DMARGIN*(dist - xi);
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


//---------------------------------- substep pressure module --------------------------------------

#define mjSUB_KMAX       16     // substep count cap
#define mjSUB_ACTBAND    0.25   // penalty activation gap as a fraction of combined
                                // thickness: pressure must exist inside the band the
                                // CCD filter protects, or the filter binds and parks
#define mjSUB_BETA       0.2    // penalty stiffness as a fraction of the explicit
                                // stability limit m*(2/h_sub)^2 at the chosen k

// per-pair accumulated-motion sum over a precomputed slot row (6 entries, -1 padded):
// the summed motion of the pair's own vertices since its last exact evaluation bounds
// the pair's gap shrink (by twice the sum); replaces the global motion accumulator,
// under which one fast vertex anywhere forces distance kernels on every near pair
static mjtNum pairMotionSum(const int* pslot, const mjtNum* vmacc) {
  mjtNum s = 0;
  for (int vi=0; vi < 6; vi++) {
    if (pslot[vi] >= 0) {
      s += vmacc[pslot[vi]];
    }
  }
  return s;
}


// largest displacement of the pair's own vertices, from precomputed per-vertex norms:
// admission bound for the CCD filter (a pair whose remaining budget exceeds twice
// this cannot cross this advance)
static mjtNum pairMaxDx(const int* pslot, const mjtNum* dxn) {
  mjtNum mu = 0;
  for (int vi=0; vi < 6; vi++) {
    if (pslot[vi] >= 0 && dxn[pslot[vi]] > mu) {
      mu = dxn[pslot[vi]];
    }
  }
  return mu;
}


// substep contact-impulse estimator: integrate a LOCAL copy of the flex vertex DOFs
// at h/k with native penalty contacts and the CCD filter per substep, environment
// frozen at its start-of-step forces, and accumulate the contact impulses applied
// along the local trajectory into jout (3*nmv, per moving vertex). the local state
// is discarded: the only output is the time-averaged contact impulse, which the
// caller injects as a force for the unmodified global integrator to consume. local
// elasticity deltas (bending) improve the prediction but are NOT accumulated — the
// global step owns elasticity; only contact terms cross the boundary
static void substepLoop(const mjModel* m, mjData* d,
                        int nmv, const int* slotgv,
                        const int* vertslot, const int* dofadr, const mjtNum* mass,
                        const mjtNum* xprev, const mjtNum* xhat, mjtNum* jout,
                        mjtNum* vertpos, mjtNum* vertswept, mjSubstepPair* pairs,
                        int npairmax, mjtNum* gapeval, mjtNum* accateval,
                        mjtNum maxdhat, mjtNum slack) {
  int nv = m->nv;
  mjtNum h = m->opt.timestep;

  mj_markStack(d);
  mjtNum* x = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* v = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* aout = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* dx = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* dxn = mjSTACKALLOC(d, nmv, mjtNum);

  // in-substep elasticity: the outer acceleration carries all internal forces at
  // their start-of-step values; the substeps add the DELTAS as the state moves, so
  // contact pushes meet an elastic load path within the window (substepping contacts
  // without elasticity is the configuration nobody in the literature uses)
  mjtNum* scat = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* kvec = mjSTACKALLOC(d, nv, mjtNum);
  int hasbend = 0;
  for (int f=0; f < m->nflex; f++) {
    if (!m->flex_interp[f] && m->flex_dim[f] == 2 && m->flex_bendingadr[f] >= 0 &&
        !m->flex_rigid[f]) {
      hasbend = 1;
    }
  }

  // equality-constrained edges as delta springs: stiffness from the row impedance
  int nedge = 0;
  int* es1 = NULL;
  int* es2 = NULL;
  mjtNum* ek = NULL;
  mjtNum* el0 = NULL;
  {
    int nemax = 0;
    for (int f=0; f < m->nflex; f++) {
      nemax += m->flex_edgenum[f];
    }
    es1 = mjSTACKALLOC(d, nemax > 0 ? nemax : 1, int);
    es2 = mjSTACKALLOC(d, nemax > 0 ? nemax : 1, int);
    ek = mjSTACKALLOC(d, nemax > 0 ? nemax : 1, mjtNum);
    el0 = mjSTACKALLOC(d, nemax > 0 ? nemax : 1, mjtNum);
    // DISABLED pending authority transfer (spec section 3): with the equality rows
    // still in the outer solve, in-substep edge springs create split authority over
    // stretch, and every anchor choice fails a gate: L0 anchoring breaks friction
    // grasps (PinchingSucceeds), window anchoring locks draped sliding (the corner
    // chord-breathing of a coarse polyline). transferring the rows out of the outer
    // solve for substepped flexes gives the springs single ownership; until then the
    // rows act alone at the outer rate, carried by the frozen acceleration
    for (int i=0; 0 && i < m->neq; i++) {
      if (m->eq_type[i] != mjEQ_FLEX || !d->eq_active[i]) {
        continue;
      }
      int f = m->eq_obj1id[i];
      if (!substepSolvable(m, f)) {
        continue;
      }
      // row impedance: k = m_eff/timeconst^2 (the constraint's solref stiffness)
      mjtNum tc = m->eq_solref[mjNREF*i] > mjMINVAL ? m->eq_solref[mjNREF*i] : 0.02;
      for (int e=0; e < m->flex_edgenum[f]; e++) {
        const int* ev = m->flex_edge + 2*(m->flex_edgeadr[f] + e);
        int sa = vertslot[m->flex_vertadr[f] + ev[0]];
        int sb = vertslot[m->flex_vertadr[f] + ev[1]];
        if (sa < 0 && sb < 0) {
          continue;
        }
        es1[nedge] = sa;
        es2[nedge] = sb;
        mjtNum me = (sa >= 0 && sb >= 0) ? 0.5*(mass[sa] + mass[sb]) :
                    (sa >= 0 ? mass[sa] : mass[sb]);
        ek[nedge] = me/(tc*tc);
        // anchor at the window-start length, not the rest length: the springs carry
        // the CHANGE in the rows' force as the state moves (delta linearization, like
        // the bending term); an L0 anchor gives them an opinion about the settled
        // stretch state and fights the outer rows (breaks friction grasps)
        if (sa >= 0 && sb >= 0) {
          mjtNum dvec[3];
          mju_sub3(dvec, xprev + 3*sa, xprev + 3*sb);
          el0[nedge] = mju_norm3(dvec);
        } else {
          el0[nedge] = m->flexedge_length0[m->flex_edgeadr[f] + e];
        }
        nedge++;
      }
    }
  }

  // pre-integration state: d->qvel is the start-of-step velocity and d->qacc is
  // forward's acceleration — on 3-slide flex vertex DOFs the mass matrix is diagonal,
  // so qacc is exactly (qfrc_smooth + qfrc_constraint)/m, the frozen outer force.
  // no integrator state is reconstructed post hoc: the qacc-trap class is absent
  for (int k=0; k < nmv; k++) {
    for (int c=0; c < 3; c++) {
      int dof = dofadr[3*k + c];
      aout[3*k + c] = d->qacc[dof];
      v[3*k + c] = d->qvel[dof];
    }
  }
  mju_copy(x, xprev, 3*nmv);
  mju_zero(jout, 3*nmv);


  // envelope broadphase once per step: boxes span (xprev, xhat) plus slack
  for (int k=0; k < nmv; k++) {
    mju_copy3(vertpos + 3*slotgv[k], xprev + 3*k);
    mju_copy3(vertswept + 3*slotgv[k], xhat + 3*k);
  }
  int npair = mj_substepBroadphase(m, d, vertpos, vertswept, maxdhat + slack, pairs, npairmax);
  if (npair > npairmax) {
    fprintf(stderr, "SUBSTEP SATURATED: npair=%d npairmax=%d\n", npair, npairmax);
    npair = npairmax;
  }

  // initial budgets: exact gaps at xprev; k from contact presence (calm steps: k=1);
  // pair slot rows precomputed once so the per-substep budget checks are array reads
  int* pairslot = mjSTACKALLOC(d, 6*npair > 0 ? 6*npair : 1, int);
  int live = 0;
  for (int i=0; i < npair; i++) {
    mjtNum vert1[9], vert2[9];
    int f1 = pairs[i].f1, f2 = pairs[i].f2;
    packElem(m, vertpos, f1, pairs[i].e1, vert1);
    packElem(m, vertpos, f2, pairs[i].e2, vert2);
    mjtNum dist = mj_substepElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
                                     NULL, NULL);
    mjtNum xi = m->flex_radius[f1] + m->flex_radius[f2];
    gapeval[i] = dist - xi;
    accateval[i] = 0;
    if (gapeval[i] < (mjSUB_ACTBAND + 0.1)*xi) {
      live = 1;
    }
    int gid[6];
    int nvp = pairVerts(m, pairs + i, gid);
    for (int vi=0; vi < 6; vi++) {
      pairslot[6*i + vi] = vi < nvp ? vertslot[gid[vi]] : -1;
    }
  }
  int ksub = live ? mjSUB_KMAX : 1;
  mjtNum hs = h/ksub;

  // per-vertex accumulated motion: budgets decay per pair, from its own vertices
  mjtNum* vmacc = mjSTACKALLOC(d, nmv, mjtNum);
  mju_zero(vmacc, nmv);

  for (int s=0; s < ksub; s++) {
    // kick: outer forces (gravity, passive, outer-rate constraint rows)
    mju_addToScl(v, aout, hs, 3*nmv);

    // bending delta: f(x) - f(xprev) = -K_bend*(x - xprev), constant stencils
    if (hasbend) {
      mju_zero(scat, nv);
      for (int k=0; k < nmv; k++) {
        for (int c=0; c < 3; c++) {
          scat[dofadr[3*k + c]] = x[3*k + c] - xprev[3*k + c];
        }
      }
      mju_zero(kvec, nv);
      mjd_flexBend_mul(m, d, kvec, scat, 1, 0);
      // restoring: f(x) - f(xprev) = -K_bend*(x - xprev) (A = ... + h^2*K_bend)
      for (int k=0; k < nmv; k++) {
        for (int c=0; c < 3; c++) {
          v[3*k + c] -= hs*kvec[dofadr[3*k + c]]/mass[k];
        }
      }
    }

    // edge-equality delta springs: restore the rest length as the state moves; the
    // outer rows already push toward it at start-of-step state, this is the change
    for (int e=0; e < nedge; e++) {
      if (es1[e] < 0 || es2[e] < 0) {
        continue;  // edges with a fixed end keep their outer-rate treatment
      }
      mjtNum dvec[3];
      mju_sub3(dvec, x + 3*es1[e], x + 3*es2[e]);
      mjtNum L = mju_norm3(dvec);
      if (L < mjMINVAL) {
        continue;
      }
      mjtNum c = L - el0[e];
      mjtNum fmag = -ek[e]*c;
      mjtNum u[3];
      mju_scl3(u, dvec, 1/L);
      mju_addToScl3(v + 3*es1[e], u, hs*fmag/mass[es1[e]]);
      mju_addToScl3(v + 3*es2[e], u, -hs*fmag/mass[es2[e]]);
    }

    // penalty contacts on live pairs: MuJoCo-style spring-damper on the gap, applied
    // through the distance gradient; stiffness set by the explicit stability budget
    for (int i=0; i < npair; i++) {
      mjtNum xi = m->flex_radius[pairs[i].f1] + m->flex_radius[pairs[i].f2];
      mjtNum act = mjSUB_ACTBAND*xi;
      mjtNum psum = pairMotionSum(pairslot + 6*i, vmacc);
      if (gapeval[i] - 2*(psum - accateval[i]) > act) {
        continue;  // budget: cannot be inside the activation band
      }
      mjtNum vert1[9], vert2[9], graddist[18];
      int f1 = pairs[i].f1, f2 = pairs[i].f2;
      int dim1 = m->flex_dim[f1], dim2 = m->flex_dim[f2];
      packElem(m, vertpos, f1, pairs[i].e1, vert1);
      packElem(m, vertpos, f2, pairs[i].e2, vert2);
      mjtNum dist = mj_substepElemDistance(vert1, dim1, vert2, dim2, NULL, graddist);
      gapeval[i] = dist - xi;
      accateval[i] = psum;
      mjtNum pen = act - gapeval[i];  // engagement depth into the activation band
      if (pen <= 0) {
        continue;
      }

      // per-pair effective mass: smallest vertex mass involved (stability-safe)
      int gid[6];
      int nvp = pairVerts(m, pairs + i, gid);
      mjtNum meff = mjMAXVAL;
      for (int vi=0; vi < nvp; vi++) {
        int slot = vertslot[gid[vi]];
        if (slot >= 0 && mass[slot] < meff) {
          meff = mass[slot];
        }
      }
      if (meff == mjMAXVAL) {
        continue;
      }
      mjtNum kp = mjSUB_BETA*meff*(2/hs)*(2/hs)/4;
      mjtNum kd = 2*mju_sqrt(kp*meff);

      // gap rate from current velocities through the gradient
      mjtNum ddot = 0;
      for (int vi=0; vi < nvp; vi++) {
        int slot = vertslot[gid[vi]];
        if (slot >= 0) {
          ddot += mju_dot3(graddist + 3*vi, v + 3*slot);
        }
      }

      // spring-damper with symmetric damping while engaged, total force clamped
      // repulsive (MuJoCo contact semantics). a one-sided damper rectifies contact
      // oscillations into net energy injection: damped on approach, free on
      // separation gains energy every cycle
      mjtNum fn = kp*pen - kd*ddot;
      if (fn <= 0) {
        continue;
      }
      for (int vi=0; vi < nvp; vi++) {
        int slot = vertslot[gid[vi]];
        if (slot >= 0) {
          mju_addToScl3(v + 3*slot, graddist + 3*vi, hs*fn/mass[slot]);
          mju_addToScl3(jout + 3*slot, graddist + 3*vi, hs*fn);
        }
      }
    }

    // drift: CCD-filtered advance (the invariant; pressure above keeps it from binding)
    // displacement and its per-vertex norms built once per substep (building them per
    // candidate pair was quadratic: candidates x vertices, catastrophic when contact-rich)
    mjtNum mu = 0;
    for (int k=0; k < nmv; k++) {
      mju_scl3(dx + 3*k, v + 3*k, hs);
      dxn[k] = mju_norm3(dx + 3*k);
      mu = dxn[k] > mu ? dxn[k] : mu;
    }
    mjtNum alpha = 1;
    for (int i=0; i < npair; i++) {
      mjtNum remaining = gapeval[i] - 2*(pairMotionSum(pairslot + 6*i, vmacc)
                                         - accateval[i]);
      // global bound rejects the bulk for one comparison; the pair bound the rest
      if (remaining > 2*mu || remaining > 2*pairMaxDx(pairslot + 6*i, dxn)) {
        continue;
      }
      mjtNum xi = m->flex_radius[pairs[i].f1] + m->flex_radius[pairs[i].f2];
      mjtNum ai = caFilterPair(m, pairs + i, vertpos, dx, vertslot, 0.01*xi);
      alpha = ai < alpha ? ai : alpha;
    }
    for (int k=0; k < nmv; k++) {
      mju_addToScl3(x + 3*k, v + 3*k, alpha*hs);
      mju_copy3(vertpos + 3*slotgv[k], x + 3*k);
      vmacc[k] += alpha*dxn[k];
    }

    // filter-consistent velocity: when the advance was clamped, remove the approaching
    // normal velocity of near-contact pairs (an inelastic impulse). without this the
    // penalty kick re-fires against the clamped position every substep and pumps
    // energy into a ratchet
    if (alpha < 1) {
      for (int i=0; i < npair; i++) {
        mjtNum psum = pairMotionSum(pairslot + 6*i, vmacc);
        mjtNum remaining = gapeval[i] - 2*(psum - accateval[i]);
        if (remaining > 2*mu || remaining > 2*pairMaxDx(pairslot + 6*i, dxn)) {
          continue;
        }
        mjtNum vert1[9], vert2[9], graddist[18];
        int f1 = pairs[i].f1, f2 = pairs[i].f2;
        packElem(m, vertpos, f1, pairs[i].e1, vert1);
        packElem(m, vertpos, f2, pairs[i].e2, vert2);
        mjtNum dist = mj_substepElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
                                         NULL, graddist);
        mjtNum xi = m->flex_radius[f1] + m->flex_radius[f2];
        gapeval[i] = dist - xi;
        accateval[i] = psum;
        if (gapeval[i] > mjSUB_ACTBAND*xi) {
          continue;
        }
        int gid[6];
        int nvp = pairVerts(m, pairs + i, gid);
        mjtNum ddot = 0, wsum = 0;
        for (int vi=0; vi < nvp; vi++) {
          int slot = vertslot[gid[vi]];
          if (slot >= 0) {
            ddot += mju_dot3(graddist + 3*vi, v + 3*slot);
            wsum += mju_dot3(graddist + 3*vi, graddist + 3*vi)/mass[slot];
          }
        }
        if (ddot >= 0 || wsum < mjMINVAL) {
          continue;
        }
        mjtNum lam = -ddot/wsum;
        for (int vi=0; vi < nvp; vi++) {
          int slot = vertslot[gid[vi]];
          if (slot >= 0) {
            mju_addToScl3(v + 3*slot, graddist + 3*vi, lam/mass[slot]);
            mju_addToScl3(jout + 3*slot, graddist + 3*vi, lam);
          }
        }
      }
    }
  }

  mj_freeStack(d);
}


// active check shared by the estimator and the projection: substep flag enabled,
// integrator owned (everything but RK4, which re-runs the forward pipeline per
// stage), at least one solvable flex
static int substepActive(const mjModel* m) {
  if (mjDISABLED(mjDSBL_SUBSTEP) || m->opt.integrator == mjINT_RK4) {
    return 0;
  }
  for (int f=0; f < m->nflex; f++) {
    if (substepSolvable(m, f)) {
      return 1;
    }
  }
  return 0;
}


// identify moving vertices: bodies with exactly 3 axis-aligned slide joints; all other
// vertices (pinned or unsupported layouts) are treated as fixed obstacles; fills the
// vertex->slot map and per-slot global-vertex/qpos/dof addresses, returns slot count
static int substepGather(const mjModel* m, int* vertslot, int* slotgv,
                         int* qposadr, int* dofadr) {
  int nmv = 0;
  for (int gv=0; gv < m->nflexvert; gv++) {
    vertslot[gv] = -1;
  }
  for (int f=0; f < m->nflex; f++) {
    if (!substepSolvable(m, f)) {
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
  return nmv;
}


// contact-impulse estimator, see header
void mj_substepSolve(const mjModel* m, mjData* d) {
  int nflexvert = m->nflexvert;
  mjtNum h = m->opt.timestep;

  if (!substepActive(m)) {
    return;
  }

  mj_markStack(d);

  int* vertslot = mjSTACKALLOC(d, nflexvert, int);
  int* slotgv = mjSTACKALLOC(d, nflexvert, int);
  int* qposadr = mjSTACKALLOC(d, 3*nflexvert, int);
  int* dofadr = mjSTACKALLOC(d, 3*nflexvert, int);
  int nmv = substepGather(m, vertslot, slotgv, qposadr, dofadr);
  if (!nmv) {
    mj_freeStack(d);
    return;
  }

  // states: x_prev from current (pre-integration) kinematics; xhat is the predicted
  // endpoint under the frozen outer dynamics, used only to size the swept envelope
  mjtNum* xprev = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* xhat = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* jout = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum* mass = mjSTACKALLOC(d, nmv, mjtNum);

  for (int k=0; k < nmv; k++) {
    int gv = slotgv[k];
    mju_copy3(xprev + 3*k, d->flexvert_xpos + 3*gv);
    // pre-integration qvel is the start-of-step velocity: the predicted endpoint
    // adds the acceleration term h^2*qacc that the integrated qpos used to carry
    for (int c=0; c < 3; c++) {
      int dof = dofadr[3*k + c];
      xhat[3*k + c] = xprev[3*k + c] + h*d->qvel[dof] + h*h*d->qacc[dof];
    }
    mass[k] = m->body_mass[m->flex_vertbodyid[gv]];
  }
  // full vertex position buffers for distance evaluation (fixed verts stay at pre-step)
  mjtNum* vertpos = mjSTACKALLOC(d, 3*nflexvert, mjtNum);
  mjtNum* vertswept = mjSTACKALLOC(d, 3*nflexvert, mjtNum);
  mju_copy(vertpos, d->flexvert_xpos, 3*nflexvert);
  mju_copy(vertswept, d->flexvert_xpos, 3*nflexvert);

  int npairmax = 64*m->nflexelem + 1024;
  mjSubstepPair* pairs = mjSTACKALLOC(d, npairmax, mjSubstepPair);
  // distance budgets: gapeval[i] is the exact surface gap of pair i at its last
  // evaluation and accateval[i] the motion accumulator at that moment; the pair's gap
  // can have shrunk by at most twice the motion accumulated since, so most pairs are
  // skipped with one subtraction instead of a distance kernel
  mjtNum* gapeval = mjSTACKALLOC(d, npairmax, mjtNum);
  mjtNum* accateval = mjSTACKALLOC(d, npairmax, mjtNum);
  // broadphase margin: the activation band of the largest flex
  mjtNum maxdhat = 0;
  for (int f=0; f < m->nflex; f++) {
    if (substepEligible(m, f) && 0.2*m->flex_radius[f] > maxdhat) {
      maxdhat = 0.2*m->flex_radius[f];
    }
  }

  // local substep simulation: native penalty contacts at h/k behind the CCD filter,
  // environment frozen; produces the accumulated contact impulse, discards its state
  substepLoop(m, d, nmv, slotgv, vertslot, dofadr, mass, xprev, xhat, jout,
              vertpos, vertswept, pairs, npairmax, gapeval, accateval,
              maxdhat, 4*maxdhat);

  // inject the time-averaged contact force for the global integrator to consume:
  // the implicit integrators and Euler-with-damping rebuild their solve from
  // qfrc_smooth + qfrc_constraint, plain Euler advances d->qacc directly — update
  // both so every owned path sees the same force (slide DOFs: diagonal mass)
  for (int k=0; k < nmv; k++) {
    for (int c=0; c < 3; c++) {
      int dof = dofadr[3*k + c];
      mjtNum fc = jout[3*k + c]/h;
      d->qfrc_constraint[dof] += fc;
      d->qacc[dof] += fc/mass[k];
    }
  }

  mj_freeStack(d);
}


// CCD projection of the integrated step, see header
void mj_substepProject(const mjModel* m, mjData* d) {
  int nflexvert = m->nflexvert;
  mjtNum h = m->opt.timestep;

  if (!substepActive(m)) {
    return;
  }

  mj_markStack(d);

  int* vertslot = mjSTACKALLOC(d, nflexvert, int);
  int* slotgv = mjSTACKALLOC(d, nflexvert, int);
  int* qposadr = mjSTACKALLOC(d, 3*nflexvert, int);
  int* dofadr = mjSTACKALLOC(d, 3*nflexvert, int);
  int nmv = substepGather(m, vertslot, slotgv, qposadr, dofadr);
  if (!nmv) {
    mj_freeStack(d);
    return;
  }

  // realized step displacement: kinematics has not re-run since integration, so
  // d->flexvert_xpos still holds the pre-step positions, and on slide joints the
  // integrators advance qpos by exactly h*qvel_new (Euler and implicit alike)
  mjtNum* dx = mjSTACKALLOC(d, 3*nmv, mjtNum);
  mjtNum mu = 0;
  for (int k=0; k < nmv; k++) {
    for (int c=0; c < 3; c++) {
      dx[3*k + c] = h*d->qvel[dofadr[3*k + c]];
    }
    mjtNum n2 = mju_dot3(dx + 3*k, dx + 3*k);
    mu = n2 > mu ? n2 : mu;
  }
  mu = mju_sqrt(mu);
  if (mu < mjMINVAL) {
    mj_freeStack(d);
    return;
  }

  // swept candidates over the realized motion
  mjtNum* vertpos = mjSTACKALLOC(d, 3*nflexvert, mjtNum);
  mjtNum* vertswept = mjSTACKALLOC(d, 3*nflexvert, mjtNum);
  mju_copy(vertpos, d->flexvert_xpos, 3*nflexvert);
  mju_copy(vertswept, d->flexvert_xpos, 3*nflexvert);
  for (int k=0; k < nmv; k++) {
    mju_addTo3(vertswept + 3*slotgv[k], dx + 3*k);
  }
  int npairmax = 64*m->nflexelem + 1024;
  mjSubstepPair* pairs = mjSTACKALLOC(d, npairmax, mjSubstepPair);
  int npair = mj_substepBroadphase(m, d, vertpos, vertswept, 0, pairs, npairmax);
  npair = npair > npairmax ? npairmax : npair;

  // one conservative-advancement sweep: the global step fraction that preserves the
  // non-crossing invariant. the estimator's pressure makes binding rare; this is the
  // safety net for the realized global motion (which includes implicit elasticity and
  // outer rows the local prediction does not see)
  mjtNum alpha = 1;
  for (int i=0; i < npair; i++) {
    // per-pair motion bound: the global bound mu admits nearly every candidate on a
    // scene with one fast vertex, sending thousands of pairs into conservative
    // advancement; the pair's own endpoint motion is the bound that prunes
    int gid[6];
    int nvp = pairVerts(m, pairs + i, gid);
    mjtNum mui = 0;
    for (int vi=0; vi < nvp; vi++) {
      int slot = vertslot[gid[vi]];
      if (slot >= 0) {
        mjtNum n2 = mju_dot3(dx + 3*slot, dx + 3*slot);
        mui = n2 > mui ? n2 : mui;
      }
    }
    if (mui < mjMINVAL) {
      continue;
    }
    mui = mju_sqrt(mui);
    mjtNum vert1[9], vert2[9];
    int f1 = pairs[i].f1, f2 = pairs[i].f2;
    packElem(m, vertpos, f1, pairs[i].e1, vert1);
    packElem(m, vertpos, f2, pairs[i].e2, vert2);
    mjtNum dist = mj_substepElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
                                     NULL, NULL);
    mjtNum xi = m->flex_radius[f1] + m->flex_radius[f2];
    // each element endpoint moves at most mui: the gap cannot shrink by more than 2*mui
    if (dist - xi > 2*mui) {
      continue;
    }
    mjtNum ai = caFilterPair(m, pairs + i, vertpos, dx, vertslot, mjMINVAL);
    alpha = ai < alpha ? ai : alpha;
  }

  // clamp: pull qpos back along the realized step
  if (alpha < 1) {
    for (int k=0; k < nmv; k++) {
      for (int c=0; c < 3; c++) {
        d->qpos[qposadr[3*k + c]] -= (1 - alpha)*dx[3*k + c];
      }
    }

    // inelastic normal impulse on near pairs: the position clamp alone leaves the
    // approaching velocity intact, so under sustained load a pair re-approaches at
    // full speed every step and the proportional keep floor grinds to floating-point
    // crossing within a handful of steps (the keep concedes 90% of the remaining
    // distance per event). removing the approaching normal velocity — the same
    // impulse the substep loop applies internally against its kick ratchet — turns
    // sustained-load approach into a force balance the estimator's pressure can hold
    for (int k=0; k < nmv; k++) {
      mju_addToScl3(vertpos + 3*slotgv[k], dx + 3*k, alpha);  // post-clamp positions
    }
    for (int i=0; i < npair; i++) {
      mjtNum vert1[9], vert2[9], graddist[18];
      int f1 = pairs[i].f1, f2 = pairs[i].f2;
      packElem(m, vertpos, f1, pairs[i].e1, vert1);
      packElem(m, vertpos, f2, pairs[i].e2, vert2);
      mjtNum dist = mj_substepElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
                                       NULL, graddist);
      mjtNum xi = m->flex_radius[f1] + m->flex_radius[f2];
      if (dist - xi > mjSUB_ACTBAND*xi) {
        continue;
      }
      int gid[6];
      int nvp = pairVerts(m, pairs + i, gid);
      mjtNum ddot = 0, wsum = 0;
      for (int vi=0; vi < nvp; vi++) {
        int slot = vertslot[gid[vi]];
        if (slot >= 0) {
          for (int c=0; c < 3; c++) {
            ddot += graddist[3*vi + c]*d->qvel[dofadr[3*slot + c]];
          }
          mjtNum gv2 = mju_dot3(graddist + 3*vi, graddist + 3*vi);
          wsum += gv2/m->body_mass[m->flex_vertbodyid[gid[vi]]];
        }
      }
      if (ddot >= 0 || wsum < mjMINVAL) {
        continue;
      }
      mjtNum lam = -ddot/wsum;
      for (int vi=0; vi < nvp; vi++) {
        int slot = vertslot[gid[vi]];
        if (slot >= 0) {
          mjtNum minv = 1/m->body_mass[m->flex_vertbodyid[gid[vi]]];
          for (int c=0; c < 3; c++) {
            d->qvel[dofadr[3*slot + c]] += lam*graddist[3*vi + c]*minv;
          }
        }
      }
    }
  }

  mj_freeStack(d);
}
