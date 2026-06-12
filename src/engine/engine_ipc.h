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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_IPC_H_
#define MUJOCO_SRC_ENGINE_ENGINE_IPC_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

// a candidate flex element pair produced by the broadphase
typedef struct {
  int f1, e1;        // first flex, element
  int f2, e2;        // second flex, element
} mjIPCPair;

// closed-form distance between two flex elements (segment or triangle primitives, by
// flex dim), with witness points and the distance gradient with respect to the element
// vertices: grad is layed out as [dvert1_0(3), dvert1_1(3), ..., dvert2_0(3), ...] with
// nvert1 = dim1+1 and nvert2 = dim2+1 entries; witness/grad pointers are nullable
// the distance is between the underlying primitives (radii are NOT subtracted)
// vert1/vert2 point at packed vertex positions, 3*(dim+1) numbers each
MJAPI mjtNum mj_ipcElemDistance(const mjtNum* vert1, int dim1, const mjtNum* vert2, int dim2,
                                mjtNum witness[6], mjtNum* grad);

// IPC barrier value and derivatives at primitive distance d, with thickness offset xi and
// activation width dhat: zero for d >= xi + dhat, smooth and unbounded as d -> xi
// b   = -(s - dhat)^2 * log(s/dhat), s = d - xi, for s in (0, dhat)
// db and ddb (nullable) receive db/dd and d2b/dd2
MJAPI mjtNum mj_ipcBarrier(mjtNum d, mjtNum xi, mjtNum dhat, mjtNum* db, mjtNum* ddb);

// broadphase over flex elements: uniform-grid spatial hash over element AABBs inflated by
// (element radius + margin); appends to pairs (capacity npairmax) all element pairs from
// IPC-eligible flexes that overlap and do not share a vertex; same-flex pairs are included
// when the flex has self-collision enabled; returns the number of pairs found (may exceed
// npairmax, in which case only npairmax were written)
// vertpos overrides d->flexvert_xpos when non-NULL (solver iterates); same layout;
// vertpos2 (nullable) extends each element AABB to the union over both position sets,
// for swept candidate gathering along a displacement
MJAPI int mj_ipcBroadphase(const mjModel* m, mjData* d, const mjtNum* vertpos,
                           const mjtNum* vertpos2, mjtNum margin, mjIPCPair* pairs,
                           int npairmax);

// return 1 if the flex pair (f1, f2 by flex id; f1 == f2 for self-collision) is handled
// by the variational flex solver: both sides collidable (dim 1 or 2), self-collision
// enabled where applicable, contype/conaffinity compatible
MJAPI int mj_ipcPairEligible(const mjModel* m, int f1, int f2);

// check flex-flex feasibility: all eligible element pairs separated by more than their
// combined radii; returns 1 if feasible, otherwise 0 with the first violating pair
// (nullable) written to badpair
MJAPI int mj_ipcFeasible(const mjModel* m, mjData* d, mjIPCPair* badpair);

// distance below which the legacy soft-contact fallback owns a flex-flex pair (the
// barrier solve's extended force plateaus there); pairs above it produce no contacts
MJAPI mjtNum mj_ipcLegacyDistance(const mjModel* m, int f1, int f2);

// barrier projection of the integrated flex state: minimizes the incremental potential
// (inertia toward the integrated target + barriers on flex-flex distances) over the
// vertex DOFs of eligible flexes (dim 1 or 2, flex_interp == 0), starting from the
// pre-step positions, with a conservative-advancement filtered line search that
// guarantees no flex-flex crossing; elasticity and all other forces act through the
// integrated target (one-step lag, same splitting as the rigid-flex soft contacts);
// pinned vertices and ineligible flexes participate as fixed obstacles
// called at the end of mj_step when mjENBL_FLEXIPC is enabled
MJAPI void mj_flexIPC(const mjModel* m, mjData* d);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_IPC_H_
