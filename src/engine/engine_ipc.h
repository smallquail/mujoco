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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_IPC_H_
#define MUJOCO_SRC_ENGINE_ENGINE_IPC_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif

// IPC variational integrator (prototype, mjINT_IPC): owns the full step. Minimizes the per-step
// incremental potential (inertia + flex elastic penalty + C-IPC contact barriers) by projected
// Newton with a step-capped line search. Currently supports a single 2D flex with self-contact
// (vertex-triangle + edge-edge) and flex-vs-static-geom contact; falls back to Euler otherwise.
MJAPI void mj_IPC(const mjModel* m, mjData* d);

// Internal IPC kernels exposed for unit tests only (engine_ipc_test.cc); not a supported API.
MJAPI mjtNum mj_ipcBarrier(mjtNum g, mjtNum ghat, mjtNum* d1, mjtNum* d2);
MJAPI mjtNum mj_ipcPtTri(const mjtNum* p, const mjtNum* a, const mjtNum* b, const mjtNum* c);
MJAPI mjtNum mj_ipcSegSeg(const mjtNum* p1, const mjtNum* p2, const mjtNum* q1, const mjtNum* q2);
MJAPI mjtNum mj_ipcGeomDist(const mjModel* m, const mjData* d, int gi, const mjtNum* x, mjtNum* n);
MJAPI int mj_ipcGeomVerts(const mjModel* m, const mjData* d, int gi, mjtNum* out);
MJAPI int mj_ipcGeomEdges(const mjModel* m, const mjData* d, int gi, mjtNum* out);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_IPC_H_
