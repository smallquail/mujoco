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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_SUBSTEP_H_
#define MUJOCO_SRC_ENGINE_ENGINE_SUBSTEP_H_

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
} mjSubstepPair;

// closed-form distance between two flex elements (segment or triangle primitives, by
// flex dim), with witness points and the distance gradient with respect to the element
// vertices: grad is layed out as [dvert1_0(3), dvert1_1(3), ..., dvert2_0(3), ...] with
// nvert1 = dim1+1 and nvert2 = dim2+1 entries; witness/grad pointers are nullable
// the distance is between the underlying primitives (radii are NOT subtracted)
// vert1/vert2 point at packed vertex positions, 3*(dim+1) numbers each
MJAPI mjtNum mj_substepElemDistance(const mjtNum* vert1, int dim1, const mjtNum* vert2, int dim2,
                                mjtNum witness[6], mjtNum* grad);


// broadphase over flex elements: uniform-grid spatial hash over element AABBs inflated by
// (element radius + margin); appends to pairs (capacity npairmax) all element pairs from
// eligible flexes that overlap and do not share a vertex; same-flex pairs are included
// when the flex has self-collision enabled; returns the number of pairs found (may exceed
// npairmax, in which case only npairmax were written)
// vertpos overrides d->flexvert_xpos when non-NULL (solver iterates); same layout;
// vertpos2 (nullable) extends each element AABB to the union over both position sets,
// for swept candidate gathering along a displacement
MJAPI int mj_substepBroadphase(const mjModel* m, mjData* d, const mjtNum* vertpos,
                           const mjtNum* vertpos2, mjtNum margin, mjSubstepPair* pairs,
                           int npairmax);

// return 1 if the flex pair (f1, f2 by flex id; f1 == f2 for self-collision) is handled
// by the variational flex solver: both sides collidable (dim 1 or 2), self-collision
// enabled where applicable, contype/conaffinity compatible
MJAPI int mj_substepPairEligible(const mjModel* m, int f1, int f2);

// check flex-flex feasibility: all eligible element pairs separated by more than their
// combined radii; returns 1 if feasible, otherwise 0 with the first violating pair
// (nullable) written to badpair
MJAPI int mj_substepFeasible(const mjModel* m, mjData* d, mjSubstepPair* badpair);

// distance below which the legacy soft-contact fallback owns a flex-flex pair (the
// barrier solve's extended force plateaus there); pairs above it produce no contacts
MJAPI mjtNum mj_substepLegacyDistance(const mjModel* m, int f1, int f2);



// substep contact-impulse estimator: called from mj_step after forward dynamics and
// BEFORE integration. runs a local substep simulation of the flex vertex DOFs (native
// penalty contacts at h/k behind the CCD filter, environment frozen at its
// start-of-step forces), accumulates the contact impulses applied along the local
// trajectory, and injects them as forces (qfrc_constraint += J/h, qacc updated
// consistently) for the unmodified global integrator to consume. the local state is
// discarded: the global integrator remains the only authority over (qpos, qvel).
// owned integrators: Euler, implicit, implicitfast (RK4 excluded); gated by the
// substep disable flag; pinned vertices and ineligible flexes are fixed obstacles
MJAPI void mj_substepSolve(const mjModel* m, mjData* d);

// CCD projection of the integrated step: called from mj_step AFTER integration.
// reconstructs the realized flex vertex displacements (h*qvel on slide joints, exact
// for Euler and the implicit integrators), runs one conservative-advancement sweep
// over swept candidate pairs, and clamps qpos back along the step by the largest
// fraction that preserves the flex-flex non-crossing invariant. when the clamp binds
// it also removes the approaching normal velocity of near pairs (inelastic impulse):
// a position clamp alone re-approaches at full speed every step and grinds through
// the proportional keep floor to floating-point crossing under sustained load
MJAPI void mj_substepProject(const mjModel* m, mjData* d);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_SUBSTEP_H_
