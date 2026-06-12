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

// Tests for engine/engine_ipc.c.

#include "src/engine/engine_ipc.h"

#include <cmath>

#include <mujoco/mujoco.h>
#include <mujoco/mjtype.h>
#include "test/fixture.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mujoco {
namespace {

using ::testing::NotNull;

using IpcTest = MujocoTest;

// two parallel 1D flex strings, axes 0.05 apart, radius 0.01
constexpr char kTwoStringsXml[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <flexcomp name="string1" type="grid" dim="1" count="5 1 1" spacing="0.1 1 1"
              radius="0.01" mass="0.05" pos="0 0 0"/>
    <flexcomp name="string2" type="grid" dim="1" count="5 1 1" spacing="0.1 1 1"
              radius="0.01" mass="0.05" pos="0 0.05 0"/>
  </worldbody>
</mujoco>
)";

static mjModel* LoadModel(const char* xml) {
  char error[1024];
  mjModel* model = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(model, NotNull()) << error;
  return model;
}

//---------------------------------- barrier ------------------------------------------------------

// barrier is zero outside the activation band, positive and decreasing inside
TEST_F(IpcTest, BarrierShape) {
  mjtNum xi = 0.002, dhat = 0.01;

  // inactive at and beyond xi + dhat
  EXPECT_EQ(mj_ipcBarrier(xi + dhat, xi, dhat, nullptr, nullptr), 0);
  EXPECT_EQ(mj_ipcBarrier(1.0, xi, dhat, nullptr, nullptr), 0);

  // positive inside, growing toward xi
  mjtNum b1 = mj_ipcBarrier(xi + 0.5*dhat, xi, dhat, nullptr, nullptr);
  mjtNum b2 = mj_ipcBarrier(xi + 0.1*dhat, xi, dhat, nullptr, nullptr);
  EXPECT_GT(b1, 0);
  EXPECT_GT(b2, b1);

  // huge at the infeasible boundary
  EXPECT_GE(mj_ipcBarrier(xi, xi, dhat, nullptr, nullptr), 1e9);
}

// barrier derivatives match finite differences
TEST_F(IpcTest, BarrierDerivatives) {
  mjtNum xi = 0.002, dhat = 0.01, eps = 1e-7;
  for (mjtNum frac : {0.15, 0.4, 0.7, 0.95}) {
    mjtNum d = xi + frac*dhat;
    mjtNum db, ddb;
    mj_ipcBarrier(d, xi, dhat, &db, &ddb);

    mjtNum bp = mj_ipcBarrier(d + eps, xi, dhat, nullptr, nullptr);
    mjtNum bm = mj_ipcBarrier(d - eps, xi, dhat, nullptr, nullptr);
    mjtNum b0 = mj_ipcBarrier(d, xi, dhat, nullptr, nullptr);

    EXPECT_NEAR(db, (bp - bm)/(2*eps), 1e-4*mju_max(1.0, fabs(db))) << "frac " << frac;
    EXPECT_NEAR(ddb, (bp - 2*b0 + bm)/(eps*eps), 1e-2*mju_max(1.0, fabs(ddb)))
        << "frac " << frac;
  }
}

//---------------------------------- element distances --------------------------------------------

// perpendicular crossing segments at height gap: distance = gap
TEST_F(IpcTest, SegSegDistance) {
  mjtNum seg1[6] = {-1, 0, 0, 1, 0, 0};
  mjtNum seg2[6] = {0, -1, 0.3, 0, 1, 0.3};
  mjtNum witness[6];
  mjtNum dist = mj_ipcElemDistance(seg1, 1, seg2, 1, witness, nullptr);
  EXPECT_NEAR(dist, 0.3, 1e-12);
  EXPECT_NEAR(witness[0], 0, 1e-12);   // closest at segment midpoints
  EXPECT_NEAR(witness[3], 0, 1e-12);
  EXPECT_NEAR(witness[5], 0.3, 1e-12);

  // endpoint case: collinear, gap between ends
  mjtNum seg3[6] = {2, 0, 0, 3, 0, 0};
  dist = mj_ipcElemDistance(seg1, 1, seg3, 1, nullptr, nullptr);
  EXPECT_NEAR(dist, 1.0, 1e-12);
}

// segment above triangle interior: perpendicular distance; piercing segment: zero
TEST_F(IpcTest, SegTriDistance) {
  mjtNum tri[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  mjtNum seg_above[6] = {0.2, 0.2, 0.5, 0.3, 0.3, 0.7};
  mjtNum dist = mj_ipcElemDistance(seg_above, 1, tri, 2, nullptr, nullptr);
  EXPECT_NEAR(dist, 0.5, 1e-12);

  // reversed argument order
  dist = mj_ipcElemDistance(tri, 2, seg_above, 1, nullptr, nullptr);
  EXPECT_NEAR(dist, 0.5, 1e-12);

  // piercing
  mjtNum seg_through[6] = {0.2, 0.2, -0.5, 0.2, 0.2, 0.5};
  dist = mj_ipcElemDistance(seg_through, 1, tri, 2, nullptr, nullptr);
  EXPECT_EQ(dist, 0);

  // outside the triangle: nearest feature is the x = 0 edge at distance 0.5
  mjtNum seg_out[6] = {-0.5, 0.5, 0, -0.5, 0.5, 1};
  dist = mj_ipcElemDistance(seg_out, 1, tri, 2, nullptr, nullptr);
  EXPECT_NEAR(dist, 0.5, 1e-12);
}

// parallel offset triangles: distance = offset; piercing triangles: zero
TEST_F(IpcTest, TriTriDistance) {
  mjtNum tri1[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  mjtNum tri2[9] = {0, 0, 0.25, 1, 0, 0.25, 0, 1, 0.25};
  mjtNum dist = mj_ipcElemDistance(tri1, 2, tri2, 2, nullptr, nullptr);
  EXPECT_NEAR(dist, 0.25, 1e-12);

  // piercing: tri2 rotated to cut through tri1
  mjtNum tri3[9] = {0.2, 0.2, -0.5, 0.4, 0.2, 0.5, 0.2, 0.4, 0.5};
  dist = mj_ipcElemDistance(tri1, 2, tri3, 2, nullptr, nullptr);
  EXPECT_EQ(dist, 0);
}

// distance gradient matches finite differences in smooth configurations
TEST_F(IpcTest, DistanceGradient) {
  mjtNum seg[6] = {-1, 0.13, 0.21, 1, -0.11, 0.27};
  mjtNum tri[9] = {-0.3, -0.4, -0.1, 1.1, 0.2, -0.05, -0.2, 1.0, 0.02};
  mjtNum grad[15];
  mjtNum dist = mj_ipcElemDistance(seg, 1, tri, 2, nullptr, grad);
  ASSERT_GT(dist, 0.01);

  mjtNum eps = 1e-7;
  mjtNum all[15];
  mju_copy(all, seg, 6);
  mju_copy(all + 6, tri, 9);
  for (int i = 0; i < 15; i++) {
    mjtNum save = all[i];
    all[i] = save + eps;
    mjtNum dp = mj_ipcElemDistance(all, 1, all + 6, 2, nullptr, nullptr);
    all[i] = save - eps;
    mjtNum dm = mj_ipcElemDistance(all, 1, all + 6, 2, nullptr, nullptr);
    all[i] = save;
    EXPECT_NEAR(grad[i], (dp - dm)/(2*eps), 1e-5) << "component " << i;
  }
}

//---------------------------------- broadphase and feasibility -----------------------------------

// strings 0.05 apart: no pairs with small margin, all aligned pairs with large margin,
// and same-flex adjacent elements are never paired
TEST_F(IpcTest, Broadphase) {
  mjModel* model = LoadModel(kTwoStringsXml);
  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  mjIPCPair pairs[256];

  // axes 0.05 apart, radii 0.01: AABB gap is 0.03; margin 0.01 per flex -> no overlap
  int n = mj_ipcBroadphase(model, data, nullptr, nullptr, 0.01, pairs, 256);
  EXPECT_EQ(n, 0);

  // margin 0.05 per flex: all cross-string pairs overlap; 4x4 elements, but no same-flex
  // pairs (adjacent elements share vertices; non-adjacent are within the same axis line
  // and overlap only if margin bridges their gap along x, which 0.05 does for neighbors
  // of neighbors only if AABBs touch: elements are 0.1 long, gap 0.1 between non-adjacent)
  n = mj_ipcBroadphase(model, data, nullptr, nullptr, 0.05, pairs, 256);
  EXPECT_GE(n, 16);  // at least all cross pairs
  for (int i = 0; i < n && i < 256; i++) {
    if (pairs[i].f1 == pairs[i].f2) {
      // same-flex pairs must not be vertex-sharing neighbors
      EXPECT_GT(abs(pairs[i].e1 - pairs[i].e2), 1);
    }
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

// separated strings are feasible; overlapping strings are not
TEST_F(IpcTest, Feasibility) {
  mjModel* model = LoadModel(kTwoStringsXml);
  mjData* data = mj_makeData(model);
  mj_forward(model, data);
  EXPECT_EQ(mj_ipcFeasible(model, data, nullptr), 1);

  // move string1 axis to 0.045: gap 0.005 < combined radii 0.02 -> violation
  for (int i = 0; i < model->flex_vertnum[0]; i++) {
    int bid = model->flex_vertbodyid[model->flex_vertadr[0] + i];
    for (int j = 0; j < model->body_jntnum[bid]; j++) {
      int jid = model->body_jntadr[bid] + j;
      if (model->jnt_axis[3*jid + 1] > 0.5) {  // the y slide joint
        data->qpos[model->jnt_qposadr[jid]] = 0.045;
      }
    }
  }
  mj_forward(model, data);
  mjIPCPair bad;
  EXPECT_EQ(mj_ipcFeasible(model, data, &bad), 0);
  EXPECT_NE(bad.f1, bad.f2);

  mj_deleteData(data);
  mj_deleteModel(model);
}


//---------------------------------- barrier projection solve -------------------------------------

// thin perpendicular strings, string1 above string2; radii 0.001 (combined thickness 2 mm,
// far below the 10 mm per-step travel at 5 m/s: the classic tunneling configuration)
constexpr char kCrossingStringsXml[] = R"(
<mujoco>
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <flexcomp name="string1" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0.05"/>
    <flexcomp name="string2" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0" quat="0.707107 0 0 0.707107"/>
  </worldbody>
</mujoco>
)";

// same strings far apart (no contact possible within the test horizon)
constexpr char kFarStringsXml[] = R"(
<mujoco>
  <option gravity="0 0 0" timestep="0.002"/>
  <worldbody>
    <flexcomp name="string1" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0.5"/>
    <flexcomp name="string2" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0" quat="0.707107 0 0 0.707107"/>
  </worldbody>
</mujoco>
)";

// bottom string pinned, top string dropped on it under gravity
constexpr char kRestingStringsXml[] = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <flexcomp name="string1" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0.01"/>
    <flexcomp name="string2" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0" quat="0.707107 0 0 0.707107">
      <pin id="0 1 2 3 4"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

// set the velocity of every vertex body of a flex (slide joints created by flexcomp)
static void SetFlexVelocity(const mjModel* m, mjData* d, int f, const mjtNum v[3]) {
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    int bid = m->flex_vertbodyid[m->flex_vertadr[f] + i];
    for (int j = 0; j < m->body_jntnum[bid]; j++) {
      int jid = m->body_jntadr[bid] + j;
      d->qvel[m->jnt_dofadr[jid]] = mju_dot3(v, m->jnt_axis + 3*jid);
    }
  }
}

// z of the center vertex of a flex
static mjtNum CenterZ(const mjModel* m, const mjData* d, int f) {
  int gv = m->flex_vertadr[f] + m->flex_vertnum[f]/2;
  return d->flexvert_xpos[3*gv + 2];
}

// a fast thin string cannot cross another string with the solver enabled, and does without
TEST_F(IpcTest, SolveBlocksCrossing) {
  for (int enabled = 0; enabled < 2; enabled++) {
    mjModel* model = LoadModel(kCrossingStringsXml);
    if (!enabled) {
      model->opt.disableflags |= mjDSBL_FLEXIPC;
    }
    mjData* data = mj_makeData(model);
    int f1 = mj_name2id(model, mjOBJ_FLEX, "string1");
    int f2 = mj_name2id(model, mjOBJ_FLEX, "string2");

    mjtNum v[3] = {0, 0, -5};
    SetFlexVelocity(model, data, f1, v);

    int ordered = 1;
    for (int i = 0; i < 30; i++) {
      mj_step(model, data);
      mj_kinematics(model, data);
      mj_flex(model, data);
      if (CenterZ(model, data, f1) < CenterZ(model, data, f2)) {
        ordered = 0;
      }
      if (enabled) {
        EXPECT_EQ(mj_ipcFeasible(model, data, nullptr), 1) << "step " << i;
      }
    }
    if (enabled) {
      EXPECT_EQ(ordered, 1) << "solver must prevent crossing";
    } else {
      EXPECT_EQ(ordered, 0) << "baseline must tunnel (else the test is too easy)";
    }

    mj_deleteData(data);
    mj_deleteModel(model);
  }
}

// with no flex-flex proximity, the solve must reproduce standard integration exactly
TEST_F(IpcTest, SolveMatchesIntegration) {
  mjModel* model = LoadModel(kFarStringsXml);
  mjData* d1 = mj_makeData(model);
  mjData* d2 = mj_makeData(model);

  mjtNum v[3] = {1, 0.5, -0.5};
  SetFlexVelocity(model, d1, 0, v);
  SetFlexVelocity(model, d2, 0, v);

  for (int i = 0; i < 10; i++) {
    model->opt.disableflags &= ~mjDSBL_FLEXIPC;
    mj_step(model, d1);
    model->opt.disableflags |= mjDSBL_FLEXIPC;
    mj_step(model, d2);
  }
  for (int i = 0; i < model->nq; i++) {
    EXPECT_NEAR(d1->qpos[i], d2->qpos[i], 1e-14) << "qpos " << i;
  }
  for (int i = 0; i < model->nv; i++) {
    EXPECT_NEAR(d1->qvel[i], d2->qvel[i], 1e-12) << "qvel " << i;
  }

  mj_deleteData(d1);
  mj_deleteData(d2);
  mj_deleteModel(model);
}

// pack the vertex positions of one flex element (requires mj_kinematics + mj_flex)
static void PackElem(const mjModel* m, const mjData* d, int f, int e, mjtNum* vert) {
  int dim = m->flex_dim[f];
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim + 1);
  for (int i = 0; i <= dim; i++) {
    mju_copy3(vert + 3*i, d->flexvert_xpos + 3*(m->flex_vertadr[f] + edata[i]));
  }
}

// minimum surface gap (distance minus combined radius) over all element pairs of two flexes
static mjtNum MinElemGap(const mjModel* m, const mjData* d, int f1, int f2) {
  mjtNum xi = m->flex_radius[f1] + m->flex_radius[f2];
  mjtNum mingap = mjMAXVAL;
  for (int e1 = 0; e1 < m->flex_elemnum[f1]; e1++) {
    for (int e2 = 0; e2 < m->flex_elemnum[f2]; e2++) {
      mjtNum vert1[9], vert2[9];
      PackElem(m, d, f1, e1, vert1);
      PackElem(m, d, f2, e2, vert2);
      mjtNum dist = mj_ipcElemDistance(vert1, m->flex_dim[f1], vert2, m->flex_dim[f2],
                                       nullptr, nullptr);
      mingap = mju_min(mingap, dist - xi);
    }
  }
  return mingap;
}

// mean vertex velocity of a flex along one axis (slide joints created by flexcomp)
static mjtNum MeanVelAxis(const mjModel* m, const mjData* d, int f, int axis) {
  mjtNum sum = 0;
  int n = 0;
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    int bid = m->flex_vertbodyid[m->flex_vertadr[f] + i];
    for (int j = 0; j < m->body_jntnum[bid]; j++) {
      int jid = m->body_jntadr[bid] + j;
      if (m->jnt_axis[3*jid + axis] == 1) {
        sum += d->qvel[m->jnt_dofadr[jid]];
        n++;
      }
    }
  }
  return n ? sum/n : 0;
}

// a string dropped onto a pinned string under gravity rests on it without crossing
TEST_F(IpcTest, SolveResting) {
  mjModel* model = LoadModel(kRestingStringsXml);
  mjData* data = mj_makeData(model);
  int f1 = mj_name2id(model, mjOBJ_FLEX, "string1");
  int f2 = mj_name2id(model, mjOBJ_FLEX, "string2");

  for (int i = 0; i < 300; i++) {
    mj_step(model, data);
  }
  mj_kinematics(model, data);
  mj_flex(model, data);

  EXPECT_EQ(mj_ipcFeasible(model, data, nullptr), 1);
  EXPECT_GT(CenterZ(model, data, f1), CenterZ(model, data, f2));
  EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 0);

  mj_deleteData(data);
  mj_deleteModel(model);
}

// resting contact must settle touching, not hovering: the equilibrium gap where the
// barrier balances gravity lies inside the activation band, a small fraction of the
// combined thickness (a wide band reads as objects floating at a visible distance)
TEST_F(IpcTest, SolveRestingGap) {
  mjModel* model = LoadModel(kRestingStringsXml);
  mjData* data = mj_makeData(model);
  int f1 = mj_name2id(model, mjOBJ_FLEX, "string1");
  int f2 = mj_name2id(model, mjOBJ_FLEX, "string2");
  mjtNum xi = model->flex_radius[f1] + model->flex_radius[f2];

  for (int i = 0; i < 300; i++) {
    mj_step(model, data);
  }
  mj_kinematics(model, data);
  mj_flex(model, data);

  mjtNum gap = MinElemGap(model, data, f1, f2);
  EXPECT_GT(gap, 0) << "resting state must be feasible";
  EXPECT_LT(gap, 0.5*xi) << "resting state must touch, not hover";

  mj_deleteData(data);
  mj_deleteModel(model);
}

// slider string resting on a parallel pinned string under gravity, sliding along its
// own axis: contact normals are purely radial, so the frictionless barrier must not
// consume tangential momentum; any freeze-type solver regression (a stalled line search
// or CCD filter zeroing the velocity writeback) manifests as loss of sliding velocity.
// note: the strings must be parallel; a perpendicular crossing drapes into a V whose
// tilted contact normals resist sliding physically
constexpr char kParallelStringsXml[] = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <flexcomp name="slider" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0.004"/>
    <flexcomp name="anchor" type="grid" dim="1" count="5 1 1" spacing="0.05 1 1"
              radius="0.001" mass="0.01" pos="0 0 0">
      <pin id="0 1 2 3 4"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

TEST_F(IpcTest, SolveKeepsTangentialMomentum) {
  mjModel* model = LoadModel(kParallelStringsXml);
  mjData* data = mj_makeData(model);
  int f1 = mj_name2id(model, mjOBJ_FLEX, "slider");
  int f2 = mj_name2id(model, mjOBJ_FLEX, "anchor");

  // slide along the string axis while resting on the anchor
  mjtNum v[3] = {1, 0, 0};
  SetFlexVelocity(model, data, f1, v);

  // 50 steps: long enough to settle and slide 100 mm, short enough that the slider
  // stays mostly on top of the 200 mm anchor (overhanging ends drape and couple)
  for (int i = 0; i < 50; i++) {
    mj_step(model, data);
    EXPECT_EQ(mj_ipcFeasible(model, data, nullptr), 1) << "step " << i;
  }
  mj_kinematics(model, data);
  mj_flex(model, data);

  EXPECT_GT(CenterZ(model, data, f1), CenterZ(model, data, f2));
  EXPECT_NEAR(MeanVelAxis(model, data, f1, 0), 1.0, 0.02)
      << "frictionless contact must not consume tangential momentum";

  mj_deleteData(data);
  mj_deleteModel(model);
}


//---------------------------------- variational elastic parity -----------------------------------

// hanging cloth with bending stiffness under gravity, implicitfast: the flag-on path
// replaces the engine's post-hoc CG (flexInterp_cgsolve) with the variational Newton;
// on a contact-free scene the two must produce the same trajectory
constexpr char kBendingClothXml[] = R"(
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <flexcomp name="cloth" type="grid" dim="2" count="6 6 1" spacing="0.05 0.05 1"
              radius="0.002" mass="0.1" pos="0 0 0.5">
      <contact selfcollide="none"/>
      <elasticity young="50" poisson="0" thickness="0.002" elastic2d="bend"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

// the variational Newton integrates stiff bending stably under implicitfast
// (the engine CG it replaced was validated against this path before deletion: max
// trajectory divergence < 1e-8 over 200 steps at the replacement commit)
TEST_F(IpcTest, SolveElasticStable) {
  mjModel* model = LoadModel(kBendingClothXml);
  ASSERT_GE(model->flex_bendingadr[0], 0) << "model must have bending stiffness";
  mjData* data = mj_makeData(model);

  for (int i = 0; i < 200; i++) {
    mj_step(model, data);
  }
  EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 0);

  // bounded and moving: free fall with bending oscillations, no blowup
  mjtNum maxq = 0;
  for (int j = 0; j < model->nq; j++) {
    maxq = mju_max(maxq, mju_abs(data->qpos[j]));
  }
  EXPECT_GT(maxq, 0.01);
  EXPECT_LT(maxq, 10.0);

  mj_deleteData(data);
  mj_deleteModel(model);
}

// overlapping (infeasible) strings at start fall back to legacy soft contacts, which
// separate them; the barrier solve takes over once feasible
TEST_F(IpcTest, InfeasibleStartHandoff) {
  mjModel* model = LoadModel(kTwoStringsXml);
  mjData* data = mj_makeData(model);

  // overlap: move string1 to 0.045 (gap 0.005 < combined radii 0.02)
  for (int i = 0; i < model->flex_vertnum[0]; i++) {
    int bid = model->flex_vertbodyid[model->flex_vertadr[0] + i];
    for (int j = 0; j < model->body_jntnum[bid]; j++) {
      int jid = model->body_jntadr[bid] + j;
      if (model->jnt_axis[3*jid + 1] > 0.5) {
        data->qpos[model->jnt_qposadr[jid]] = 0.045;
      }
    }
  }
  mj_forward(model, data);
  ASSERT_EQ(mj_ipcFeasible(model, data, nullptr), 0);

  // legacy soft contacts push the strings apart; no crash, eventually feasible
  for (int i = 0; i < 200; i++) {
    mj_step(model, data);
  }
  mj_kinematics(model, data);
  mj_flex(model, data);
  EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 0);
  EXPECT_EQ(mj_ipcFeasible(model, data, nullptr), 1);

  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mujoco
