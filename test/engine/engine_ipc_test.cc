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

// Tests for the IPC variational integrator, engine/engine_ipc.c.
//
// Most tests exercise the geometry/barrier kernels directly (no stepping). The two behavioral tests
// take a single mj_step on a 3x3 cloth: one checks free fall, one checks the intersection-free
// guarantee (a fast cloth cannot tunnel a plane in one step).

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

static mjModel* Load(const char* xml) {
  char error[1024];
  mjModel* model = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(model, NotNull()) << error;
  return model;
}

// id of the first geom in the model
static int FirstGeom(const mjModel* m) { return 0; }

//---------------------------------- element distances --------------------------------------------

// point-triangle distance: interior (perpendicular), edge region, vertex region
TEST_F(IpcTest, PointTriangleDistance) {
  mjtNum a[3] = {0, 0, 0}, b[3] = {1, 0, 0}, c[3] = {0, 1, 0};

  mjtNum p_above[3] = {0.2, 0.2, 0.5};        // over the interior
  EXPECT_NEAR(mj_ipcPtTri(p_above, a, b, c), 0.5, 1e-12);

  mjtNum p_edge[3] = {-1, 0.5, 0};            // nearest the x=0 edge
  EXPECT_NEAR(mj_ipcPtTri(p_edge, a, b, c), 1.0, 1e-12);

  mjtNum p_vert[3] = {-3, -4, 0};             // nearest vertex a
  EXPECT_NEAR(mj_ipcPtTri(p_vert, a, b, c), 5.0, 1e-12);

  mjtNum p_on[3] = {0.25, 0.25, 0};           // on the triangle
  EXPECT_NEAR(mj_ipcPtTri(p_on, a, b, c), 0.0, 1e-12);
}

// segment-segment distance: perpendicular crossing, collinear gap, parallel offset
TEST_F(IpcTest, SegmentSegmentDistance) {
  mjtNum p1[3] = {-1, 0, 0}, p2[3] = {1, 0, 0};

  mjtNum q1[3] = {0, -1, 0.3}, q2[3] = {0, 1, 0.3};       // perpendicular, 0.3 above
  EXPECT_NEAR(mj_ipcSegSeg(p1, p2, q1, q2), 0.3, 1e-12);

  mjtNum r1[3] = {2, 0, 0}, r2[3] = {3, 0, 0};            // collinear, gap 1
  EXPECT_NEAR(mj_ipcSegSeg(p1, p2, r1, r2), 1.0, 1e-12);

  mjtNum s1[3] = {-1, 0, 0.5}, s2[3] = {1, 0, 0.5};       // parallel, 0.5 above
  EXPECT_NEAR(mj_ipcSegSeg(p1, p2, s1, s2), 0.5, 1e-12);
}

//---------------------------------- geom distance ------------------------------------------------

constexpr char kPrimitivesXml[] = R"(
<mujoco>
  <worldbody>
    <geom name="box" type="box" size="0.1 0.2 0.3" pos="0 0 0"/>
    <geom name="sphere" type="sphere" size="0.1" pos="1 0 0"/>
    <geom name="plane" type="plane" size="0 0 1" pos="0 0 -1"/>
  </worldbody>
</mujoco>
)";

TEST_F(IpcTest, GeomDistance) {
  mjModel* m = Load(kPrimitivesXml);
  mjData* d = mj_makeData(m);
  mj_forward(m, d);
  int box = mj_name2id(m, mjOBJ_GEOM, "box");
  int sphere = mj_name2id(m, mjOBJ_GEOM, "sphere");
  int plane = mj_name2id(m, mjOBJ_GEOM, "plane");
  mjtNum n[3];

  // box (half-extent 0.1 in x): point on +x at 0.5 -> surface distance 0.4, normal +x
  mjtNum px[3] = {0.5, 0, 0};
  EXPECT_NEAR(mj_ipcGeomDist(m, d, box, px, n), 0.4, 1e-9);
  EXPECT_NEAR(n[0], 1, 1e-9); EXPECT_NEAR(n[1], 0, 1e-9); EXPECT_NEAR(n[2], 0, 1e-9);

  // interior point -> negative signed distance
  mjtNum pc[3] = {0, 0, 0};
  EXPECT_LT(mj_ipcGeomDist(m, d, box, pc, n), 0);

  // sphere radius 0.1 at (1,0,0): point at (1.3,0,0) -> 0.2, normal +x
  mjtNum ps[3] = {1.3, 0, 0};
  EXPECT_NEAR(mj_ipcGeomDist(m, d, sphere, ps, n), 0.2, 1e-9);
  EXPECT_NEAR(n[0], 1, 1e-9);

  // plane at z=-1: point at z=0 -> 1.0, normal +z
  mjtNum pp[3] = {0.3, -0.2, 0};
  EXPECT_NEAR(mj_ipcGeomDist(m, d, plane, pp, n), 1.0, 1e-9);
  EXPECT_NEAR(n[2], 1, 1e-9);

  mj_deleteData(d);
  mj_deleteModel(m);
}

//---------------------------------- geom sharp features ------------------------------------------

// a box exposes its 8 corners (at +/-size) and 12 edges
TEST_F(IpcTest, BoxFeatures) {
  constexpr char xml[] = R"(
  <mujoco><worldbody>
    <geom type="box" size="0.1 0.2 0.3"/>
  </worldbody></mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  mj_forward(m, d);

  mjtNum verts[8*3], edges[12*6];
  int nv = mj_ipcGeomVerts(m, d, FirstGeom(m), verts);
  int ne = mj_ipcGeomEdges(m, d, FirstGeom(m), edges);
  EXPECT_EQ(nv, 8);
  EXPECT_EQ(ne, 12);
  for (int i = 0; i < nv; i++) {
    EXPECT_NEAR(std::fabs(verts[3*i + 0]), 0.1, 1e-9);
    EXPECT_NEAR(std::fabs(verts[3*i + 1]), 0.2, 1e-9);
    EXPECT_NEAR(std::fabs(verts[3*i + 2]), 0.3, 1e-9);
  }
  // every box edge has unit length along exactly one axis (here 0.2, 0.4, or 0.6)
  for (int i = 0; i < ne; i++) {
    mjtNum dx = edges[6*i+3] - edges[6*i+0];
    mjtNum dy = edges[6*i+4] - edges[6*i+1];
    mjtNum dz = edges[6*i+5] - edges[6*i+2];
    mjtNum len = std::sqrt(dx*dx + dy*dy + dz*dz);
    EXPECT_TRUE(std::fabs(len-0.2) < 1e-9 || std::fabs(len-0.4) < 1e-9 || std::fabs(len-0.6) < 1e-9)
        << "edge " << i << " length " << len;
  }
  mj_deleteData(d);
  mj_deleteModel(m);
}

// a convex mesh exposes its vertices and its (deduplicated) hull edges; a tetrahedron has 4 and 6
TEST_F(IpcTest, MeshFeatures) {
  constexpr char xml[] = R"(
  <mujoco>
    <asset><mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/></asset>
    <worldbody><geom type="mesh" mesh="tet"/></worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  mj_forward(m, d);

  mjtNum verts[64*3], edges[256*6];
  int nv = mj_ipcGeomVerts(m, d, FirstGeom(m), verts);
  int ne = mj_ipcGeomEdges(m, d, FirstGeom(m), edges);
  EXPECT_EQ(nv, 4);   // tetrahedron vertices
  EXPECT_EQ(ne, 6);   // tetrahedron edges (each shared hull edge emitted once)
  mj_deleteData(d);
  mj_deleteModel(m);
}

//---------------------------------- integrator behavior ------------------------------------------

// a single 2D cloth, no contact: one step is free fall (qvel gains -g*dt on every free vertex)
constexpr char kClothXml[] = R"(
<mujoco>
  <option timestep="0.002" integrator="ipc"/>
  <worldbody>
    <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
              radius="0.005" mass="0.05" pos="0 0 0.5"/>
  </worldbody>
</mujoco>
)";

TEST_F(IpcTest, FreeFall) {
  mjModel* m = Load(kClothXml);
  mjData* d = mj_makeData(m);
  int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
  ASSERT_GE(f, 0);

  mj_step(m, d);
  EXPECT_NEAR(d->time, m->opt.timestep, 1e-12);

  // every free vertex slide-joint along z should hold v = g_z * dt after one step
  mjtNum want = m->opt.gravity[2] * m->opt.timestep;
  int checked = 0;
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    int bid = m->flex_vertbodyid[m->flex_vertadr[f] + i];
    for (int j = 0; j < m->body_jntnum[bid]; j++) {
      int jid = m->body_jntadr[bid] + j;
      if (m->jnt_type[jid] == mjJNT_SLIDE && m->jnt_axis[3*jid + 2] > 0.5) {
        EXPECT_NEAR(d->qvel[m->jnt_dofadr[jid]], want, 1e-6) << "vertex " << i;
        checked++;
      }
    }
  }
  EXPECT_GT(checked, 0);   // the model really did expose free z slide joints
  EXPECT_FALSE(std::isnan(d->qpos[0]));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// the intersection-free guarantee: a cloth driven hard at a plane cannot pass through in one step
// (a single explicit Euler step at this speed would put it far below the plane).
TEST_F(IpcTest, ContactBlocksTunneling) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="ipc"/>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1" pos="0 0 0"/>
      <flexcomp name="cloth" type="grid" dim="2" count="3 3 1" spacing="0.05 0.05 1"
                radius="0.005" mass="0.05" pos="0 0 0.05"/>
    </worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  int f = mj_name2id(m, mjOBJ_FLEX, "cloth");
  ASSERT_GE(f, 0);

  // drive every vertex straight down at 50 m/s (0.1 m of travel per 2 ms step, far past the plane)
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    int bid = m->flex_vertbodyid[m->flex_vertadr[f] + i];
    for (int j = 0; j < m->body_jntnum[bid]; j++) {
      int jid = m->body_jntadr[bid] + j;
      if (m->jnt_type[jid] == mjJNT_SLIDE && m->jnt_axis[3*jid + 2] > 0.5) {
        d->qvel[m->jnt_dofadr[jid]] = -50;
      }
    }
  }

  mj_step(m, d);
  mj_kinematics(m, d);
  mj_flex(m, d);

  // no vertex crossed the plane (center stays above z=0; the radius keeps the surface above that)
  mjtNum minz = 1e30;
  for (int i = 0; i < m->flex_vertnum[f]; i++) {
    mjtNum z = d->flexvert_xpos[3*(m->flex_vertadr[f] + i) + 2];
    if (z < minz) minz = z;
  }
  EXPECT_GT(minz, 0) << "cloth tunneled through the plane";
  EXPECT_FALSE(std::isnan(minz));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// RUNG 0 of the all-joint rebuild: the per-tree generalized-coordinate core (mj_ipcTree). With NO contact the
// smooth forces enter explicitly through the predictor (qacc_smooth), so the variational step reduces to
// semi-implicit Euler EXACTLY -- the only things under test are the predictor, the configuration manifold
// (mj_integratePos/differentiatePos exp/log for quaternion DOFs), and the coupled per-tree mass (mj_mulM /
// mj_solveM). A free, a freely-spinning (quaternion + Coriolis), and an articulated (coupled M(q)) body under
// integrator="ipc" must therefore match integrator="Euler" to machine precision.
TEST_F(IpcTest, RigidNoContactMatchesEuler) {
  struct Scene { const char* name; const char* xml; };
  const Scene scenes[] = {
    {"free_ballistic", R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody><body pos="0 0 1"><freejoint/><geom type="box" size="0.1 0.15 0.2" mass="1"/></body></worldbody>
        <keyframe><key qvel="1 0.5 -0.3 0 0 0"/></keyframe></mujoco>)"},
    {"free_spin", R"(
      <mujoco><option timestep="0.002" gravity="0 0 0"/>
        <worldbody><body pos="0 0 1"><freejoint/><geom type="box" size="0.1 0.2 0.3" mass="1"/></body></worldbody>
        <keyframe><key qvel="0 0 0 2 1.3 0.7"/></keyframe></mujoco>)"},
    {"hinge_chain", R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81"/>
        <worldbody>
          <body pos="0 0 1"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
            <body pos="0.3 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/>
              <body pos="0.3 0 0"><joint type="hinge" axis="0 1 0"/><geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="1"/></body>
            </body></body>
        </worldbody></mujoco>)"},
  };
  const int N = 500;
  for (const Scene& s : scenes) {
    mjModel* mi = Load(s.xml); mi->opt.integrator = mjINT_IPC;
    mjModel* me = Load(s.xml); me->opt.integrator = mjINT_EULER;
    mjData* di = mj_makeData(mi);
    mjData* de = mj_makeData(me);
    if (mi->nkey > 0) { mj_resetDataKeyframe(mi, di, 0); mj_resetDataKeyframe(me, de, 0); }
    for (int k = 0; k < N; k++) { mj_step(mi, di); mj_step(me, de); }
    // manifold qpos error (mj_differentiatePos -- never componentwise quaternion subtraction) + qvel error
    mjtNum dq[64];
    mj_differentiatePos(mi, dq, 1.0, de->qpos, di->qpos);
    mjtNum eq = 0, ev = 0;
    for (int i = 0; i < mi->nv; i++) {
      mjtNum a = std::fabs(dq[i]); if (a > eq) eq = a;
      mjtNum b = std::fabs(di->qvel[i] - de->qvel[i]); if (b > ev) ev = b;
    }
    EXPECT_LT(eq, 1e-9) << s.name << ": qpos (manifold) IPC vs Euler should match to round-off";
    EXPECT_LT(ev, 1e-9) << s.name << ": qvel IPC vs Euler should match to round-off";
    mj_deleteData(di); mj_deleteModel(mi);
    mj_deleteData(de); mj_deleteModel(me);
  }
}

// CONTACT rung: the SIMPLE one-sided penalty 0.5*k*min(0,gap)^2 (k = mass/h^2) assembled per-tree into the Newton
// (gap evaluated LIVE -> no tunnel). A sphere rests on the floor at the penalty's analytic penetration radius-mg/k;
// spheres stack via the sphere-sphere DOF coupling; both settle with no blow-up (implicit Euler damps the normal
// bounce). The undamped penalty is NOT MuJoCo's solref contact -- damping + the MuJoCo-equivalent form are next.
TEST_F(IpcTest, RigidContactRestsAndStacks) {
  // single sphere drops onto the floor -> rests at radius - mg/k, no tunnel, settles
  {
    mjModel* m = Load(R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81" integrator="ipc"/>
        <worldbody><geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.4"><freejoint/><geom type="sphere" size="0.1" mass="1" condim="1"/></body>
        </worldbody></mujoco>)");
    mjData* d = mj_makeData(m);
    for (int s = 0; s < 600; s++) mj_step(m, d);
    mjtNum z = d->geom_xpos[3*m->body_geomadr[1] + 2];
    mjtNum mgk = 9.81 * 0.002 * 0.002;                        // mg/k with k = mass/h^2
    EXPECT_NEAR(z, 0.1 - mgk, 1e-4) << "sphere should rest at radius - mg/k";
    EXPECT_GT(z, 0.05) << "sphere tunneled through the floor";
    EXPECT_LT(std::abs(d->qvel[2]), 1e-3) << "sphere should settle";
    EXPECT_FALSE(std::isnan(d->qvel[2]));
    mj_deleteData(d); mj_deleteModel(m);
  }
  // two spheres stack -> bottom on floor, top one diameter above (sphere-sphere Hessian coupling)
  {
    mjModel* m = Load(R"(
      <mujoco><option timestep="0.002" gravity="0 0 -9.81" integrator="ipc"/>
        <worldbody><geom type="plane" size="3 3 0.1"/>
          <body pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.1" mass="1" condim="1"/></body>
          <body pos="0 0 0.33"><freejoint/><geom type="sphere" size="0.1" mass="1" condim="1"/></body>
        </worldbody></mujoco>)");
    mjData* d = mj_makeData(m);
    for (int s = 0; s < 800; s++) mj_step(m, d);
    mjtNum zb = d->geom_xpos[3*m->body_geomadr[1] + 2];
    mjtNum zt = d->geom_xpos[3*m->body_geomadr[2] + 2];
    EXPECT_NEAR(zb, 0.0999, 3e-3) << "bottom sphere should rest on the floor";
    EXPECT_NEAR(zt, 0.2999, 3e-3) << "top sphere should stack one diameter up";
    EXPECT_FALSE(std::isnan(zt));
    mj_deleteData(d); mj_deleteModel(m);
  }
}

// GENERAL ARTICULATED contact: the penalty is driven by MuJoCo's own collision (mj_collision -- mj_step disables
// contact for the IPC predictor) and assembled through the FULL kinematic chain via mj_jac (b = J_B^T n - J_A^T n),
// with the gap LIVE through nonlinear FK. A capsule on a HINGE (its contact Jacobian runs through the joint, not a
// translation DOF) swings onto the floor and must NOT tunnel through it -- this is the humanoid-foot case in
// miniature (capsule geom + articulated body), which the old sphere-only / "first 3 linear DOFs" path missed.
TEST_F(IpcTest, ArticulatedCapsuleDoesNotTunnel) {
  mjModel* m = Load(R"(
    <mujoco><option timestep="0.002" gravity="0 0 -9.81" integrator="ipc"/>
      <worldbody><geom type="plane" size="3 3 0.1"/>
        <body pos="0 0 0.3"><joint type="hinge" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.05" mass="1"/>
        </body>
      </worldbody></mujoco>)");
  mjData* d = mj_makeData(m);
  int gcap = m->body_geomadr[1];
  mjtNum worst = 1e9;                                          // lowest capsule SURFACE point over the run
  for (int s = 0; s < 700; s++) {
    mj_step(m, d);
    // capsule long axis = local z = xmat column 2; lowest surface z = center_z - |halflen*axis_z| - radius
    mjtNum cz = d->geom_xpos[3*gcap + 2], axz = d->geom_xmat[9*gcap + 8];
    mjtNum lowest = cz - std::abs(m->geom_size[3*gcap + 1]*axz) - m->geom_size[3*gcap];
    if (lowest < worst) worst = lowest;
    ASSERT_FALSE(std::isnan(d->qvel[0])) << "NaN at step " << s;
  }
  EXPECT_GT(worst, -0.03) << "capsule tunneled through the floor (lowest surface z = " << worst << ")";
  // it comes to rest propped on the floor (hinge can't fall through): the bob is supported, qvel settles
  EXPECT_LT(std::abs(d->qvel[0]), 5e-2) << "hinge should settle against the floor";
  mj_deleteData(d); mj_deleteModel(m);
}

}  // namespace
}  // namespace mujoco
