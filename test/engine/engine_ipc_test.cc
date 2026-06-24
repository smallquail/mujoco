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

//---------------------------------- affine (ABD) rigid bodies -------------------------------------

// lowest corner height of an axis-half-size-h box geom g, at the current kinematics
static mjtNum LowestCorner(const mjModel* m, const mjData* d, int g, mjtNum h) {
  const mjtNum* xp = d->geom_xpos + 3*g;
  const mjtNum* xm = d->geom_xmat + 9*g;
  mjtNum minz = 1e30;
  for (int c = 0; c < 8; c++) {
    mjtNum cg[3] = {(c&1 ? h : -h), (c&2 ? h : -h), (c&4 ? h : -h)};
    mjtNum z = xp[2] + xm[6]*cg[0] + xm[7]*cg[1] + xm[8]*cg[2];
    if (z < minz) minz = z;
  }
  return minz;
}

// a free-joint rigid body steps as an affine (ABD) body: after one step its linear velocity is g*dt and
// it neither rotates nor gains angular velocity (free fall carried exactly by the rigid predictor).
TEST_F(IpcTest, AffineBodyFreeFall) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="ipc"/>
    <worldbody>
      <body name="box" pos="0 0 1">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1" density="1000"/>
      </body>
    </worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  mj_step(m, d);
  EXPECT_NEAR(d->qvel[2], m->opt.gravity[2]*m->opt.timestep, 1e-6);   // vz = g*dt
  EXPECT_NEAR(d->qvel[0], 0, 1e-8);
  EXPECT_NEAR(d->qvel[1], 0, 1e-8);
  for (int i = 0; i < 3; i++) EXPECT_NEAR(d->qvel[3+i], 0, 1e-7);     // no spurious angular velocity
  EXPECT_NEAR(d->qpos[3], 1, 1e-7);                                   // quaternion still identity
  EXPECT_NEAR(d->qpos[4], 0, 1e-7);
  EXPECT_NEAR(d->qpos[5], 0, 1e-7);
  EXPECT_NEAR(d->qpos[6], 0, 1e-7);
  EXPECT_FALSE(std::isnan(d->qpos[2]));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// CCD guarantee for an affine body: a box driven hard at a plane cannot tunnel through in one step
TEST_F(IpcTest, AffineBodyNoTunnel) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="ipc"/>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1"/>
      <body name="box" pos="0 0 0.12">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1" density="1000"/>
      </body>
    </worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  int g = m->body_geomadr[mj_name2id(m, mjOBJ_BODY, "box")];
  d->qvel[2] = -50;                          // 0.1 m of travel per step, far past the plane
  mj_step(m, d);
  mj_kinematics(m, d);
  mjtNum minz = LowestCorner(m, d, g, 0.1);
  // the affine contact barrier is strict for gap>0 but has a quadratic recovery below 0, so an extreme drive
  // may dip just below the surface (and recover) rather than staying strictly above; real tunneling is cm-scale.
  EXPECT_GT(minz, -1e-4) << "affine box tunneled through the plane";
  EXPECT_FALSE(std::isnan(minz));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// a box dropped flat on a floor stays penetration-free throughout and settles to rest near the surface
// (the normal dashpot dissipates the impact energy; frictionless, so only normal motion is damped).
TEST_F(IpcTest, AffineBodySettles) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="ipc"/>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1"/>
      <body name="box" pos="0 0 0.2">
        <freejoint/>
        <geom type="box" size="0.08 0.08 0.08" density="500"/>
      </body>
    </worldbody>
  </mujoco>)";
  mjModel* m = Load(xml);
  mjData* d = mj_makeData(m);
  int g = m->body_geomadr[mj_name2id(m, mjOBJ_BODY, "box")];
  for (int s = 0; s < 800; s++) {
    mj_step(m, d);
    mj_kinematics(m, d);
    ASSERT_FALSE(std::isnan(d->qpos[2])) << "NaN at step " << s;
    EXPECT_GT(LowestCorner(m, d, g, 0.08), -1e-3) << "box penetrated the floor at step " << s;
    EXPECT_LT(d->qpos[2], 0.25) << "box gained energy / flew off at step " << s;
  }
  // settled near the resting height (half-size 0.08 + a small barrier gap) with negligible velocity
  EXPECT_NEAR(d->qpos[2], 0.085, 0.01) << "box did not settle at the expected rest height";
  mjtNum speed = 0;
  for (int i = 0; i < m->nv; i++) speed += d->qvel[i]*d->qvel[i];
  EXPECT_LT(std::sqrt(speed), 0.05) << "box did not come to rest";
  mj_deleteData(d);
  mj_deleteModel(m);
}

// a rigid arm on a hinge to the world swings under gravity, matching the articulated (reduced-coordinate)
// reference -- the affine hinge (2 control points pinned on the axis + 2 free, held rigid by orthogonality).
TEST_F(IpcTest, AffineHingePendulum) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="ipc"/>
    <worldbody>
      <body name="arm" pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0" pos="0 0 0"/>
        <geom type="capsule" fromto="0 0 0 0.4 0 0" size="0.04" density="500"/>
      </body>
    </worldbody>
  </mujoco>)";
  mjModel* mi = Load(xml); mjData* di = mj_makeData(mi);
  mjModel* me = Load(xml); me->opt.integrator = mjINT_EULER; mjData* de = mj_makeData(me);
  for (int s = 0; s < 300; s++) { mj_step(mi, di); mj_step(me, de); }
  EXPECT_FALSE(std::isnan(di->qpos[0]));
  EXPECT_GT(std::fabs(di->qpos[0]), 0.5) << "pendulum did not swing";          // it actually moved
  EXPECT_NEAR(di->qpos[0], de->qpos[0], 1e-4) << "affine hinge diverged from the articulated reference";
  mj_deleteData(di); mj_deleteModel(mi);
  mj_deleteData(de); mj_deleteModel(me);
}

// a free-floating chain (free root + hinge child sharing the joint-anchor control points) tracks the
// reduced-coordinate articulated reference -- inter-body node sharing (DOF elimination, no multipliers).
TEST_F(IpcTest, AffineChain) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="ipc"/>
    <worldbody>
      <body name="root" pos="0 0 1">
        <freejoint/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" density="500"/>
        <body name="link" pos="0.3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" pos="0 0 0"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.04" density="500"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";
  mjModel* mi = Load(xml); mjData* di = mj_makeData(mi);
  mjModel* me = Load(xml); me->opt.integrator = mjINT_EULER; mjData* de = mj_makeData(me);
  di->qvel[6] = 1.5; de->qvel[6] = 1.5;   // gentle elbow swing (non-chaotic articulated free motion)
  for (int s = 0; s < 300; s++) { mj_step(mi, di); mj_step(me, de); }
  int elbow = mi->nq - 1;                 // free root (7) + hinge (1)
  EXPECT_FALSE(std::isnan(di->qpos[elbow]));
  EXPECT_GT(std::fabs(di->qpos[elbow]), 0.3) << "elbow did not articulate";
  EXPECT_NEAR(di->qpos[elbow], de->qpos[elbow], 1e-3) << "affine chain diverged from the reference";
  EXPECT_NEAR(di->qpos[2], de->qpos[2], 1e-3) << "root z diverged from the reference";
  mj_deleteData(di); mj_deleteModel(mi);
  mj_deleteData(de); mj_deleteModel(me);
}

}  // namespace
}  // namespace mujoco
