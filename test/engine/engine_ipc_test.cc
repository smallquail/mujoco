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

//---------------------------------- barrier ------------------------------------------------------

// C-IPC offset barrier: zero outside the band, positive and increasing toward the surface
TEST_F(IpcTest, BarrierShape) {
  mjtNum ghat = 0.01;

  // inactive at and beyond the support
  EXPECT_EQ(mj_ipcBarrier(ghat, ghat, nullptr, nullptr), 0);
  EXPECT_EQ(mj_ipcBarrier(2*ghat, ghat, nullptr, nullptr), 0);
  EXPECT_EQ(mj_ipcBarrier(-0.001, ghat, nullptr, nullptr), 0);

  // positive inside, larger closer to contact (g -> 0)
  mjtNum b_far  = mj_ipcBarrier(0.8*ghat, ghat, nullptr, nullptr);
  mjtNum b_mid  = mj_ipcBarrier(0.3*ghat, ghat, nullptr, nullptr);
  mjtNum b_near = mj_ipcBarrier(0.01*ghat, ghat, nullptr, nullptr);
  EXPECT_GT(b_far, 0);
  EXPECT_GT(b_mid, b_far);
  EXPECT_GT(b_near, b_mid);
}

// barrier 1st/2nd derivatives match finite differences
TEST_F(IpcTest, BarrierDerivatives) {
  mjtNum ghat = 0.01, eps = 1e-7;
  for (mjtNum frac : {0.2, 0.5, 0.8, 0.95}) {
    mjtNum g = frac*ghat, d1, d2;
    mj_ipcBarrier(g, ghat, &d1, &d2);
    mjtNum bp = mj_ipcBarrier(g + eps, ghat, nullptr, nullptr);
    mjtNum bm = mj_ipcBarrier(g - eps, ghat, nullptr, nullptr);
    mjtNum b0 = mj_ipcBarrier(g, ghat, nullptr, nullptr);
    EXPECT_NEAR(d1, (bp - bm)/(2*eps), 1e-4*mju_max(1.0, std::fabs(d1))) << "frac " << frac;
    EXPECT_NEAR(d2, (bp - 2*b0 + bm)/(eps*eps), 1e-2*mju_max(1.0, std::fabs(d2))) << "frac " << frac;
  }
}

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

}  // namespace
}  // namespace mujoco
