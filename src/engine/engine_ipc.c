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

#include "engine/engine_ipc.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>   // qsort (sparse Hessian pattern build)

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_forward.h"      // mj_Euler (fallback)
#include "engine/engine_util_blas.h"    // mju_dot3
#include "engine/engine_util_errmem.h"  // mju_malloc, mju_free

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

// IPC log-barrier on a surface gap g (C-IPC offset) and its 1st/2nd derivatives, for 0 < g < gh
static mjtNum ipc_Bg (mjtNum g, mjtNum gh) { return (g > 0 && g < gh) ? -(g-gh)*(g-gh)*log(g/gh) : 0.0; }
static mjtNum ipc_Bd (mjtNum g, mjtNum gh) { mjtNum u=g-gh; return -2*u*log(g/gh) - u*u/g; }
static mjtNum ipc_Bdd(mjtNum g, mjtNum gh) { mjtNum u=g-gh; return -2*log(g/gh) - 4*u/g + u*u/(g*g); }

// one active contact. type: 0 vertex-triangle self, 1 edge-edge self, 2 flex-vertex vs geom
// surface, 3 geom-corner vs flex-triangle, 4 geom-edge vs flex-edge. idx/gi meaning per type
// (see ipc_conGap). The geom side is static, so its features (gv/ge) are precomputed once per step.
typedef struct { int type; int idx[4]; int gi; } ipcCon;

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
    return dd - (rad[v] + r);   // point radius + flex (triangle) radius; rad[v]==r for flex self
  }
  case 1: {   // edge-edge self-contact (edge a1b1 against edge a2b2)
    int a1=con->idx[0], b1=con->idx[1], a2=con->idx[2], b2=con->idx[3];
    mjtNum cp1[3], cp2[3], st[2], dd = ipc_segSeg(&x[3*a1], &x[3*b1], &x[3*a2], &x[3*b2], cp1, cp2, st);
    for (int k=0; k < 3; k++) n[k] = (cp1[k]-cp2[k])/dd;
    idv[0]=a1; idv[1]=b1; idv[2]=a2; idv[3]=b2;
    cw[0]=1-st[0]; cw[1]=st[0]; cw[2]=-(1-st[1]); cw[3]=-st[1]; *nidx=4;
    return dd - 2*r;
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
    return dd - r;
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
    return dd - r;
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
  case 0:  return ipc_ptTri(P[0], P[1], P[2], P[3], cp, w) - (rad[con->idx[0]] + r);
  case 1:  return ipc_segSeg(P[0], P[1], P[2], P[3], c1, c2, st) - 2*r;
  case 2:  { mjtNum nn[3];
             return ipc_geomDist(m, con->gi, d->geom_xpos+3*con->gi, d->geom_xmat+9*con->gi, P[0], nn, 1e30) - rad[con->idx[0]]; }
  case 3:  return ipc_ptTri(&gv[3*con->idx[0]], P[0], P[1], P[2], cp, w) - r;
  case 5:  { mjtNum dd=0; for (int k=0; k < 3; k++){ mjtNum t=P[0][k]-P[1][k]; dd+=t*t; }
             return mju_sqrt(dd) - (rad[con->idx[0]]+rad[con->idx[1]]); }
  default: { const mjtNum* eg = &ge[6*con->idx[0]]; return ipc_segSeg(eg, eg+3, P[0], P[1], c1, c2, st) - r; }
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
                      const int* fidx, const ipcCon* cand, int ncand, const mjtNum* cgap) {
  mjtNum alpha = 1.0;
  for (int c=0; c < ncand; c++) {
    const ipcCon* con = &cand[c];
    int v[4], nv, self = (con->type <= 1) && (con->idx[0] < nfv);   // genuine flex-flex self-contact only
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
    mjtNum g0 = cgap[c];   // gap at x, cached by ipc_try this Newton iter (== ipc_conGapAdv at t=0)
    if (g0 <= 0) continue;   // already touching/penetrating: the barrier + energy own it
    if (l <= 0.8*g0) continue;   // full alpha=1 step shrinks gap by <= l, stays above the 20% floor
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
typedef struct { mjtNum n[3], cw[4], bdd; int f[4], nidx; } ipcCC;

// One membrane element (2D triangle): the 3 flex vertices (global vg, free-dof fv, -1 if pinned),
// the FEM stiffness metric M (symmetric 3x3 over the element's 3 edges, read from flex_stiffness),
// and the rest squared edge lengths Lr2. We consume MuJoCo's P1 FEM membrane energy (engine_passive,
// Kharevych et al.): E = 1/4 e^T M e with e[edge] = L^2 - L0^2 (frame-invariant). The force is the
// gradient (matches mj_flexPassiveStretch exactly) and the Hessian is its Gauss-Newton part.
typedef struct { int vg[3], fv[3]; mjtNum M[6]; mjtNum Lr2[3];
                 mjtNum kD;        // Rayleigh damping coefficient = flex_damping/h (constant)
                 mjtNum ep[3];     // elongations e_a at the previous step (xold), refreshed each step
               } ipcElem;
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


// evaluate a candidate contact at x; if active (0<g<ghat) accumulate its barrier gradient and the
// Hessian diagonal (for the Jacobi preconditioner), and cache the GN block (ccache) + contact (acon).
static void ipc_try(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x,
                    const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad, mjtNum ghat, mjtNum kappa,
                    const int* fidx, mjtNum* grad,
                    ipcCon* acon, ipcCC* ccache, int* nacon, int amax, mjtNum* gout) {
  mjtNum n[3], cw[4]; int idv[4], nidx;
  mjtNum g = ipc_conGap(&con, m, d, x, gv, ge, r, rad, n, idv, cw, &nidx, ghat);
  *gout = g;   // cache the gap at x so CCD (g0) and the E0 energy can reuse it without recomputing
  if (g <= 0 || g >= ghat) return;
  if (*nacon >= amax) return;
  mjtNum bd = kappa*ipc_Bd(g, ghat), bdd = kappa*ipc_Bdd(g, ghat);
  ipcCC* cc = &ccache[*nacon];
  cc->bdd = bdd; cc->nidx = nidx;
  for (int k=0; k < 3; k++) cc->n[k] = n[k];
  for (int p=0; p < nidx; p++) {
    int fp = fidx[idv[p]];
    cc->cw[p] = cw[p]; cc->f[p] = fp;
    if (fp < 0) continue;
    for (int i=0; i < 3; i++) grad[3*fp+i] += bd*cw[p]*n[i];   // barrier gradient (force into grad)
  }
  acon[*nacon] = con;                                          // GN Hessian block (bdd, n, cw) cached
  (*nacon)++;
}

// matrix-free Hessian-vector product Hp = H*p in the free-dof space (size N): inertia (mdiag) + the
// full FEM membrane element Hessian + contact Gauss-Newton blocks (ccache). The element Hessian is
// the FULL analytic second derivative of E = 1/4 e^T M e:
//   H = 2 sum_ab M[a,b] g_a g_b^T          (the "Gauss-Newton" / material part, g_a = edge vector)
//     + sum_a Me_a (edge-Laplacian_a (x) I) (the geometric/stress part, Me_a = sum_b M[a,b] e_b)
// applied in factored form per element (no projection): estr caches g_a (9) and Me_a (3) this Newton
// iter. The inertia term mdiag dominates so the total H is normally positive-definite for CG.
static void ipc_applyH(const mjtNum* p, mjtNum* Hp, int N, const mjtNum* mdiag,
                       int nelem, const ipcElem* elems, const mjtNum* estr,
                       const ipcCC* ccache, int nacon) {
  for (int i=0; i < N; i++) Hp[i] = mdiag[i]*p[i];
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
      mjtNum me = es[9+a];                                 // geometric tension (Me_eff, damped)
      int p0 = el->fv[ipc_eedge[a][0]], p1 = el->fv[ipc_eedge[a][1]];
      if (p0 >= 0) for (int k=0; k < 3; k++) Hp[3*p0+k] += ca*es[3*a+k] + me*rel[a][k];
      if (p1 >= 0) for (int k=0; k < 3; k++) Hp[3*p1+k] -= ca*es[3*a+k] + me*rel[a][k];
    }
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

// sparse symmetric Hessian for the IC(0) preconditioner: lower-tri CSR + column structure + workspace.
// Built/factored by the ipc_sp* helpers below mj_IPC; applied here as the CG preconditioner.
typedef struct {
  int N, nnz;
  int *rownnz, *rowadr, *colind;          // lower-tri CSR pattern
  mjtNum *val, *L;                        // H values, IC(0) factor (same pattern)
  int *crownnz, *crowadr, *crow, *cidx;   // column structure: rows in each col + their CSR value index
  mjtNum *w; int *jw;                     // IC(0) workspace: dense row accumulator + pattern marker
} ipcSparse;
static void ipc_icApply(const ipcSparse* sp, mjtNum* z, const mjtNum* r);   // defined below

// matrix-free preconditioned CG: solve H dx = -grad. The matvec H*p stays matrix-free (ipc_applyH);
// the preconditioner is the IC(0) factor of the assembled sparse H (sp), applied via ipc_icApply.
// Returns the iteration count (for diagnostics).
static int ipc_pcg(mjtNum* dx, const mjtNum* grad, int N, const ipcSparse* sp, const mjtNum* mdiag,
                   int nelem, const ipcElem* elems, const mjtNum* estr,
                   const ipcCC* ccache, int nacon, mjtNum* rcg, mjtNum* zcg, mjtNum* pcg, mjtNum* Hp) {
  mjtNum rz = 0, r0 = 0;
  for (int i=0; i < N; i++) { dx[i] = 0; rcg[i] = -grad[i]; r0 += rcg[i]*rcg[i]; }
  ipc_icApply(sp, zcg, rcg);
  for (int i=0; i < N; i++) { pcg[i] = zcg[i]; rz += rcg[i]*zcg[i]; }
  if (r0 < 1e-30) return 0;
  int it = 0;
  for (; it < 200; it++) {
    ipc_applyH(pcg, Hp, N, mdiag, nelem, elems, estr, ccache, nacon);
    mjtNum pHp = 0; for (int i=0; i < N; i++) pHp += pcg[i]*Hp[i];
    if (pHp <= 1e-30) break;
    mjtNum alpha = rz/pHp;
    mjtNum rr = 0;
    for (int i=0; i < N; i++) { dx[i] += alpha*pcg[i]; rcg[i] -= alpha*Hp[i]; rr += rcg[i]*rcg[i]; }
    if (rr < 1e-8*r0) break;
    ipc_icApply(sp, zcg, rcg);
    mjtNum rznew = 0; for (int i=0; i < N; i++) rznew += rcg[i]*zcg[i];
    mjtNum beta = rznew/rz; rz = rznew;
    for (int i=0; i < N; i++) pcg[i] = zcg[i] + beta*pcg[i];
  }
  return it;
}

// IPC incremental-potential energy: inertia + edge-stretch penalty + contact barriers over the
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

// inertia + stretch energy only (no contact barrier); used to assemble E0 from cached gaps
static mjtNum ipc_energyBase(const mjModel* m, int nfv, int nelem, const ipcElem* elems,
                             const mjtNum* x, const mjtNum* xtil, const int* fidx,
                             const mjtNum* mass, mjtNum h) {
  mjtNum E = 0, ih2 = 1.0/(h*h);
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum mh = mass[v]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  return E + ipc_stretchEnergy(nelem, elems, x);
}

static mjtNum ipc_energy(const mjModel* m, const mjData* d, int nfv, int nelem, const ipcElem* elems,
                         const mjtNum* x, const mjtNum* xtil, const int* fidx, const mjtNum* mass,
                         mjtNum h, mjtNum r, const mjtNum* rad, mjtNum ghat, mjtNum kappa,
                         const mjtNum* gv, const mjtNum* ge, const ipcCon* acon, int nacon) {
  mjtNum E = 0, ih2 = 1.0/(h*h);
  for (int v=0; v < nfv; v++) if (fidx[v] >= 0) {
    mjtNum mh = mass[v]*ih2;
    for (int c=0; c < 3; c++) { mjtNum t = x[3*v+c] - xtil[3*v+c]; E += 0.5*mh*t*t; }
  }
  E += ipc_stretchEnergy(nelem, elems, x);
  for (int c=0; c < nacon; c++) {
    mjtNum n[3], cw[4]; int idv[4], nidx;
    mjtNum g = ipc_conGap(&acon[c], m, d, x, gv, ge, r, rad, n, idv, cw, &nidx, 1e30);
    if (g > 0 && g < ghat) E += kappa*ipc_Bg(g, ghat);
  }
  return E;
}


// append a candidate contact if its gap at x is below the (margin-inflated) detection threshold
static void ipc_addCand(ipcCon con, const mjModel* m, const mjData* d, const mjtNum* x,
                        const mjtNum* gv, const mjtNum* ge, mjtNum r, const mjtNum* rad, mjtNum thresh,
                        ipcCon* cand, int* nc, int candmax) {
  if (*nc >= candmax) return;
  mjtNum n[3], cw[4]; int idv[4], nidx;
  mjtNum g = ipc_conGap(&con, m, d, x, gv, ge, r, rad, n, idv, cw, &nidx, thresh);
  if (g > 0 && g < thresh) cand[(*nc)++] = con;
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
                          mjtNum threshGeom, int nfv, int npt, int ne, const int* el, int en, int ea,
                          const int* fidx, int f, const int* elemedge, int doself, ipcCon* cand, int candmax) {
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
      ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, cand, &nc, candmax);
    }
  }

  // sphere-sphere (type 5): all pairs of rigid points -- nsph is tiny so the O(nsph^2) scan is cheap
  for (int p=nfv; p < npt; p++) for (int q=p+1; q < npt; q++) {
    ipcCon con = {5, {p, q, 0, 0}, -1};
    ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, cand, &nc, candmax);
  }

  // ---- geom-feature and self candidates via the flex element BVH (ipc_bvhBox) ----
  int bvhadr = m->flex_bvhadr[f];
  if (bvhadr < 0 || (ne == 0 && en == 0)) return nc;   // no element BVH -> only flex-vertex-vs-geom
  int bvhnum = m->flex_bvhnum[f];
  int* stk    = (int*) mju_malloc((bvhnum > 0 ? bvhnum : 1)*sizeof(int));
  int* outel  = (int*) mju_malloc((ne > 0 ? ne : 1)*sizeof(int));
  int* stampG = (int*) mju_malloc((en > 0 ? en : 1)*sizeof(int));
  for (int e=0; e < en; e++) stampG[e] = -1;
  int qid = 0;
  // query half >= thresh + r: node AABBs already include one radius, and a candidate is within thresh of
  // the (one- or two-radius) surface gap, so this is a conservative superset -- no near pair is missed.
  mjtNum qhv[3] = {thresh+r, thresh+r, thresh+r};

  // rigid point (sphere) vs flex triangle: each sphere center queries the triangle BVH (reuses the
  // vertex-triangle type 0, with the sphere as the "vertex" -> gap = dd - (rad[sphere]+r), coupling the
  // sphere dof and the triangle's flex dofs). Always on (not gated by self-collide); the sphere is never
  // a member of a flex triangle so no self-skip is needed. Generated BEFORE the (potentially huge) flex
  // geom-feature candidate flood so the few sphere candidates are never crowded out by the candmax cap --
  // that crowd-out previously dropped the sphere's contact mid-fall and let it tunnel through the bag.
  for (int p=nfv; p < npt; p++) {
    mjtNum qh[3] = {thresh+rad[p], thresh+rad[p], thresh+rad[p]};
    int n = ipc_bvhBox(m, d, f, &x[3*p], qh, stk, outel, ne);
    for (int i=0; i < n; i++) { int e = outel[i];
      ipcCon con = {0, {p, el[3*e], el[3*e+1], el[3*e+2]}, -1};
      ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, cand, &nc, candmax);
    }
  }

  // geom-corner vs flex-triangle (type 3): each geom vertex queries the triangle BVH. The geom is STATIC,
  // so this is a one-sided contact -- a pair can only close at the flex's speed. It uses the tighter
  // threshGeom (vs the two-sided sphere/self thresh): the geom features (a 36-piece convex decomposition
  // here -> ~1600 edges, ~1000 corners) otherwise generate hundreds of thousands of far candidates that
  // overflow candmax and crowd out the near, contact-relevant ones (which let the bag sink into the bin).
  mjtNum qhvG[3] = {threshGeom+r, threshGeom+r, threshGeom+r};
  for (int c=0; c < ngv; c++) {
    int n = ipc_bvhBox(m, d, f, &gv[3*c], qhvG, stk, outel, ne);
    for (int i=0; i < n; i++) { int e = outel[i];
      ipcCon con = {3, {c, el[3*e], el[3*e+1], el[3*e+2]}, -1};
      ipc_addCand(con, m, d, x, gv, ge, r, rad, threshGeom, cand, &nc, candmax);
    }
  }
  // geom-edge vs flex-edge (type 4): each geom edge queries the BVH with its (inflated) box, then the
  // overlapping triangles' edges; dedup the shared edges per query via stampG.
  for (int c=0; c < nge; c++) {
    const mjtNum* p0 = &ge[6*c]; const mjtNum* p1 = &ge[6*c+3];
    mjtNum qc[3], qh[3];
    for (int k=0; k < 3; k++) { qc[k] = 0.5*(p0[k]+p1[k]); qh[k] = 0.5*mju_abs(p1[k]-p0[k]) + threshGeom+r; }
    int n = ipc_bvhBox(m, d, f, qc, qh, stk, outel, ne);
    qid++;
    for (int i=0; i < n; i++) { int e = outel[i];
      for (int j=0; j < 3; j++) { int e2 = elemedge[3*e+j];
        if (stampG[e2] == qid) continue; stampG[e2] = qid;
        ipcCon con = {4, {c, m->flex_edge[2*(ea+e2)], m->flex_edge[2*(ea+e2)+1], 0}, -1};
        ipc_addCand(con, m, d, x, gv, ge, r, rad, threshGeom, cand, &nc, candmax);
      }
    }
  }

  if (doself) {
    // vertex-triangle self: each vertex queries the triangle BVH
    for (int v=0; v < nfv; v++) {
      int n = ipc_bvhBox(m, d, f, &x[3*v], qhv, stk, outel, ne);
      for (int i=0; i < n; i++) { int e = outel[i];
        int A = el[3*e], B = el[3*e+1], C = el[3*e+2];
        if (v==A || v==B || v==C) continue;
        ipcCon con = {0, {v, A, B, C}, -1};
        ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, cand, &nc, candmax);
      }
    }
    // edge-edge self: each flex edge queries the BVH, then the overlapping triangles' edges (canonical
    // e2 > e1, non-adjacent); dedup per query via stampG.
    for (int e1=0; e1 < en; e1++) {
      int a1 = m->flex_edge[2*(ea+e1)], b1 = m->flex_edge[2*(ea+e1)+1];
      mjtNum qc[3], qh[3];
      for (int k=0; k < 3; k++) { qc[k]=0.5*(x[3*a1+k]+x[3*b1+k]); qh[k]=0.5*mju_abs(x[3*a1+k]-x[3*b1+k])+thresh+r; }
      int n = ipc_bvhBox(m, d, f, qc, qh, stk, outel, ne);
      qid++;
      for (int i=0; i < n; i++) { int e = outel[i];
        for (int j=0; j < 3; j++) { int e2 = elemedge[3*e+j];
          if (e2 <= e1 || stampG[e2] == qid) continue; stampG[e2] = qid;
          int a2 = m->flex_edge[2*(ea+e2)], b2 = m->flex_edge[2*(ea+e2)+1];
          if (a1==a2 || a1==b2 || b1==a2 || b1==b2) continue;
          ipcCon con = {1, {a1, b1, a2, b2}, -1};
          ipc_addCand(con, m, d, x, gv, ge, r, rad, thresh, cand, &nc, candmax);
        }
      }
    }
  }
  mju_free(stk); mju_free(outel); mju_free(stampG);
  return nc;
}

// ---------------------------------------------------------------------------------------------------
// Sparse symmetric Hessian helpers (struct ipcSparse declared above ipc_pcg). Lower-triangular CSR
// (colind sorted ascending per row, diagonal last) plus a column structure (the transpose, with the
// CSR value index of each entry) so the factorization and back-substitution can walk columns.
static int ipc_cmpll(const void* a, const void* b) {
  long long x = *(const long long*)a, y = *(const long long*)b;
  return x < y ? -1 : (x > y ? 1 : 0);
}

// add coupling between free-dof nodes fi, fj (a full 3x3 block) to the lower-triangular key list
static void ipc_addBlock(long long* key, int* nk, int fi, int fj, int N) {
  if (fi < 0 || fj < 0) return;
  for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
    int I = 3*fi+a, J = 3*fj+b;
    if (J <= I) key[(*nk)++] = (long long)I*N + J;
  }
}

// cache of the sorted-unique mesh+diagonal keys: these never change for a given model, so re-qsorting
// them every step (the dominant cost of the pattern build) is wasted. Rebuilt when (model,N) changes.
// NOTE: single-threaded assumption (one model stepped at a time) -- fine for single-env CPU eval.
static const mjModel* g_spm = NULL;
static int g_spN = -1, g_spNmk = 0;
static long long* g_spMkeys = NULL;

// build the lower-tri CSR pattern + column structure from mesh elements + candidate contacts
static void ipc_spBuild(const mjModel* m, ipcSparse* sp, int N, int ne, const ipcElem* elems,
                        int ncand, const ipcCon* cand, const int* fidx) {
  sp->N = N;
  // (re)build the cached mesh+diagonal sorted-unique keys only when the model/N changes
  if (m != g_spm || N != g_spN) {
    int mcap = N + 81*ne + 1;
    long long* mk = (long long*) mju_malloc(mcap*sizeof(long long));
    int nmk = 0;
    for (int i=0; i < N; i++) mk[nmk++] = (long long)i*N + i;          // diagonal
    for (int t=0; t < ne; t++) for (int i=0; i < 3; i++) for (int j=0; j < 3; j++)
      ipc_addBlock(mk, &nmk, elems[t].fv[i], elems[t].fv[j], N);       // membrane: element verts
    qsort(mk, nmk, sizeof(long long), ipc_cmpll);
    int u = 0; for (int i=0; i < nmk; i++) if (i==0 || mk[i] != mk[i-1]) mk[u++] = mk[i];  // dedup in place
    mju_free(g_spMkeys); g_spMkeys = mk; g_spNmk = u; g_spm = m; g_spN = N;
  }
  // contact keys (variable per step): collect + sort, then merge with the cached mesh keys
  int ccap = 1; for (int c=0; c < ncand; c++) ccap += 144;
  long long* ck = (long long*) mju_malloc(ccap*sizeof(long long));
  int nck = 0;
  for (int c=0; c < ncand; c++) {
    // flex-vs-geom contacts (types 3,4) and point-vs-geom (type 2) couple only flex verts that already
    // share a mesh element (or a single point), so their couplings are already in the mesh pattern -- skip.
    // Types that couple nodes NOT sharing an element add new pattern entries and MUST be included: flex
    // self-contact (0,1), sphere-vs-flex (0, sphere node <-> triangle), and sphere-vs-sphere (5).
    if (cand[c].type >= 2 && cand[c].type != 5) continue;
    int v[4], nv; ipc_conVerts(&cand[c], v, &nv);
    for (int i=0; i < nv; i++) for (int j=0; j < nv; j++) ipc_addBlock(ck, &nck, fidx[v[i]], fidx[v[j]], N);
  }
  qsort(ck, nck, sizeof(long long), ipc_cmpll);
  // merge the two sorted lists into a sorted-unique key list (avoids re-sorting the big mesh list)
  long long* key = (long long*) mju_malloc((g_spNmk + nck + 1)*sizeof(long long));
  int nk = 0, ia = 0, ib = 0; long long prev = -1;
  while (ia < g_spNmk || ib < nck) {
    long long v = (ib >= nck || (ia < g_spNmk && g_spMkeys[ia] <= ck[ib])) ? g_spMkeys[ia++] : ck[ib++];
    if (v != prev) { key[nk++] = v; prev = v; }
  }
  mju_free(ck);
  // dedup -> nnz
  int nnz = 0;
  for (int i=0; i < nk; i++) if (i == 0 || key[i] != key[i-1]) nnz++;
  sp->nnz = nnz;
  sp->rownnz = (int*) mju_malloc(N*sizeof(int));
  sp->rowadr = (int*) mju_malloc(N*sizeof(int));
  sp->colind = (int*) mju_malloc(nnz*sizeof(int));
  sp->val    = (mjtNum*) mju_malloc(nnz*sizeof(mjtNum));
  sp->L      = (mjtNum*) mju_malloc(nnz*sizeof(mjtNum));
  for (int i=0; i < N; i++) sp->rownnz[i] = 0;
  int e = 0;
  for (int i=0; i < nk; i++) {
    if (i > 0 && key[i] == key[i-1]) continue;
    int row = (int)(key[i] / N), col = (int)(key[i] % N);
    sp->colind[e] = col; sp->rownnz[row]++; e++;
  }
  sp->rowadr[0] = 0;
  for (int i=1; i < N; i++) sp->rowadr[i] = sp->rowadr[i-1] + sp->rownnz[i-1];
  // column structure (transpose with CSR value index)
  sp->crownnz = (int*) mju_malloc(N*sizeof(int));
  sp->crowadr = (int*) mju_malloc(N*sizeof(int));
  sp->crow    = (int*) mju_malloc(nnz*sizeof(int));
  sp->cidx    = (int*) mju_malloc(nnz*sizeof(int));
  for (int i=0; i < N; i++) sp->crownnz[i] = 0;
  for (int i=0; i < N; i++) for (int t=0; t < sp->rownnz[i]; t++) sp->crownnz[sp->colind[sp->rowadr[i]+t]]++;
  sp->crowadr[0] = 0;
  for (int i=1; i < N; i++) sp->crowadr[i] = sp->crowadr[i-1] + sp->crownnz[i-1];
  int* fill = sp->jw = (int*) mju_malloc(N*sizeof(int));   // reuse jw as temp fill counter
  for (int i=0; i < N; i++) fill[i] = 0;
  for (int i=0; i < N; i++) for (int t=0; t < sp->rownnz[i]; t++) {
    int idx = sp->rowadr[i]+t, col = sp->colind[idx];
    int pos = sp->crowadr[col] + fill[col]++;
    sp->crow[pos] = i; sp->cidx[pos] = idx;
  }
  sp->w  = (mjtNum*) mju_malloc(N*sizeof(mjtNum));
  for (int i=0; i < N; i++) { sp->w[i] = 0; sp->jw[i] = 0; }
  mju_free(key);
}

static void ipc_spFree(ipcSparse* sp) {
  mju_free(sp->rownnz); mju_free(sp->rowadr); mju_free(sp->colind);
  mju_free(sp->val); mju_free(sp->L);
  mju_free(sp->crownnz); mju_free(sp->crowadr); mju_free(sp->crow); mju_free(sp->cidx);
  mju_free(sp->w); mju_free(sp->jw);
}

// find the CSR value index of lower-tri entry (row, col<=row); -1 if not in pattern (binary search)
static int ipc_spIdx(const ipcSparse* sp, int row, int col) {
  int lo = sp->rowadr[row], hi = lo + sp->rownnz[row] - 1;
  while (lo <= hi) { int mid = (lo+hi)/2; int c = sp->colind[mid];
    if (c == col) return mid; else if (c < col) lo = mid+1; else hi = mid-1; }
  return -1;
}

// incomplete LDL^T factorization (symmetric ILU, no fill): sp->val (lower-tri H) -> unit-lower L in
// the off-diagonals of sp->L + signed diagonal D in the diagonal slot. No square root, so it handles
// indefinite H (negative D is fine). Left-looking with a dense row accumulator; the column structure
// supplies L[m,k]. The diagonal magnitude is floored at `shift` (keeping its sign) to avoid 1/0.
static void ipc_ic0(ipcSparse* sp, mjtNum shift) {
  int N = sp->N;
  const int *rn = sp->rownnz, *ra = sp->rowadr, *ci = sp->colind;
  const int *crn = sp->crownnz, *cra = sp->crowadr, *crow = sp->crow, *cidx = sp->cidx;
  mjtNum* L = sp->L; const mjtNum* H = sp->val; mjtNum* w = sp->w; int* jw = sp->jw;
  for (int i=0; i < N; i++) {
    int ai = ra[i], ni = rn[i];
    for (int t=0; t < ni; t++) { int j = ci[ai+t]; w[j] = H[ai+t]; jw[j] = 1; }   // gather row i
    for (int t=0; t < ni-1; t++) {                                  // off-diagonals k < i, ascending
      int k = ci[ai+t];
      mjtNum Dk = L[ra[k]+rn[k]-1];                                 // D[k]
      mjtNum Lik = w[k] / Dk;                                       // unit-lower L[i,k]
      w[k] = Lik;
      int ck = cra[k], nck = crn[k];                                // walk column k: rows m > k
      for (int s=0; s < nck; s++) {
        int m = crow[ck+s]; if (m <= k) continue;
        if (jw[m]) { mjtNum Lmk = (m == i) ? Lik : L[cidx[ck+s]]; w[m] -= Lik*Dk*Lmk; }
      }
    }
    mjtNum Di = w[i];                                               // signed pivot
    if (Di < shift && Di > -shift) Di = (Di >= 0 ? shift : -shift);
    L[ai+ni-1] = Di;
    for (int t=0; t < ni-1; t++) L[ai+t] = w[ci[ai+t]];             // store unit-L L[i,j], j<i
    for (int t=0; t < ni; t++) { int j = ci[ai+t]; w[j] = 0; jw[j] = 0; }
  }
}

// apply the incomplete-LDL^T preconditioner z = (L D L^T)^-1 r: forward unit-L solve L u = r, scale by
// D^-1, back unit-L^T solve L^T z = D^-1 u. (L diagonal slot stores D; off-diagonals are unit-L.)
static void ipc_icApply(const ipcSparse* sp, mjtNum* z, const mjtNum* r) {
  int N = sp->N;
  const int *rn = sp->rownnz, *ra = sp->rowadr, *ci = sp->colind;
  const int *crn = sp->crownnz, *cra = sp->crowadr, *crow = sp->crow, *cidx = sp->cidx;
  const mjtNum* L = sp->L;
  for (int i=0; i < N; i++) z[i] = r[i];
  for (int i=0; i < N; i++) {                                        // forward: L u = r (unit-lower)
    int ai = ra[i], ni = rn[i];
    mjtNum s = z[i];
    for (int t=0; t < ni-1; t++) s -= L[ai+t]*z[ci[ai+t]];
    z[i] = s;
  }
  for (int i=0; i < N; i++) z[i] /= L[ra[i]+rn[i]-1];                // scale by D^-1
  for (int i=N-1; i >= 0; i--) {                                     // back: L^T z = D^-1 u (unit-upper)
    int ck = cra[i], nck = crn[i];
    mjtNum s = z[i];
    for (int t=0; t < nck; t++) { int m = crow[ck+t]; if (m > i) s -= L[cidx[ck+t]]*z[m]; }
    z[i] = s;
  }
}

// IPC-style variational integrator (PROTOTYPE, phases 0a/0b): a single 2D flex -- inertia + edge-
// stretch penalty + vertex-triangle SELF-CONTACT log-barrier -- by projected-Newton with a step-
// capped (crude-CCD) line search, starting each step from the last collision-free state.
// Intersection-free and freeze-free in validation, but PROTOTYPE quality: dense solve + per-step
// malloc (small flex only), vertex-triangle only (no edge-edge), step-cap is not a rigorous CCD,
// kappa hardcoded. Falls back to Euler if there is no 2D flex.
void mj_IPC(const mjModel* m, mjData* d) {
  mjtNum h = m->opt.timestep, kappa = 100.0;
  int f = -1;
  for (int i=0; i < m->nflex; i++) if (m->flex_dim[i] == 2) { f = i; break; }
  if (f < 0) { mj_Euler(m, d); return; }
  int nfv = m->flex_vertnum[f], va = m->flex_vertadr[f];
  int ea = m->flex_edgeadr[f], en = m->flex_edgenum[f];
  int ne = m->flex_elemnum[f];
  const int* el = m->flex_elem + m->flex_elemdataadr[f];
  mjtNum r = m->flex_radius[f], ghat = r;   // contact activates within ghat of the 2r surface gap

  // free points = bodies with 3 slide DOFs: the flex vertices (elastic) plus any standalone 3-slide body
  // carrying a sphere geom (rigid point). Generic so rigid/ABD bodies can extend this later -- a free point
  // just needs dof/qpos addr, mass, radius, and its body (for the local-slide-frame rotation R); only the
  // xold POSITION source differs (flex vertex vs geom center). Count the spheres first to size the arrays.
  char* isflexvert = (char*) mju_malloc((m->nbody > 0 ? m->nbody : 1)*sizeof(char));
  for (int b=0; b < m->nbody; b++) isflexvert[b] = 0;
  for (int v=0; v < nfv; v++) isflexvert[m->flex_vertbodyid[va+v]] = 1;
  int nsph = 0;
  for (int b=0; b < m->nbody; b++) {
    if (isflexvert[b] || m->body_dofnum[b] != 3 || m->jnt_type[m->body_jntadr[b]] != mjJNT_SLIDE) continue;
    for (int g=m->body_geomadr[b]; g < m->body_geomadr[b]+m->body_geomnum[b]; g++)
      if (m->geom_type[g] == mjGEOM_SPHERE) { nsph++; break; }
  }
  int npt = nfv + nsph;
  int* dofadr = (int*) mju_malloc(npt*sizeof(int));
  int* qpadr  = (int*) mju_malloc(npt*sizeof(int));   // qpos address (NOT dof address: differs after
  int* fidx   = (int*) mju_malloc(npt*sizeof(int));   // free/ball joints, which have more qpos than dof)
  mjtNum* mass = (mjtNum*) mju_malloc(npt*sizeof(mjtNum));
  mjtNum* rad  = (mjtNum*) mju_malloc(npt*sizeof(mjtNum));   // per-point radius (flex_radius / sphere size)
  int* pbody  = (int*) mju_malloc(npt*sizeof(int));          // body id, for the slide-frame rotation R
  int* pgeom  = (int*) mju_malloc(npt*sizeof(int));          // sphere geom id (>=0); -1 for a flex vertex
  int nfree = 0;
  for (int v=0; v < nfv; v++) {
    int bid = m->flex_vertbodyid[va+v];
    dofadr[v] = -1; qpadr[v] = -1; fidx[v] = -1; mass[v] = 0; rad[v] = r; pbody[v] = bid; pgeom[v] = -1;
    if (m->body_dofnum[bid] == 3) {
      int da = m->body_dofadr[bid];
      dofadr[v] = da; qpadr[v] = m->jnt_qposadr[m->body_jntadr[bid]]; fidx[v] = nfree++;
      mass[v] = d->qM[m->M_rowadr[da] + m->M_rownnz[da] - 1];   // diagonal (point mass)
    }
  }
  for (int b=0, p=nfv; b < m->nbody; b++) {                  // append the rigid sphere points
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
  mju_free(isflexvert);
  int N = 3*nfree, Na = (N > 0 ? N : 1);
  // FEM membrane elements: consume MuJoCo's precomputed flex_stiffness metric (from the model's
  // <elasticity young=.../> tag) per element + the rest squared edge lengths. No hand-rolled springs.
  int sa = m->flex_stiffnessadr[f];
  const int* elemedge = m->flex_elemedge + m->flex_elemedgeadr[f];
  ipcElem* elems = (ipcElem*) mju_malloc((ne > 0 ? ne : 1)*sizeof(ipcElem));
  for (int t=0; t < ne; t++) {
    for (int i=0; i < 3; i++) { elems[t].vg[i] = el[3*t+i]; elems[t].fv[i] = fidx[el[3*t+i]]; }
    for (int j=0; j < 6; j++) elems[t].M[j] = (sa >= 0) ? m->flex_stiffness[sa + 21*t + j] : 0;
    for (int a=0; a < 3; a++) {
      mjtNum L0 = m->flexedge_length0[ea + elemedge[3*t+a]]; elems[t].Lr2[a] = L0*L0;
      elems[t].ep[a] = 0;
    }
    elems[t].kD = (h > 0) ? m->flex_damping[f]/h : 0;   // stiffness-prop. Rayleigh damping (Kharevych 5.2)
  }
  int amax = npt*64 + 1024;                   // capacity of the active-contact list
  ipcCon* acon = (ipcCon*) mju_malloc(amax*sizeof(ipcCon));
  ipcCC* ccache = (ipcCC*) mju_malloc(amax*sizeof(ipcCC));
  int candmax = npt*192 + 8192;               // capacity of the per-step candidate list (sized for the
                                              // geom-feature-heavy bag-in-bin contact: ~160k at npt~1100)
  ipcCon* cand = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon));
  mjtNum* cgap = (mjtNum*) mju_malloc(candmax*sizeof(mjtNum));   // per-candidate gap at x (try->ccd/E0)
  ipcCon* candLS = (ipcCon*) mju_malloc(candmax*sizeof(ipcCon)); // line-search subset (can activate this step)
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
  mjtNum* estr = (mjtNum*) mju_malloc((ne > 0 ? 12*ne : 1)*sizeof(mjtNum)); // per-element g_a (9) + Me_a (3)
                                                                           // (full, PSD-projected, this Newton iter)

  const mjtNum* vx = d->flexvert_xpos;
  for (int v=0; v < npt; v++) {
    // xold position source: flex vertex from flexvert_xpos, rigid sphere from its geom center
    if (pgeom[v] < 0) for (int c=0; c < 3; c++) xold[3*v+c] = vx[3*(va+v)+c];
    else              for (int c=0; c < 3; c++) xold[3*v+c] = d->geom_xpos[3*pgeom[v]+c];
    if (fidx[v] >= 0) {
      int da = dofadr[v];
      // qvel/qacc_smooth are in the body's local slide frame; map to WORLD via the body rotation R
      // (= d->xmat). R=I for unrotated bodies (cloth, the balls); non-identity for the rotated bag.
      const mjtNum* R = d->xmat + 9*pbody[v];
      mjtNum vw[3], aw[3];
      mju_mulMatVec3(vw, R, d->qvel + da);
      mju_mulMatVec3(aw, R, d->qacc_smooth + da);
      for (int c=0; c < 3; c++)
        xtil[3*v+c] = xold[3*v+c] + h*vw[c] + h*h*aw[c];
    } else {
      for (int c=0; c < 3; c++) xtil[3*v+c] = xold[3*v+c];   // pinned: fixed
    }
  }
  for (int i=0; i < 3*npt; i++) x[i] = xold[i];   // start from last collision-free state (feasibility)

  mjtNum ih2 = 1.0/(h*h);
  for (int i=0; i < N; i++) mdiag[i] = 0;                          // inertia diagonal (fixed per step)
  for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
    int fi = fidx[v]; for (int c=0; c < 3; c++) mdiag[3*fi+c] = mass[v]*ih2;
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
  int doself = (m->flex_selfcollide[f] != mjFLEXSELF_NONE);   // honor the flex's selfcollide flag
  int ncand = ipc_candidates(m, d, x, gv, ge, ngv, nge, r, rad, thresh, threshGeom, nfv, npt, ne, el, en, ea, fidx,
                             f, elemedge, doself, cand, candmax);
  // warm start: with no candidate contacts within thresh, the predictor x~ is collision-free (the thresh
  // margin covers the step displacement), so it is a far better feasible initial guess than xold and Newton
  // converges in ~1 iteration instead of ~2 -- halving the cost of contact-free steps.
  if (ncand == 0) for (int i=0; i < 3*npt; i++) x[i] = xtil[i];
  // sparse Hessian pattern (mesh + candidate-contact couplings) for the IC(0) preconditioner, once/step
  // Rayleigh damping uses the previous-step elongations e_prev = e(xold); xold is fixed over the Newton
  // loop, so cache e_prev once per step. Only needed when damping is on.
  if (m->flex_damping[f] > 0)
    for (int t=0; t < ne; t++) { mjtNum dtmp[3][3]; ipc_elemEval(&elems[t], xold, dtmp, elems[t].ep); }
  ipcSparse sp;
  if (N > 0) ipc_spBuild(m, &sp, N, ne, elems, ncand, cand, fidx);
  // precompute element -> CSR scatter indices once per step (the pattern is fixed over the Newton loop), so
  // the per-Newton assembly avoids a binary-search ipc_spIdx per matrix entry. -1 marks skipped entries
  // (upper-tri or pinned vertex). Order matches the assembly loop: idx = (i*3+j)*9 + a*3+b.
  int* escat = (N > 0) ? (int*) mju_malloc(ne*81*sizeof(int)) : NULL;
  if (N > 0) for (int t=0; t < ne; t++) {
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
  for (int it=0; it < 40 && N > 0; it++) {
    for (int i=0; i < N; i++) grad[i] = 0;
    for (int i=0; i < sp.nnz; i++) sp.val[i] = 0;              // sparse Hessian (for the IC(0) precond)
    for (int fi=0; fi < N/3; fi++) for (int c=0; c < 3; c++)   // inertia: diagonal
      sp.val[ipc_spIdx(&sp, 3*fi+c, 3*fi+c)] += mdiag[3*fi+c];
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
      for (int a=0; a < 3; a++) {                            // geometric stress part
        int p = ipc_eedge[a][0], q = ipc_eedge[a][1];
        for (int k=0; k < 3; k++) {
          He[3*p+k][3*p+k] += Me[a]; He[3*q+k][3*q+k] += Me[a];
          He[3*p+k][3*q+k] -= Me[a]; He[3*q+k][3*p+k] -= Me[a]; } }
      for (int i=0; i < 3; i++) for (int j=0; j < 3; j++) {   // scatter lower-tri into sp (precomputed idx)
        int base = t*81 + (i*3+j)*9;
        for (int a=0; a < 3; a++) for (int b=0; b < 3; b++) {
          int id = escat[base + a*3+b]; if (id >= 0) sp.val[id] += He[3*i+a][3*j+b]; } }
    }
    // assemble active contacts. First Newton iter: scan all candidates (initializes cgap exactly). Later
    // iters: only re-test the active set {cgap < ghat} (cgap is a maintained lower bound, so this set
    // contains every candidate that is or could be active). grad/acon/ccache get the active barrier blocks.
    int nacon = 0;
    if (it == 0) {
      for (int c=0; c < ncand; c++)
        ipc_try(cand[c], m, d, x, gv, ge, r, rad, ghat, kappa, fidx, grad, acon, ccache, &nacon, amax, &cgap[c]);
    } else {
      for (int c=0; c < ncand; c++) if (cgap[c] < ghat)
        ipc_try(cand[c], m, d, x, gv, ge, r, rad, ghat, kappa, fidx, grad, acon, ccache, &nacon, amax, &cgap[c]);
    }
    for (int c=0; c < nacon; c++) {                          // contact GN: bdd * (cw n)(cw n)^T
      const ipcCC* cc = &ccache[c];
      for (int p=0; p < cc->nidx; p++) for (int q=0; q < cc->nidx; q++) {
        int fp = cc->f[p], fq = cc->f[q]; if (fp < 0 || fq < 0) continue;
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
    mjtNum gn = 0; for (int i=0; i < N; i++) gn += grad[i]*grad[i];
    mjtNum gnorm = sqrt(gn);
    if (g0 < 0) g0 = gnorm;                                   // initial residual (first Newton iteration)
    if (gnorm < 1e-7*g0 + 1e-9) break;
    ipc_ic0(&sp, 1e-9);                                       // incomplete-LDL^T factor (signed pivots)
    ipc_pcg(dx, grad, N, &sp, mdiag, ne, elems, estr, ccache, nacon, rcg, zcg, pcg, Hpv);
    mjtNum maxdx = 0;   // max free-vertex displacement of the Newton direction (for the lower-bound decay)
    for (int v=0; v < N/3; v++) { mjtNum n2 = dx[3*v]*dx[3*v]+dx[3*v+1]*dx[3*v+1]+dx[3*v+2]*dx[3*v+2];
                                  if (n2 > maxdx) maxdx = n2; }
    maxdx = sqrt(maxdx);
    // proper IPC line search: alpha bounded by the rigorous additive CCD over the candidate contacts
    // (alpha -> ~1 when nothing is closing, so the full Newton step resolves contacts; no pair can
    // cross), then Armijo backtrack on the energy (which iterates the candidate set, so it sees
    // contacts activating during the step). Replaces the arbitrary 0.4*ghat step-cap.
    mjtNum cap = ipc_ccd(m, d, x, dx, gv, ge, r, rad, nfv, fidx, cand, ncand, cgap);
    // E0 = energy at the current x. Reuse the gaps ipc_try just cached for the barrier term, so no
    // candidate closest-point is recomputed here (the line-search trials below are at xn, so they do).
    mjtNum E0 = ipc_energyBase(m, npt, ne, elems, x, xtil, fidx, mass, h);
    for (int c=0; c < ncand; c++) if (cgap[c] > 0 && cgap[c] < ghat) E0 += kappa*ipc_Bg(cgap[c], ghat);
    // line-search subset: a candidate can contribute to the barrier at xn = x + alpha*dx (alpha in
    // [0,cap], cap<=1) only if its gap can drop below ghat. Its gap drop over a full step is bounded
    // by the sum of |dx| over its flex vertices, so keep only c with cgap[c] < ghat + that bound.
    int nls = 0;
    for (int c=0; c < ncand; c++) {
      int vv[4], nvv; ipc_conVerts(&cand[c], vv, &nvv);
      mjtNum lub = 0;
      for (int q=0; q < nvv; q++) { int fq = fidx[vv[q]]; if (fq < 0) continue;
        lub += sqrt(dx[3*fq]*dx[3*fq] + dx[3*fq+1]*dx[3*fq+1] + dx[3*fq+2]*dx[3*fq+2]); }
      if (cgap[c] < ghat + lub) candLS[nls++] = cand[c];
    }
    mjtNum gdx = 0; for (int i=0; i < N; i++) gdx += grad[i]*dx[i];
    mjtNum alpha = cap;
    int lsok = 0;
    for (int ls=0; ls < 30; ls++) {
      for (int i=0; i < 3*npt; i++) xn[i] = x[i];
      for (int v=0; v < npt; v++) if (fidx[v] >= 0) { int fi = fidx[v];
        for (int c=0; c < 3; c++) xn[3*v+c] = x[3*v+c] + alpha*dx[3*fi+c]; }
      if (ipc_energy(m, d, npt, ne, elems, xn, xtil, fidx, mass, h, r, rad, ghat, kappa,
                     gv, ge, candLS, nls) <= E0 + 1e-4*alpha*gdx) { lsok = 1; break; }
      alpha *= 0.5;
    }
    // stagnation guard: if the line search exhausts all backtracks without an energy decrease, the
    // (descent) Newton step cannot reduce the energy even at alpha -> 0, i.e. we are sitting at the
    // minimum to numerical precision. Stop instead of grinding out further no-progress iterations.
    if (!lsok) break;
    for (int i=0; i < 3*npt; i++) x[i] = xn[i];
    // keep cgap a valid lower bound: every gap can shrink by at most 4 vertices * the max vertex step
    mjtNum dgap = 4.0*alpha*maxdx;
    for (int c=0; c < ncand; c++) cgap[c] -= dgap;
  }
  for (int v=0; v < npt; v++) if (fidx[v] >= 0) {
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
  d->time += h;
  if (N > 0) ipc_spFree(&sp);
  mju_free(escat);
  mju_free(dofadr); mju_free(qpadr); mju_free(fidx); mju_free(mass); mju_free(elems);
  mju_free(rad); mju_free(pbody); mju_free(pgeom);
  mju_free(acon); mju_free(ccache); mju_free(cand); mju_free(cgap); mju_free(candLS);
  mju_free(gv); mju_free(ge); mju_free(estr);
  mju_free(x); mju_free(xtil); mju_free(xold); mju_free(xn);
  mju_free(grad); mju_free(dx); mju_free(mdiag);
  mju_free(rcg); mju_free(zcg); mju_free(pcg); mju_free(Hpv);
}


// -------------------------------------------------------------------------------------------------
// Thin wrappers exposing the internal IPC kernels to the unit tests (engine_ipc_test.cc). These let
// the barrier and the geometry/contact distance functions be checked directly, without stepping a
// model. Not a supported API; kept here so the kernels themselves stay static.

// C-IPC offset barrier on surface gap g (support (0,ghat)): value, and optionally 1st/2nd derivative.
mjtNum mj_ipcBarrier(mjtNum g, mjtNum ghat, mjtNum* d1, mjtNum* d2) {
  if (g > 0 && g < ghat) {
    if (d1) *d1 = ipc_Bd(g, ghat);
    if (d2) *d2 = ipc_Bdd(g, ghat);
    return ipc_Bg(g, ghat);
  }
  if (d1) *d1 = 0;
  if (d2) *d2 = 0;
  return 0;
}

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
