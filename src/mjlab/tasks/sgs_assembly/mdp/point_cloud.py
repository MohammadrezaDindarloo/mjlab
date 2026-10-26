"""Scene point cloud in the gripper frame (the SGS teacher's scene observation).

Paper (App. A.3.2): the teacher observes "a scene point cloud of the gripper, held
object, and board expressed in the gripper frame"; (App. C.2/C.4) with small
Gaussian noise on the points (1 mm per step for rod-in-hole).

How it works here (no rendering, no camera):
  1. At start-up, sample points once, uniformly by area, on the surfaces of each
     part: the gripper (visual meshes of the hand and both fingers), the peg
     (its box) and the hole fixture (its four wall boxes). Each point is stored
     in the frame of the body it sits on, so it moves with that body.
  2. Every step, move each point with its body (world frame), then express it in
     the gripper (TCP site) frame:  p_tcp = R_tcp^T (p_world - x_tcp).
The point *order* is fixed (gripper points first, then peg, then hole), so a
label term can tell the network which part each point belongs to.

Counts are our choice (the paper does not give them): see ``POINTS``. Within the
gripper, points are split per body (``GRIPPER_SPLIT``): by area alone almost all
would land on the large, static hand housing (58 of 64, one on a finger), while
the fingers are what shows the gripper opening.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.managers.manager_base import ManagerTermBase
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

# Part -> (number of points, bodies whose geoms are sampled, which geoms).
POINTS: dict[str, int] = {"gripper": 64, "peg": 128, "hole": 128}
PART_BODIES: dict[str, tuple[str, ...]] = {
  "gripper": ("robot/hand", "robot/left_finger", "robot/right_finger"),
  "peg": ("peg/peg",),
  "hole": ("hole/hole",),
}
GRIPPER_SPLIT: dict[str, int] = {
  "robot/hand": 32,
  "robot/left_finger": 16,
  "robot/right_finger": 16,
}
VISUAL_GROUP = 2  # Gripper: sample what a camera would see (visual meshes).
COLLISION_GROUPS = (0, 3)  # Peg and hole: their (box) geoms.
TCP_SITE = "robot/gripper"
# Plain ints: numpy ints do not compare equal to mujoco.mjtGeom enum members.
BOX, MESH = int(mujoco.mjtGeom.mjGEOM_BOX), int(mujoco.mjtGeom.mjGEOM_MESH)
SAMPLE_SEED = 0


def _box_surface(size: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
  """n points uniform by area on a box with half-sizes ``size`` (geom frame)."""
  hx, hy, hz = size
  # Faces: normal axis, sign, area.
  faces = [(ax, s) for ax in range(3) for s in (-1.0, 1.0)]
  areas = np.array([4 * np.prod(np.delete(size, ax)) for ax, _ in faces])
  pick = rng.choice(len(faces), size=n, p=areas / areas.sum())
  pts = rng.uniform(-1.0, 1.0, size=(n, 3)) * np.array([hx, hy, hz])
  for i, f in enumerate(pick):
    ax, s = faces[f]
    pts[i, ax] = s * size[ax]
  return pts


def _mesh_surface(
  model: mujoco.MjModel, mesh_id: int, n: int, rng: np.random.Generator
) -> np.ndarray:
  """n points uniform by area on a mesh's triangles (mesh = geom frame)."""
  v0, nv = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
  f0, nf = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
  verts = model.mesh_vert[v0 : v0 + nv]
  tri = verts[model.mesh_face[f0 : f0 + nf]]  # (nf, 3 corners, 3)
  areas = 0.5 * np.linalg.norm(
    np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
  )
  pick = rng.choice(nf, size=n, p=areas / areas.sum())
  # Uniform barycentric coordinates.
  r1, r2 = rng.random(n), rng.random(n)
  s = np.sqrt(r1)
  w = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
  return np.einsum("nk,nkd->nd", w, tri[pick])


def _geom_area(model: mujoco.MjModel, g: int) -> float:
  if int(model.geom_type[g]) == BOX:
    hx, hy, hz = model.geom_size[g]
    return 8.0 * (hx * hy + hy * hz + hx * hz)
  m = model.geom_dataid[g]
  f0, nf = model.mesh_faceadr[m], model.mesh_facenum[m]
  v0 = model.mesh_vertadr[m]
  tri = model.mesh_vert[v0 + model.mesh_face[f0 : f0 + nf]]
  return float(
    0.5
    * np.linalg.norm(
      np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    ).sum()
  )


def sample_part(
  model: mujoco.MjModel, part: str, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
  """Sample n surface points of one part. Returns (body ids (n,), local xyz (n,3)),
  each point expressed in the frame of the body it belongs to."""
  if part == "gripper":
    assert sum(GRIPPER_SPLIT.values()) == n, (GRIPPER_SPLIT, n)
    pieces = [
      _sample_bodies(model, (body,), (VISUAL_GROUP,), k, rng)
      for body, k in GRIPPER_SPLIT.items()
    ]
    return np.concatenate([b for b, _ in pieces]), np.concatenate(
      [p for _, p in pieces]
    )
  return _sample_bodies(model, PART_BODIES[part], COLLISION_GROUPS, n, rng)


def _sample_bodies(
  model: mujoco.MjModel,
  body_names: tuple[str, ...],
  groups: tuple[int, ...],
  n: int,
  rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
  """n points uniform by area over the box/mesh geoms of the given bodies."""
  bodies = {model.body(b).id for b in body_names}
  geoms = [
    g
    for g in range(model.ngeom)
    if model.geom_bodyid[g] in bodies
    and model.geom_group[g] in groups
    and int(model.geom_type[g]) in (BOX, MESH)
  ]
  assert geoms, f"no geoms to sample on {body_names}"
  areas = np.array([_geom_area(model, g) for g in geoms])
  counts = rng.multinomial(n, areas / areas.sum())
  body_ids, local = [], []
  for g, k in zip(geoms, counts, strict=True):
    if k == 0:
      continue
    if int(model.geom_type[g]) == BOX:
      pts = _box_surface(model.geom_size[g], k, rng)
    else:
      pts = _mesh_surface(model, model.geom_dataid[g], k, rng)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[g])
    local.append(pts @ rot.reshape(3, 3).T + model.geom_pos[g])  # Geom -> body.
    body_ids.append(np.full(k, model.geom_bodyid[g]))
  return np.concatenate(body_ids), np.concatenate(local)


class scene_point_cloud(ManagerTermBase):
  """(num_envs, P, 3) xyz of the scene points in the gripper (TCP) frame."""

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    model = env.sim.mj_model
    rng = np.random.default_rng(SAMPLE_SEED)
    body_ids, local = [], []
    for part, n in POINTS.items():
      b, p = sample_part(model, part, n, rng)
      body_ids.append(b)
      local.append(p)
    self.body_ids = torch.tensor(np.concatenate(body_ids), device=env.device)
    self.local = torch.tensor(
      np.concatenate(local), dtype=torch.float32, device=env.device
    )
    self.site_id = model.site(TCP_SITE).id

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    d = env.sim.data
    pos = d.xpos[:, self.body_ids]  # (N, P, 3) world position of each point's body
    quat = d.xquat[:, self.body_ids]  # (N, P, 4)
    world = pos + quat_apply(quat, self.local.expand_as(pos))
    tcp_pos = d.site_xpos[:, self.site_id]  # (N, 3)
    tcp_mat = d.site_xmat[:, self.site_id].reshape(-1, 3, 3)  # (N, 3, 3)
    # p_tcp = R^T (p - x):  row-vector form  (p - x) @ R.
    return torch.einsum("npi,nij->npj", world - tcp_pos[:, None], tcp_mat)


def point_cloud_labels(env: ManagerBasedRlEnv) -> torch.Tensor:
  """(num_envs, P, 3) one-hot part label of each point: gripper, peg, hole."""
  labels = torch.cat(
    [
      torch.eye(len(POINTS), device=env.device)[i].expand(n, -1)
      for i, n in enumerate(POINTS.values())
    ]
  )
  return labels.expand(env.num_envs, -1, -1)
