"""Reset-deck generators (CPU MuJoCo + IK, offline).

Reaching family (SGS App. A.2): "samples the object's pose anywhere in the
workspace, then positions the end-effector to be aligned with the object's grasp
frame using inverse kinematics. The gripper aperture is randomized, so the reset
does not guarantee that the object is already secured in a stable grasp."

Per candidate:
  1. peg orientation uniform over all rotations; position uniform in
     ``REACH_X/Y`` with the centre at most ``REACH_Z_MAX`` high and the lowest
     corner at least 1 mm above the table,
  2. among the peg's grasp groups (objects/peg_hole/grasps.py), keep those whose
     approach points within 60 deg of straight down (a robot on the table cannot
     grasp from below; for any peg orientation at least one exists), pick one at
     random, and within it take the equivalent hand rotation closest to the home
     TCP orientation (least wrist rotation; the peg's symmetry makes them equal),
  3. IK for that TCP pose with a null-space pull toward a *randomised* comfortable
     posture (home + uniform +-``POSTURE_NOISE`` per joint): the 7th degree of
     freedom varies between states but stays away from the joint limits,
  4. finger opening uniform in [peg half-width + 1 mm, fully open],
  5. reject if a joint is beyond ``MAX_JOINT_RANGE_FRACTION`` of its soft range
     (the gripper orientation forces an extreme posture; ~5 % of candidates),
     if anything penetrates (> 0.5 mm), an arm link comes within 5 mm of
     the table or the hole, or two arm links / arm and hand intersect. Arm links
     do not collide in the sim (only the hand does), so they are checked here
     geometrically.
The states are stored with zero velocity; the sim decides what happens next (the
peg may fall if the fingers are too open, as in the paper).
"""

import math

import mujoco
import numpy as np

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.scene import Scene
from mjlab.tasks.sgs_assembly.objects.peg_hole.grasps import grasp_groups
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg
from mjlab.tasks.sgs_assembly.resets.deck import Family, ResetDeck
from mjlab.tasks.sgs_assembly.robots.ik import SiteIK

REACH_X = (0.35, 0.70)  # [m] env frame.
REACH_Y = (-0.30, 0.30)
REACH_Z_MAX = 0.30  # [m] highest peg centre.
MAX_APPROACH_TILT = math.radians(60.0)  # Approach within this of straight down.
PENETRATION_TOL = 0.0005  # [m]
ARM_CLEARANCE = 0.005  # [m]
POSTURE_NOISE = 0.3  # [rad] spread of the preferred arm posture around home.
MAX_JOINT_RANGE_FRACTION = 0.9  # Of the half-range from the soft-limit midpoint.
FINGER_OPEN = 0.04  # [m]
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
ARM_JOINTS = [f"robot/fr3_joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["robot/finger_joint1", "robot/finger_joint2"]


def random_quat(rng: np.random.Generator) -> np.ndarray:
  """Uniform random rotation (w, x, y, z), Shoemake's method."""
  u1, u2, u3 = rng.random(3)
  a, b = math.sqrt(1 - u1), math.sqrt(u1)
  return np.array(
    [
      b * math.cos(2 * math.pi * u3),
      a * math.sin(2 * math.pi * u2),
      a * math.cos(2 * math.pi * u2),
      b * math.sin(2 * math.pi * u3),
    ]
  )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
  mat = np.zeros(9)
  mujoco.mju_quat2Mat(mat, q)
  return mat.reshape(3, 3)


class DeckBuilder:
  def __init__(self, env_cfg: ManagerBasedRlEnvCfg, peg_hole: PegHoleCfg):
    self.cfg = peg_hole
    self.model = Scene(env_cfg.scene, device="cpu").compile()
    env_cfg.sim.mujoco.apply(self.model)
    self.data = mujoco.MjData(self.model)
    m = self.model
    self.ik = SiteIK(m, "robot/gripper", ARM_JOINTS, max_iters=300)
    # Respect the soft limits the action term clamps to (95 % of the range).
    mid = (self.ik.lower + self.ik.upper) / 2
    half = (self.ik.upper - self.ik.lower) / 2 * 0.95
    self.ik.lower, self.ik.upper = mid - half, mid + half
    self.arm_adr = self.ik.qadr
    self.finger_adr = [m.jnt_qposadr[m.joint(j).id] for j in FINGER_JOINTS]
    self.peg_adr = m.jnt_qposadr[m.joint("peg/peg_joint").id]
    self.peg_half = np.array([peg_hole.peg_width / 2] * 2 + [peg_hole.peg_length / 2])
    self.arm_geoms = [
      m.geom(f"robot/fr3_link{i}_collision").id for i in range(1, 8)
    ]  # link0 is the base, bolted to the table.
    self.env_geoms = [m.geom("terrain").id] + [
      m.geom(f"hole/hole_wall_{w}").id for w in ("px", "nx", "py", "ny")
    ]
    # The hole (and the robot base) are mocap bodies: their placement lives in
    # mjlab's merged keyframe, not in body_pos. Load it once; set_state only
    # touches joints, so the mocap poses stay put.
    mujoco.mj_resetDataKeyframe(m, self.data, 0)
    mujoco.mj_forward(m, self.data)
    self.hole_pos = self.data.xpos[m.body("hole/hole").id].copy()
    self.rot_home = self.ik.forward(HOME_Q)[1]
    self.self_pairs = self._self_collision_pairs()

  def _self_collision_pairs(self) -> list[tuple[int, int]]:
    """Arm-arm (non-adjacent links) and arm-hand geom pairs to check.

    The convex-hull meshes of some pairs already touch in the home posture; those
    pairs are skipped (they cannot tell a real self-collision from the hull
    approximation).
    """
    m = self.model
    links = [m.geom(f"robot/fr3_link{i}_collision").id for i in range(8)]
    hand = [
      m.geom(f"robot/{n}").id
      for n in ("hand_capsule", "left_finger_box_1", "right_finger_box_1")
    ]
    cand = [(links[i], links[j]) for i in range(8) for j in range(i + 2, 8)]
    cand += [(links[i], h) for i in range(6) for h in hand]
    self.set_state(
      HOME_Q,
      [0.04, 0.04],
      np.array([0.55, -0.15, 0.04]),
      np.array([1.0, 0.0, 0.0, 0.0]),
    )
    fromto = np.zeros(6)
    return [
      (g1, g2)
      for g1, g2 in cand
      if mujoco.mj_geomDistance(m, self.data, g1, g2, 0.05, fromto) > 0.005
    ]

  def set_state(self, arm_q, finger_q, peg_pos, peg_quat) -> None:
    d = self.data
    d.qvel[:] = 0.0
    d.qpos[self.arm_adr] = arm_q
    d.qpos[self.finger_adr] = finger_q
    d.qpos[self.peg_adr : self.peg_adr + 3] = peg_pos
    d.qpos[self.peg_adr + 3 : self.peg_adr + 7] = peg_quat
    mujoco.mj_forward(self.model, d)

  def is_valid(self) -> bool:
    m, d = self.model, self.data
    for k in range(d.ncon):
      if d.contact[k].dist < -PENETRATION_TOL:
        return False
    fromto = np.zeros(6)
    for g in self.arm_geoms:
      for e in self.env_geoms:
        if mujoco.mj_geomDistance(m, d, g, e, ARM_CLEARANCE, fromto) < ARM_CLEARANCE:
          return False
    for g1, g2 in self.self_pairs:
      if mujoco.mj_geomDistance(m, d, g1, g2, 0.01, fromto) < 0.0:
        return False
    return True

  def reaching(self, n: int, rng: np.random.Generator) -> ResetDeck:
    rows: list[tuple] = []
    tries = 0
    while len(rows) < n:
      tries += 1
      quat = random_quat(rng)
      rot = quat_to_mat(quat)
      lowest = np.abs(rot) @ self.peg_half  # Half-extent along world z.
      z_lo = lowest[2] + 0.001
      pos = np.array(
        [rng.uniform(*REACH_X), rng.uniform(*REACH_Y), rng.uniform(z_lo, REACH_Z_MAX)]
      )
      # Grasp groups whose approach (z_hand) points mostly down in the world.
      groups = [
        [(pos + rot @ g_pos, rot @ g_rot) for g_pos, g_rot in group]
        for group in grasp_groups(self.cfg, rng)
        if (rot @ group[0][1])[2, 2] <= -math.cos(MAX_APPROACH_TILT)
      ]
      group = groups[rng.integers(len(groups))]
      # Equivalent hand rotations: take the one needing the least wrist turn.
      tcp_pos, tcp_rot = min(group, key=lambda g: self._angle_to_home(g[1]))
      q_nominal = np.clip(
        HOME_Q + rng.uniform(-POSTURE_NOISE, POSTURE_NOISE, 7),
        self.ik.lower,
        self.ik.upper,
      )
      arm_q = self._solve_ik(tcp_pos, tcp_rot, q_nominal, rng)
      if arm_q is None or not self._comfortable(arm_q):
        continue
      finger_q = rng.uniform(self.cfg.peg_width / 2 + 0.001, FINGER_OPEN)
      self.set_state(arm_q, [finger_q, finger_q], pos, quat)
      if not self.is_valid():
        continue
      rows.append((arm_q, finger_q, pos, quat))
    print(f"reaching: {n} states from {tries} candidates ({n / tries:.0%} kept)")
    return self._to_deck(rows, Family.REACHING)

  def _comfortable(self, q: np.ndarray) -> bool:
    mid = (self.ik.lower + self.ik.upper) / 2
    half = (self.ik.upper - self.ik.lower) / 2
    return bool(np.all(np.abs(q - mid) <= MAX_JOINT_RANGE_FRACTION * half))

  def _angle_to_home(self, tcp_rot: np.ndarray) -> float:
    cos = (np.trace(self.rot_home.T @ tcp_rot) - 1.0) / 2.0
    return math.acos(float(np.clip(cos, -1.0, 1.0)))

  def _solve_ik(self, pos, rot, q_nominal, rng) -> np.ndarray | None:
    seeds = [q_nominal] + [q_nominal + rng.uniform(-0.5, 0.5, 7) for _ in range(2)]
    for seed in seeds:
      seed = np.clip(seed, self.ik.lower, self.ik.upper)
      q, ok = self.ik.solve(seed, pos, rot, q_nominal=q_nominal)
      if ok:
        return q
    return None

  def _to_deck(self, rows: list[tuple], family: Family) -> ResetDeck:
    arm_q, finger_q, pos, quat = (np.array(x) for x in zip(*rows, strict=True))
    return ResetDeck(
      arm_q=arm_q,
      finger_q=finger_q,
      peg_pos=pos,
      peg_quat=quat,
      family=np.full(len(rows), int(family), dtype=np.int8),
      meta={
        "peg_width": self.cfg.peg_width,
        "peg_length": self.cfg.peg_length,
        "clearance": self.cfg.clearance,
        "hole_depth": self.cfg.hole_depth,
        "hole_pos": self.hole_pos.tolist(),
        "arm_joints": ARM_JOINTS,
        "finger_joints": FINGER_JOINTS,
        "frames": "env frame, robot base at origin, quaternions wxyz",
      },
    )
