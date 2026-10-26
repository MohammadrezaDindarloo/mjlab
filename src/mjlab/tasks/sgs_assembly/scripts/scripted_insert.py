"""Scripted pick-and-place: the FR3 grasps the peg and inserts it in the hole.

Not a learned policy: a hand-written state machine that uses privileged state
(peg pose) and IK. Its job is to show the task is solvable with this gripper,
clearance and friction through real dynamics, and that success fires from real
contact. It talks to the env through the same 8-D action a policy would output
(7 delta joint targets + binary gripper), so it also exercises the action terms.

Phases (TCP = tool-centre point between the fingertips, gripper pointing down):
  APPROACH   TCP 10 cm above the grasp point, gripper open, yaw aligned to a peg face
  DESCEND    straight down to the grasp point (2 cm below the peg's top)
  CLOSE      close the gripper, wait 0.5 s
  LIFT       straight up to z = 0.25 m
  TRANSPORT  move so the peg tip hovers 2 cm above the hole axis, peg yaw aligned
             with the hole; closed loop on the *measured* peg pose, which cancels
             any offset of the peg inside the grasp
  INSERT     lower the peg tip slowly along the hole axis until success ends the
             episode (the env then resets and the script starts over)
Every step the commanded TCP moves toward the phase goal with capped speed and
acceleration, IK turns it into joint targets, and the action is
(q_ik - q_target) / scale, scaled down uniformly if any joint exceeds 1.

Result (64 envs x 40 s): ~94 % of episodes succeed in ~9 s. The rest drop the peg
during LIFT: the peg slowly rotates in the grasp, the two pads penetrate it
unequally (~1.0 vs ~0.3 mm, soft contacts), and at ~4.5 deg tilt it is squeezed
out. Friction is never the limit (|f_t|/f_n <= 0.7 << mu = 2).

  # Watch it (viewer starts paused: press play in the viser panel).
  uv run python src/mjlab/tasks/sgs_assembly/scripts/scripted_insert.py --viewer
  # Headless: success statistics over 16 envs x 30 s.
  uv run python src/mjlab/tasks/sgs_assembly/scripts/scripted_insert.py
"""

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import torch
import tyro

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import HOLE_POS
from mjlab.tasks.sgs_assembly.mdp.actions import DeltaJointPositionAction
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg
from mjlab.tasks.sgs_assembly.robots.franka.franka_constants import FR3_INFO
from mjlab.tasks.sgs_assembly.robots.ik import SiteIK
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer import ViserPlayViewer

PEG = PegHoleCfg()
HOLE = np.array(HOLE_POS)
GRASP_BELOW_TOP = 0.02  # [m] TCP below the peg's top end when grasping.
APPROACH_HEIGHT = 0.10  # [m]
LIFT_Z = 0.25  # [m]
HOVER = 0.02  # [m] peg tip above the hole entrance before inserting.
SPEED_FAST = 0.25  # [m/s] commanded TCP speed.
SPEED_INSERT = 0.03  # [m/s]
YAW_RATE = 1.5  # [rad/s]
# Acceleration limits. Without them the command jumps from 0 to full speed in
# one step, and the shaking makes ~25 % of grasps drop the peg (see README).
ACCEL = 0.5  # [m/s^2]
YAW_ACCEL = 2.0  # [rad/s^2]
CLOSE_TIME = 0.5  # [s]
REACHED = 0.003  # [m] measured TCP within this of the command = "arrived".
ALIGNED = 0.0003  # [m] peg tip within this of the hole axis before inserting.


class Phase(IntEnum):
  APPROACH = 0
  DESCEND = 1
  CLOSE = 2
  LIFT = 3
  TRANSPORT = 4
  INSERT = 5


@dataclass
class EnvState:
  phase: Phase = Phase.APPROACH
  timer: float = 0.0
  aligned_steps: int = 0
  cmd_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
  cmd_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
  cmd_yaw: float = 0.0
  cmd_yaw_rate: np.ndarray = field(default_factory=lambda: np.zeros(1))
  grasp_pos: np.ndarray | None = None
  grasp_yaw: float = 0.0


def move_toward(
  x: np.ndarray, v: np.ndarray, goal: np.ndarray, vmax: float, amax: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
  """One step of a trapezoidal velocity profile from x toward goal.

  Speed ramps up and down at most amax, never exceeds vmax, and the profile
  decelerates so it stops at the goal (v_des = sqrt(2 * amax * distance)).
  Returns the new (x, v).
  """
  delta = goal - x
  dist = float(np.linalg.norm(delta))
  if dist < 1e-9:
    return goal.copy(), np.zeros_like(v)
  v_des = delta / dist * min(vmax, math.sqrt(2.0 * amax * dist))
  dv = v_des - v
  dv_norm = float(np.linalg.norm(dv))
  if dv_norm > amax * dt:
    dv *= amax * dt / dv_norm
  v = v + dv
  if float(np.linalg.norm(v)) * dt >= dist:  # Would reach or pass the goal.
    return goal.copy(), np.zeros_like(v)
  return x + v * dt, v


def wrap_quarter(a: float) -> float:
  """Wrap an angle to [-pi/4, pi/4): a square is symmetric under 90 deg turns."""
  return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


def yaw_of(rot: np.ndarray) -> float:
  return math.atan2(rot[1, 0], rot[0, 0])


def rot_z(yaw: float) -> np.ndarray:
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ScriptedInsertPolicy:
  def __init__(self, env: ManagerBasedRlEnv):
    self.env = env
    self.dt = env.step_dt
    self.n = env.num_envs
    arm = env.action_manager.get_term("arm")
    assert isinstance(arm, DeltaJointPositionAction)
    self.arm = arm
    self.scale = arm.cfg.scale
    joints = [f"robot/fr3_joint{i}" for i in range(1, 8)]
    self.ik = SiteIK(env.sim.mj_model, f"robot/{FR3_INFO.ee_site}", joints)
    home = env.scene["robot"].data.default_joint_pos[0, :7].cpu().numpy()
    self.rot_home = self.ik.forward(home)[1]  # Gripper pointing down, yaw 0.
    self.states = [EnvState() for _ in range(self.n)]
    self._fresh = [True] * self.n

  def reset(self, dones: torch.Tensor | None = None) -> None:
    ids = range(self.n) if dones is None else dones.nonzero().flatten().tolist()
    for i in ids:
      self.states[i] = EnvState()
      self._fresh[i] = True

  def _measure(self) -> dict[str, np.ndarray]:
    origins = self.env.scene.env_origins
    robot, peg = self.env.scene["robot"].data, self.env.scene["peg"].data
    site = self.env.scene["robot"].find_sites((FR3_INFO.ee_site,))[0][0]
    return {
      "tcp": (robot.site_pos_w[:, site] - origins).cpu().numpy(),
      "tcp_rot": matrix_from_quat(robot.site_quat_w[:, site]).cpu().numpy(),
      "peg": (peg.root_link_pos_w - origins).cpu().numpy(),
      "peg_rot": matrix_from_quat(peg.root_link_quat_w).cpu().numpy(),
      "q_target": self.arm.target.cpu().numpy(),
    }

  def __call__(self, obs) -> torch.Tensor:
    del obs  # Uses privileged state instead.
    m = self._measure()
    actions = np.zeros((self.n, 8), dtype=np.float32)
    for i, st in enumerate(self.states):
      tcp, tcp_rot = m["tcp"][i], m["tcp_rot"][i]
      hand_yaw = yaw_of(tcp_rot @ self.rot_home.T)
      if self._fresh[i]:
        st.cmd_pos, st.cmd_yaw, self._fresh[i] = tcp.copy(), hand_yaw, False
      peg_c, peg_rot = m["peg"][i], m["peg_rot"][i]
      axis = peg_rot[:, 2] * (1.0 if peg_rot[2, 2] >= 0 else -1.0)
      peg_tip = peg_c - PEG.peg_length / 2 * axis  # Lower end.
      goal, yaw, speed, grip = self._phase_goal(
        st, tcp, hand_yaw, peg_c, peg_rot, peg_tip
      )

      # Move the commanded pose toward the goal with capped speed AND acceleration.
      st.cmd_pos, st.cmd_vel = move_toward(
        st.cmd_pos, st.cmd_vel, goal, speed, ACCEL, self.dt
      )
      cmd_yaw, st.cmd_yaw_rate = move_toward(
        np.array([st.cmd_yaw]),
        st.cmd_yaw_rate,
        np.array([yaw]),
        YAW_RATE,
        YAW_ACCEL,
        self.dt,
      )
      st.cmd_yaw = float(cmd_yaw[0])
      arrived = bool(
        np.linalg.norm(goal - st.cmd_pos) < 1e-6
        and abs(yaw - st.cmd_yaw) < 1e-6
        and np.linalg.norm(tcp - st.cmd_pos) < REACHED
      )
      self._advance(st, arrived, peg_tip)

      q_target = m["q_target"][i]
      q_des, _ = self.ik.solve(q_target, st.cmd_pos, rot_z(st.cmd_yaw) @ self.rot_home)
      # Scale the whole arm action uniformly into [-1, 1]. Clipping joints one by
      # one would change the joint-space direction and tilt the hand off its path.
      a_arm = (q_des - q_target) / self.scale
      actions[i, :7] = a_arm / max(1.0, float(np.abs(a_arm).max()))
      actions[i, 7] = grip
    return torch.from_numpy(actions).to(self.env.device)

  def _phase_goal(self, st, tcp, hand_yaw, peg_c, peg_rot, peg_tip):
    """Returns (goal TCP position, goal hand yaw, speed, gripper action)."""
    if st.grasp_pos is None:  # Entering APPROACH: plan the grasp from the peg pose.
      top = peg_c[2] + PEG.peg_length / 2
      st.grasp_pos = np.array([peg_c[0], peg_c[1], top - GRASP_BELOW_TOP])
      st.grasp_yaw = wrap_quarter(yaw_of(peg_rot))
    if st.phase == Phase.APPROACH:
      above = st.grasp_pos + np.array([0.0, 0.0, APPROACH_HEIGHT])
      return above, st.grasp_yaw, SPEED_FAST, 1.0
    if st.phase == Phase.DESCEND:
      return st.grasp_pos, st.grasp_yaw, SPEED_FAST / 2, 1.0
    if st.phase == Phase.CLOSE:
      return st.cmd_pos, st.cmd_yaw, 0.0, -1.0
    if st.phase == Phase.LIFT:
      lifted = np.array([st.cmd_pos[0], st.cmd_pos[1], LIFT_Z])
      return lifted, st.cmd_yaw, SPEED_FAST, -1.0
    # TRANSPORT / INSERT: servo the measured peg tip onto the hole axis.
    # The tip-to-TCP offset is fixed by the grasp, so goal = desired tip - offset.
    offset = peg_tip - tcp
    peg_yaw_in_hand = yaw_of(peg_rot) - hand_yaw
    # Hand yaw that turns the peg square onto the (unrotated) hole square,
    # choosing the equivalent 90-degree option closest to the current command.
    yaw = st.cmd_yaw + wrap_quarter(-peg_yaw_in_hand - st.cmd_yaw)
    if st.phase == Phase.TRANSPORT:
      tip_goal = HOLE + np.array([0.0, 0.0, PEG.hole_depth + HOVER])
      return tip_goal - offset, yaw, SPEED_FAST, -1.0
    tip_goal = HOLE + np.array([0.0, 0.0, -0.002])  # Slightly below the bottom.
    return tip_goal - offset, yaw, SPEED_INSERT, -1.0

  def _advance(self, st: EnvState, arrived: bool, peg_tip: np.ndarray) -> None:
    st.timer += self.dt
    nxt = None
    if st.phase in (Phase.APPROACH, Phase.DESCEND, Phase.LIFT) and arrived:
      nxt = Phase(st.phase + 1)
    elif st.phase == Phase.CLOSE and st.timer >= CLOSE_TIME:
      nxt = Phase.LIFT
    elif st.phase == Phase.TRANSPORT:
      lateral = np.linalg.norm(peg_tip[:2] - HOLE[:2])
      st.aligned_steps = st.aligned_steps + 1 if arrived and lateral < ALIGNED else 0
      if st.aligned_steps >= 5:
        nxt = Phase.INSERT
    if nxt is not None:
      st.phase, st.timer = nxt, 0.0


def main(viewer: bool = False, num_envs: int = 16, seconds: float = 30.0) -> None:
  """Args:
  viewer: Open the viser viewer instead of running headless statistics.
  num_envs: Parallel envs.
  seconds: Sim time for the headless run.
  """
  cfg = load_env_cfg("Mjlab-SGS-PegHole-FR3")
  # The script needs ~9 s per insertion; the training time-out (10 s) would cut
  # slow runs off, so give the demo more time. This does not change the task.
  cfg.episode_length_s = 20.0
  cfg.scene.num_envs = num_envs if not viewer else min(num_envs, 4)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  policy = ScriptedInsertPolicy(env)
  if viewer:
    ViserPlayViewer(wrapped, policy).run()
    return

  tm = env.termination_manager
  outcomes = {name: 0 for name in tm.active_terms}
  episode_time = []
  wrapped.reset()
  policy.reset()
  for _ in range(int(seconds / env.step_dt)):
    obs = wrapped.get_observations()
    elapsed = env.episode_length_buf.clone() + 1
    _, _, dones, _ = wrapped.step(policy(obs))
    for name in outcomes:
      fired = tm.get_term(name)
      outcomes[name] += int(fired.sum())
      if name == "success" and fired.any():
        episode_time += (elapsed[fired] * env.step_dt).tolist()
    policy.reset(dones)
  total = sum(outcomes[k] for k in outcomes if k != "time_out") + outcomes["time_out"]
  print(f"\n{num_envs} envs x {seconds} s: episode outcomes {outcomes}")
  if episode_time:
    print(
      f"success rate {outcomes['success'] / max(total, 1):.0%}, "
      f"mean time to success {np.mean(episode_time):.2f} s"
    )
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
