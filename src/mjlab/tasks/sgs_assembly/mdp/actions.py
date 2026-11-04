"""Action terms: relative Cartesian operational-space control and a binary gripper.

The policy outputs one vector a = [a_arm (6), a_grip (1)]; Eq. 5's action
penalties act on this raw vector.

Arm (SGS insertion experiments, which follow OmniReset's "relative Cartesian
operational-space control interface"; gains, scales and control law from
OmniReset's released code, UWLab ``RelCartesianOSCAction``):

  once per policy step:   x* = x + scale * a_arm      (x = current TCP pose)
  every physics step:     tau = J^T (Kp e - Kd J qd)  (e = 6D error to x*)

``x*`` is re-anchored at the *measured* pose every policy step, so pushing
against an obstacle never winds up: the force is bounded by Kp times one step.
Increments are expressed in the robot base frame, rotations as axis-angle,
composed on the left (rotation about base axes through the TCP). There is no
mass-matrix weighting, as in OmniReset.

Ours, not OmniReset's: OmniReset's UR5e has 6 joints, the FR3 has 7, so
J^T F leaves one joint motion (the elbow swinging) uncontrolled. A weak joint
spring toward the posture at reset, projected so it cannot move the TCP,
holds it:  tau += (I - J^T (J J^T + l^2 I)^-1 J) (k_n (q0 - q) - d_n qd).
Torque limits are the arm's motor actuator limits (robot constants).

Gripper: SGS uses one binary open/close command (paper App. A.3.2). a_grip > 0
opens, a_grip <= 0 closes. The closing force is limited by the gripper actuator's
effort limit (robot constants), not by this term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_from_angle_axis,
  quat_from_matrix,
  quat_inv,
  quat_mul,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

# Damping of the null-space projector's pseudo-inverse (keeps it bounded near
# singular postures).
_NULLSPACE_LAMBDA = 1e-2


@dataclass(frozen=True)
class OscGains:
  """Relative Cartesian OSC parameters, per axis (x, y, z, rx, ry, rz)."""

  scale: tuple[float, ...]
  """Pose increment for action 1 [m, m, m, rad, rad, rad]."""
  stiffness: tuple[float, ...]
  """Kp [N/m, N/m, N/m, Nm/rad, Nm/rad, Nm/rad]."""
  damping_ratio: tuple[float, ...]
  """Kd = 2 * sqrt(Kp) * damping_ratio."""
  nullspace_stiffness: float
  """Joint-space posture spring k_n [Nm/rad] (ours; 0 disables it)."""
  nullspace_damping: float
  """Joint-space posture damper d_n [Nm s/rad] (ours)."""


@dataclass(kw_only=True)
class RelativeCartesianOscActionCfg(ActionTermCfg):
  joint_names: tuple[str, ...]
  """Arm joints, driven by torque (motor) actuators."""
  site_name: str
  """Controlled frame: the TCP site."""
  gains: OscGains

  def build(self, env: ManagerBasedRlEnv) -> RelativeCartesianOscAction:
    return RelativeCartesianOscAction(self, env)


class RelativeCartesianOscAction(ActionTerm):
  cfg: RelativeCartesianOscActionCfg
  _entity: Entity

  def __init__(self, cfg: RelativeCartesianOscActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    ids, _ = self._entity.find_joints(cfg.joint_names)
    self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._dof_ids = self._entity.indexing.joint_v_adr[self._joint_ids]
    site_local, _ = self._entity.find_sites(cfg.site_name)
    self._site_id = int(self._entity.indexing.site_ids[site_local[0]].item())
    body_id = int(env.sim.mj_model.site_bodyid[self._site_id])

    g = cfg.gains
    dev = self.device
    self._scale = torch.tensor(g.scale, device=dev)
    self._kp = torch.tensor(g.stiffness, device=dev)
    self._kd = 2.0 * torch.sqrt(self._kp) * torch.tensor(g.damping_ratio, device=dev)

    n = self.num_envs
    self._raw = torch.zeros(n, 6, device=dev)
    self._pos_des = torch.zeros(n, 3, device=dev)
    self._quat_des = torch.zeros(n, 4, device=dev)
    self._quat_des[:, 0] = 1.0
    self._q_posture = torch.zeros(n, len(ids), device=dev)

    # Site Jacobian on the GPU (as mjlab's DifferentialIKAction does).
    nv = env.sim.mj_model.nv
    with wp.ScopedDevice(env.sim.wp_device):
      self._jacp_wp = wp.zeros((n, 3, nv), dtype=float)
      self._jacr_wp = wp.zeros((n, 3, nv), dtype=float)
      self._point_wp = wp.zeros(n, dtype=wp.vec3)
      self._body_wp = wp.full(n, body_id, dtype=wp.int32)
    self._jacp = wp.to_torch(self._jacp_wp)
    self._jacr = wp.to_torch(self._jacr_wp)
    self._point = wp.to_torch(self._point_wp).view(n, 3)

  @property
  def action_dim(self) -> int:
    return 6

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    """Once per policy step: desired TCP pose = measured pose + increment."""
    self._raw[:] = actions
    delta = actions * self._scale
    base_quat = self._entity.data.root_link_quat_w
    pos, quat = self._tcp_pose()
    self._pos_des[:] = pos + quat_apply(base_quat, delta[:, :3])
    rotvec = quat_apply(base_quat, delta[:, 3:])
    angle = torch.linalg.norm(rotvec, dim=1)
    axis = rotvec / angle.clamp(min=1e-9).unsqueeze(1)
    self._quat_des[:] = quat_mul(quat_from_angle_axis(angle, axis), quat)

  def apply_actions(self) -> None:
    """Every physics step: task-space PD torques plus the null-space posture."""
    pos, quat = self._tcp_pose()
    q = self._entity.data.joint_pos[:, self._joint_ids]
    qd = self._entity.data.joint_vel[:, self._joint_ids]
    jac = self._jacobian(pos)  # (n, 6, n_arm), world frame.

    err = torch.cat(
      [
        self._pos_des - pos,
        axis_angle_from_quat(quat_mul(self._quat_des, quat_inv(quat))),
      ],
      dim=1,
    )
    vel = torch.einsum("bij,bj->bi", jac, qd)
    force = self._kp * err - self._kd * vel
    tau = torch.einsum("bji,bj->bi", jac, force)

    g = self.cfg.gains
    if g.nullspace_stiffness > 0.0:
      jjt = jac @ jac.transpose(1, 2)
      jjt.diagonal(dim1=1, dim2=2).add_(_NULLSPACE_LAMBDA**2)
      eye = torch.eye(q.shape[1], device=self.device)
      proj = eye - jac.transpose(1, 2) @ torch.linalg.solve(jjt, jac)
      posture = g.nullspace_stiffness * (self._q_posture - q) - g.nullspace_damping * qd
      tau = tau + torch.einsum("bij,bj->bi", proj, posture)

    self._entity.set_joint_effort_target(tau, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    # Reset events have already written the new joint state: hold it.
    pos, quat = self._tcp_pose()
    self._pos_des[env_ids] = pos[env_ids]
    self._quat_des[env_ids] = quat[env_ids]
    q = self._entity.data.joint_pos[:, self._joint_ids]
    self._q_posture[env_ids] = q[env_ids]

  def _tcp_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
    data = self._env.sim.data
    pos = data.site_xpos[:, self._site_id]
    quat = quat_from_matrix(data.site_xmat[:, self._site_id])
    return pos, quat

  def _jacobian(self, point: torch.Tensor) -> torch.Tensor:
    self._point[:] = point
    with wp.ScopedDevice(self._env.sim.wp_device):
      mjwarp.jac(
        self._env.sim.wp_model,
        self._env.sim.wp_data,
        self._jacp_wp,
        self._jacr_wp,
        self._point_wp,
        self._body_wp,
      )
    return torch.cat(
      [self._jacp[:, :, self._dof_ids], self._jacr[:, :, self._dof_ids]], dim=1
    )


@dataclass(kw_only=True)
class BinaryGripperActionCfg(ActionTermCfg):
  joint_name: str
  """The actuated gripper joint."""
  open_pos: float
  closed_pos: float

  def build(self, env: ManagerBasedRlEnv) -> BinaryGripperAction:
    return BinaryGripperAction(self, env)


class BinaryGripperAction(ActionTerm):
  cfg: BinaryGripperActionCfg
  _entity: Entity

  def __init__(self, cfg: BinaryGripperActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    ids, _ = self._entity.find_joints((cfg.joint_name,))
    assert len(ids) == 1, f"Expected one gripper joint, got {ids}"
    self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._raw = torch.zeros(self.num_envs, 1, device=self.device)
    self._target = torch.full((self.num_envs, 1), cfg.open_pos, device=self.device)

  @property
  def action_dim(self) -> int:
    return 1

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    self._target = torch.where(
      actions > 0.0,
      torch.full_like(actions, self.cfg.open_pos),
      torch.full_like(actions, self.cfg.closed_pos),
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    # Keep the aperture the reset left the fingers at until the first action.
    self._target[env_ids] = self._entity.data.joint_pos[env_ids][:, self._joint_ids]
