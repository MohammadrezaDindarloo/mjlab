"""Action terms: delta joint-position arm control and a binary gripper.

The policy outputs one vector a = [a_arm (n_arm), a_grip (1)]; Eq. 5's action
penalties act on this raw vector.

Arm: q_target <- clamp(q_target + scale * a_arm, soft joint limits), with q_target
set to the measured joint positions at reset. Accumulating the target (instead of
"home + scale * a") means a zero action holds the current pose, so any reset state
(e.g. a reset-deck grasp far from home) is not undone on the first step.

Gripper: SGS uses one binary open/close command (paper App. A.3.2). a_grip > 0
opens, a_grip <= 0 closes. The closing force is limited by the gripper actuator's
effort limit (robot constants), not by this term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class DeltaJointPositionActionCfg(ActionTermCfg):
  joint_names: tuple[str, ...]
  scale: float
  """Target change per policy step for a = 1 [rad]."""

  def build(self, env: ManagerBasedRlEnv) -> DeltaJointPositionAction:
    return DeltaJointPositionAction(self, env)


class DeltaJointPositionAction(ActionTerm):
  cfg: DeltaJointPositionActionCfg
  _entity: Entity

  def __init__(self, cfg: DeltaJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    ids, _ = self._entity.find_joints(cfg.joint_names)
    self._joint_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._raw = torch.zeros(self.num_envs, len(ids), device=self.device)
    self._target = self._entity.data.default_joint_pos[:, self._joint_ids].clone()
    limits = self._entity.data.soft_joint_pos_limits[:, self._joint_ids]
    self._lower, self._upper = limits[..., 0], limits[..., 1]

  @property
  def action_dim(self) -> int:
    return self._raw.shape[1]

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw

  @property
  def target(self) -> torch.Tensor:
    """Current arm joint-position targets, (num_envs, n_arm). Read-only use."""
    return self._target

  def process_actions(self, actions: torch.Tensor) -> None:
    self._raw[:] = actions
    target = self._target + self.cfg.scale * actions
    self._target = torch.clamp(target, self._lower, self._upper)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._target, joint_ids=self._joint_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw[env_ids] = 0.0
    # Reset events have already written the new joint state at this point.
    joint_pos = self._entity.data.joint_pos[:, self._joint_ids]
    self._target[env_ids] = joint_pos[env_ids]


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
