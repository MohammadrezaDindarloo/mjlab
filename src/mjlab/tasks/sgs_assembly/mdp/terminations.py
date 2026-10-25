"""Terminations: success S(s) and failure F(s) (SGS App. A.3.2), plus mjlab's
time-out.

F(s) = any arm joint faster than ``factor`` x its hardware limit (factor = 2 in
the paper), or the held object outside the workspace box.
The reward reads these same terms, so reward and episode end can never disagree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.sgs_assembly.mdp.success import InsertionGatesCfg, insertion_success

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def success(
  env: ManagerBasedRlEnv,
  peg_half_length: float,
  hole_depth: float,
  gates: InsertionGatesCfg,
) -> torch.Tensor:
  return insertion_success(env, peg_half_length, hole_depth, gates)


def joint_vel_exceeded(
  env: ManagerBasedRlEnv,
  limits: tuple[float, ...],
  factor: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """``limits`` is ordered like the joints matched by ``asset_cfg.joint_names``."""
  qd = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  limit = torch.tensor(limits, device=env.device)
  return torch.any(qd.abs() > factor * limit, dim=1)


def object_out_of_bounds(
  env: ManagerBasedRlEnv,
  entity_name: str,
  lower: tuple[float, float, float],
  upper: tuple[float, float, float],
) -> torch.Tensor:
  """Object origin outside the box [lower, upper], in the env frame [m]."""
  pos = env.scene[entity_name].data.root_link_pos_w - env.scene.env_origins
  lo = torch.tensor(lower, device=env.device)
  hi = torch.tensor(upper, device=env.device)
  return torch.any((pos < lo) | (pos > hi), dim=1)
