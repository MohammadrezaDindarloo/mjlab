"""SGS manipulation reward, paper Eq. 5 (App. A.3.2):

  r = l_succ * 1[S] - l_fail * 1[F]
      - l_act * C(||a_t||^2) - l_da * C(||a_t - a_{t-1}||^2) - l_qd * C(||qd_arm||^2)

with C(x) = min(x, 1e4) and a the raw policy output. Each term below returns the
*unweighted* quantity; the weights (with sign) live in ``SgsRewardWeights`` and
are attached in env_cfg.py. ``scale_rewards_by_dt`` is False, so these are the
per-step values exactly as in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CLAMP_MAX = 1e4  # C(x) = min(x, 1e4).


@dataclass(frozen=True)
class SgsRewardWeights:
  success: float = 100.0
  failure: float = 1.0
  action: float = 1e-5
  action_rate: float = 1e-4
  arm_joint_vel: float = 1e-4


def _clamp(x: torch.Tensor) -> torch.Tensor:
  return torch.clamp(x, max=CLAMP_MAX)


def terminated_by(env: ManagerBasedRlEnv, term_names: tuple[str, ...]) -> torch.Tensor:
  """1.0 where any of the named termination terms fired this step.

  Termination terms are computed before rewards in the same env step, so this
  is exactly the indicator 1[S] or 1[F] that ends the episode.
  """
  tm = env.termination_manager
  fired = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for name in term_names:
    fired |= tm.get_term(name)
  return fired.float()


def action_l2_clamped(env: ManagerBasedRlEnv) -> torch.Tensor:
  a = env.action_manager.action
  return _clamp(torch.sum(a**2, dim=1))


def action_rate_l2_clamped(env: ManagerBasedRlEnv) -> torch.Tensor:
  da = env.action_manager.action - env.action_manager.prev_action
  return _clamp(torch.sum(da**2, dim=1))


def joint_vel_l2_clamped(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  qd = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  return _clamp(torch.sum(qd**2, dim=1))
