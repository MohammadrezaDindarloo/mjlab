"""FR3 + peg-in-hole task variants.

Layout (env frame, metres): robot base at the origin, hole fixture at
``HOLE_POS``, peg standing upright around ``PEG_XY``. The FR3 home TCP is at about
(0.55, 0, 0.52), above the space between the two.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.sgs_assembly.env_cfg import make_assembly_env_cfg
from mjlab.tasks.sgs_assembly.mdp import observations, rewards, terminations
from mjlab.tasks.sgs_assembly.mdp.success import InsertionGatesCfg
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import (
  PegHoleCfg,
  hole_entity_cfg,
  peg_entity_cfg,
)
from mjlab.tasks.sgs_assembly.physics_cfg import PhysicsCfg
from mjlab.tasks.sgs_assembly.robots.franka.franka_constants import FR3_INFO

HOLE_POS = (0.55, 0.15, 0.0)
PEG_XY = (0.55, -0.15)
# Simple peg reset (until the reset deck exists): offsets around PEG_XY and yaw.
PEG_RESET_RANGE = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)}
# Held-object workspace box for the failure term, env frame [m]. The lower z bound
# catches a peg pushed through the table.
WORKSPACE_LOWER = (0.0, -0.6, -0.05)
WORKSPACE_UPPER = (1.0, 0.6, 0.8)


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.episode_length_s = int(1e9)


def fr3_robot_only_env_cfg(
  play: bool = False, physics: PhysicsCfg | None = None
) -> ManagerBasedRlEnvCfg:
  """Just the robot on the table: for checking the arm, gripper and gains."""
  cfg = make_assembly_env_cfg(FR3_INFO, physics or PhysicsCfg())
  if play:
    _apply_play_overrides(cfg)
  return cfg


def fr3_peg_hole_env_cfg(
  play: bool = False,
  physics: PhysicsCfg | None = None,
  peg_hole: PegHoleCfg | None = None,
  gates: InsertionGatesCfg | None = None,
  weights: rewards.SgsRewardWeights | None = None,
) -> ManagerBasedRlEnvCfg:
  physics = physics or PhysicsCfg()
  peg_hole = peg_hole or PegHoleCfg()
  gates = gates or InsertionGatesCfg()
  weights = weights or rewards.SgsRewardWeights()
  peg_pos = (*PEG_XY, peg_hole.peg_length / 2)  # Standing on the table.
  cfg = make_assembly_env_cfg(
    FR3_INFO,
    physics,
    objects={
      "peg": peg_entity_cfg(peg_hole, physics.contact, pos=peg_pos),
      "hole": hole_entity_cfg(peg_hole, physics.contact, pos=HOLE_POS),
    },
    weights=weights,
  )

  # Observations: the scene in the gripper frame (stand-in for the point cloud),
  # plus the peg relative to the hole. Same terms for actor and critic.
  ee_cfg = SceneEntityCfg("robot", site_names=(FR3_INFO.ee_site,))
  scene_terms = {
    "peg_in_ee": ObservationTermCfg(
      func=observations.entity_pose_in_ee,
      params={"entity_name": "peg", "robot_cfg": ee_cfg},
    ),
    "hole_in_ee": ObservationTermCfg(
      func=observations.entity_pose_in_ee,
      params={"entity_name": "hole", "robot_cfg": ee_cfg},
    ),
    "peg_in_hole": ObservationTermCfg(
      func=observations.entity_pose_in_entity,
      params={"entity_name": "peg", "frame_entity_name": "hole"},
    ),
  }
  for group in ("actor", "critic"):
    cfg.observations[group].terms.update(scene_terms)

  # Success S(s): terminal, rewarded with +weights.success.
  insertion = {
    "peg_half_length": peg_hole.peg_length / 2,
    "hole_depth": peg_hole.hole_depth,
    "gates": gates,
  }
  cfg.terminations["success"] = TerminationTermCfg(
    func=terminations.success, params=insertion
  )
  cfg.rewards["success"] = RewardTermCfg(
    func=rewards.terminated_by,
    weight=weights.success,
    params={"term_names": ("success",)},
  )

  # Failure F(s): also the held object leaving the workspace.
  cfg.terminations["peg_out_of_bounds"] = TerminationTermCfg(
    func=terminations.object_out_of_bounds,
    params={
      "entity_name": "peg",
      "lower": WORKSPACE_LOWER,
      "upper": WORKSPACE_UPPER,
    },
  )
  cfg.rewards["failure"].params["term_names"] += ("peg_out_of_bounds",)

  cfg.events["reset_peg_pose"].params["pose_range"] = dict(PEG_RESET_RANGE)

  if play:
    _apply_play_overrides(cfg)
  return cfg
