"""FR3 + peg-in-hole task variants.

Layout (env frame, metres): robot base at the origin, hole fixture at
``HOLE_POS``, peg standing upright around ``PEG_XY``. The FR3 home TCP is at about
(0.55, 0, 0.52), above the space between the two.
"""

import math
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.sgs_assembly.env_cfg import make_assembly_env_cfg
from mjlab.tasks.sgs_assembly.mdp import (
  events,
  observations,
  point_cloud,
  rewards,
  terminations,
)
from mjlab.tasks.sgs_assembly.mdp.success import InsertionGatesCfg
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import (
  PegHoleCfg,
  hole_entity_cfg,
  peg_entity_cfg,
)
from mjlab.tasks.sgs_assembly.physics_cfg import PhysicsCfg
from mjlab.tasks.sgs_assembly.robots.franka.franka_constants import FR3_INFO
from mjlab.tasks.sgs_assembly.sampling.sampler import SgsCfg
from mjlab.utils.noise.noise_cfg import GaussianNoiseCfg

ObsMode = Literal["pointcloud", "pose"]

HOLE_POS = (0.55, 0.15, 0.0)
PEG_XY = (0.55, -0.15)
# Simple peg reset (until the reset deck exists): offsets around PEG_XY and yaw.
PEG_RESET_RANGE = {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)}
# Held-object workspace box for the failure term, env frame [m]. The lower z bound
# catches a peg pushed through the table.
WORKSPACE_LOWER = (0.0, -0.6, -0.05)
WORKSPACE_UPPER = (1.0, 0.6, 0.8)
# Default reset deck, built by scripts/build_deck.py (path relative to the repo
# root, like mjlab's logs/; not committed).
DEFAULT_DECK_PATH = "logs/sgs_decks/fr3_peg_hole.npz"
POINT_CLOUD_NOISE_STD = 0.001  # [m] per point per step (paper App. C.4).


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
  obs_mode: ObsMode = "pointcloud",
) -> ManagerBasedRlEnvCfg:
  """``obs_mode``: "pointcloud" (paper App. A.3.2 / C.3: the actor sees the scene
  only as the point cloud; the critic also gets exact poses) or "pose" (actor
  gets the peg/hole poses too; for debugging and comparison)."""
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

  # Exact peg/hole poses in the gripper frame and peg-in-hole. Privileged for the
  # critic (asymmetric actor-critic, paper App. C.3); the actor gets them only in
  # "pose" mode, otherwise it sees the scene through the point cloud.
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
  cfg.observations["critic"].terms.update(scene_terms)
  if obs_mode == "pose":
    cfg.observations["actor"].terms.update(scene_terms)

  # Scene point cloud in the gripper frame (paper App. A.3.2), (N, 320, 6):
  # xyz with 1 mm Gaussian noise during training (App. C.4) + part label.
  cfg.observations["point_cloud"] = ObservationGroupCfg(
    terms={
      "xyz": ObservationTermCfg(
        func=point_cloud.scene_point_cloud,
        noise=GaussianNoiseCfg(std=POINT_CLOUD_NOISE_STD),
      ),
      "label": ObservationTermCfg(func=point_cloud.point_cloud_labels),
    },
    concatenate_terms=True,
    concatenate_dim=-1,
    enable_corruption=not play,
  )

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


def fr3_peg_hole_deck_env_cfg(
  play: bool = False,
  deck_path: str = DEFAULT_DECK_PATH,
  families: tuple[str, ...] | None = None,
  sampler: Literal["uniform", "sgs"] = "uniform",
  sgs: SgsCfg | None = None,
  **kwargs,
) -> ManagerBasedRlEnvCfg:
  """Peg-in-hole where every episode starts from a reset-deck entry.

  Replaces the simple resets (robot at home, upright random peg) by
  ``reset_from_deck``; robot-base and hole placement events stay. ``families``
  (e.g. ("NEAR_GOAL",)) limits the starts to those deck families. ``sampler``
  picks the entries: "uniform" (paper baseline) or "sgs" (Success Guided
  Sampling, with the paper's values in ``SgsCfg`` unless ``sgs`` is given).
  """
  cfg = fr3_peg_hole_env_cfg(play=play, **kwargs)
  del cfg.events["reset_peg_pose"], cfg.events["reset_robot_joints"]
  cfg.events["reset_from_deck"] = EventTermCfg(
    func=events.reset_from_deck,
    mode="reset",
    params={
      "deck_path": deck_path,
      "robot_name": "robot",
      "peg_name": "peg",
      "arm_joints": FR3_INFO.arm_joints,
      "finger_joints": ("finger_joint1", "finger_joint2"),
      "families": families,
      "sampler": sampler,
      "sgs": sgs or SgsCfg(),
    },
  )
  # Every step on every GPU: merge outcomes across GPUs, log family statistics.
  cfg.events["sync_deck_outcomes"] = EventTermCfg(
    func=events.sync_deck_outcomes, mode="step"
  )
  return cfg
