"""FR3 + peg-in-hole task variants.

Layout (env frame, metres): robot base at the origin, hole fixture at
``HOLE_POS``, peg standing upright at ``PEG_POS``. The FR3 home TCP is at about
(0.55, 0, 0.52), above the space between the two.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.tasks.sgs_assembly.env_cfg import make_assembly_env_cfg
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import (
  PegHoleCfg,
  hole_entity_cfg,
  peg_entity_cfg,
)
from mjlab.tasks.sgs_assembly.physics_cfg import PhysicsCfg
from mjlab.tasks.sgs_assembly.robots.franka.franka_constants import FR3_INFO

HOLE_POS = (0.55, 0.15, 0.0)
PEG_XY = (0.55, -0.15)


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
) -> ManagerBasedRlEnvCfg:
  physics = physics or PhysicsCfg()
  peg_hole = peg_hole or PegHoleCfg()
  peg_pos = (*PEG_XY, peg_hole.peg_length / 2)  # Standing on the table.
  cfg = make_assembly_env_cfg(
    FR3_INFO,
    physics,
    objects={
      "peg": peg_entity_cfg(peg_hole, physics.contact, pos=peg_pos),
      "hole": hole_entity_cfg(peg_hole, physics.contact, pos=HOLE_POS),
    },
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg
