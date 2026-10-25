"""Robots for the assembly tasks.

Each robot module exposes a ``RobotInfo``: everything the generic assembly env
config needs to know about a robot, so adding a robot never touches ``env_cfg.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass

from mjlab.entity import EntityCfg
from mjlab.tasks.sgs_assembly.physics_cfg import ContactCfg


@dataclass(frozen=True)
class RobotInfo:
  make_entity_cfg: Callable[[ContactCfg], EntityCfg]
  """Builds the robot entity, with contact parameters taken from ``ContactCfg``."""
  action_scale: dict[str, float]
  """Joint-position action scale per actuator regex: target = default + scale * a."""
  arm_joints: tuple[str, ...]
  """Regexes for the arm joints (used e.g. by the |qdot_arm|^2 penalty)."""
  gripper_joints: tuple[str, ...]
  """Regexes for the gripper joints."""
  joint_vel_limits: dict[str, float]
  """Hardware joint velocity limits [rad/s] (MuJoCo does not store these)."""
  ee_site: str
  """Tool-centre-point site between the fingertips."""
  pad_geoms: str
  """Regex for the fingertip pad geoms (friction target, contact logging)."""
  root_body: str
  """Robot base body, used to place the camera."""
