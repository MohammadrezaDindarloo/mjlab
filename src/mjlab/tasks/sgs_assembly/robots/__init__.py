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
  arm_joints: tuple[str, ...]
  """Regexes for the arm joints, in joint order."""
  arm_joint_vel_limits: tuple[float, ...]
  """Hardware velocity limit per arm joint [rad/s], same order as ``arm_joints``
  (MuJoCo does not store these)."""
  arm_action_scale: float
  """Joint-target change per policy step for action 1 [rad]."""
  gripper_joint: str
  """The actuated gripper joint."""
  gripper_open: float
  """Gripper joint target for 'open'."""
  gripper_closed: float
  """Gripper joint target for 'close'."""
  ee_site: str
  """Tool-centre-point site between the fingertips."""
  pad_geoms: str
  """Regex for the fingertip pad geoms (friction target, contact logging)."""
  root_body: str
  """Robot base body, used to place the camera."""
