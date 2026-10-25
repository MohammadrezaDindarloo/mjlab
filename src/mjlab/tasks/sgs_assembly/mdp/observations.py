"""Privileged-state observations (stand-in for the paper's point cloud).

The SGS teacher sees the scene *expressed in the gripper frame* (App. C.3). Until
point clouds exist we give the same information as poses: the peg and the hole in
the end-effector (TCP) frame, plus the end-effector pose in the robot base frame.

Every pose is a 9-vector [position (3), rotation (6)], where the rotation is the
first two columns of the rotation matrix. Unlike quaternions (q and -q are the
same rotation) or Euler angles, this is continuous, which is easier for a network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def pose_to_vec(pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
  """(N, 3) position + (N, 4) wxyz quaternion -> (N, 9) [pos, R[:, 0], R[:, 1]]."""
  rot = matrix_from_quat(quat)
  return torch.cat([pos, rot[:, :, 0], rot[:, :, 1]], dim=1)


def _ee_pose_w(
  env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  data = env.scene[robot_cfg.name].data
  site = robot_cfg.site_ids
  return data.site_pos_w[:, site].squeeze(1), data.site_quat_w[:, site].squeeze(1)


def ee_pose_in_base(env: ManagerBasedRlEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
  """End-effector (TCP site) pose in the robot base frame."""
  data = env.scene[robot_cfg.name].data
  ee_pos, ee_quat = _ee_pose_w(env, robot_cfg)
  pos, quat = subtract_frame_transforms(
    data.root_link_pos_w, data.root_link_quat_w, ee_pos, ee_quat
  )
  return pose_to_vec(pos, quat)


def entity_pose_in_ee(
  env: ManagerBasedRlEnv, entity_name: str, robot_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Pose of an entity's root frame (e.g. peg, hole) in the end-effector frame."""
  obj = env.scene[entity_name].data
  ee_pos, ee_quat = _ee_pose_w(env, robot_cfg)
  pos, quat = subtract_frame_transforms(
    ee_pos, ee_quat, obj.root_link_pos_w, obj.root_link_quat_w
  )
  return pose_to_vec(pos, quat)


def entity_pose_in_entity(
  env: ManagerBasedRlEnv, entity_name: str, frame_entity_name: str
) -> torch.Tensor:
  """Pose of one entity's root frame in another's (e.g. peg in hole frame)."""
  obj = env.scene[entity_name].data
  ref = env.scene[frame_entity_name].data
  pos, quat = subtract_frame_transforms(
    ref.root_link_pos_w, ref.root_link_quat_w, obj.root_link_pos_w, obj.root_link_quat_w
  )
  return pose_to_vec(pos, quat)
