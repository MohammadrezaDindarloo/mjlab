"""Success predicate S(s) for peg-in-hole insertion (SGS App. A.3.2).

All quantities are computed in the hole frame (origin on the table at the hole
axis, +z up; see objects/peg_hole/peg_hole.py):

  u      peg axis (unit vector),       c  peg centre,      h  peg half-length
  e      the peg end that is lower:    e = c - h*u if u_z >= 0 else c + h*u
  tilt   angle between peg axis and hole axis, sign-free:  acos(|u_z|)
  lateral  distance of e from the hole axis:              ||(e_x, e_y)||
  depth    how far e is below the hole entrance:          hole_depth - e_z

S = (tilt <= max_tilt) and (lateral <= max_lateral) and (depth >= min_depth).

Symmetry (paper App. C.3): the peg is identical end-to-end and has a square
cross-section, so either end may go in first and yaw about the axis is not gated.
A fully seated peg has e_z = 0, i.e. depth = hole_depth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply, subtract_frame_transforms

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(frozen=True)
class InsertionGatesCfg:
  max_tilt_rad: float = math.radians(5.0)
  max_lateral: float = 0.003  # [m]
  min_depth: float = 0.025  # [m]; the default hole is 0.03 m deep.


def insertion_state(
  env: ManagerBasedRlEnv,
  peg_half_length: float,
  hole_depth: float,
  peg_name: str = "peg",
  hole_name: str = "hole",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Returns (tilt [rad], lateral [m], depth [m]), each of shape (num_envs,)."""
  peg = env.scene[peg_name].data
  hole = env.scene[hole_name].data
  c, q = subtract_frame_transforms(
    hole.root_link_pos_w,
    hole.root_link_quat_w,
    peg.root_link_pos_w,
    peg.root_link_quat_w,
  )
  z = torch.zeros_like(c)
  z[:, 2] = 1.0
  u = quat_apply(q, z)  # Peg axis in the hole frame.
  sign = torch.where(u[:, 2:3] >= 0.0, 1.0, -1.0)
  e = c - sign * peg_half_length * u  # Lower peg end.
  tilt = torch.acos(torch.clamp(u[:, 2].abs(), max=1.0))
  lateral = torch.linalg.norm(e[:, :2], dim=1)
  depth = hole_depth - e[:, 2]
  return tilt, lateral, depth


def insertion_success(
  env: ManagerBasedRlEnv,
  peg_half_length: float,
  hole_depth: float,
  gates: InsertionGatesCfg,
) -> torch.Tensor:
  tilt, lateral, depth = insertion_state(env, peg_half_length, hole_depth)
  return (
    (tilt <= gates.max_tilt_rad)
    & (lateral <= gates.max_lateral)
    & (depth >= gates.min_depth)
  )
