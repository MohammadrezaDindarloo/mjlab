"""Grasp frames on the square peg: where the TCP may be, relative to the peg.

A grasp frame is a TCP pose in the peg frame. The hand axes (TCP site frame) are
  x_hand: along the pad width,  y_hand: finger sliding direction (pad normal),
  z_hand: approach direction (palm -> fingertips).
The pads squeeze along y_hand, so y_hand must be a face normal of the peg (+-x_p
or +-y_p). Two kinds of grasp:
  side grasp: approach along a face normal, peg axis along x_hand; the TCP can
              sit anywhere along the peg axis where the pad still fits;
  end grasp:  approach along the peg axis into one end, TCP ``END_GRASP_DEPTH``
              inside that end (the grasp the scripted demo uses).

Frames are returned in groups of *equivalent* grasps: same TCP position and
approach direction, differing only by a rotation of the hand about the approach
axis that the peg's symmetry makes irrelevant (side: 180 deg flip, 2 members;
end: 90 deg steps, 4 members). A generator picks a group (which contact) and then
whichever member suits the robot best (e.g. least wrist rotation).
"""

import numpy as np

from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg

PAD_HALF_WIDTH = 0.00875  # [m] Franka Hand pad half-extent along x_hand.
END_GRASP_DEPTH = 0.02  # [m] TCP this far inside the peg end for an end grasp.

_AXES = np.eye(3)

Grasp = tuple[np.ndarray, np.ndarray]  # (TCP position, TCP rotation), peg frame.


def _frame(z_hand: np.ndarray, y_hand: np.ndarray) -> np.ndarray:
  """Rotation matrix with columns (x_hand, y_hand, z_hand), right-handed."""
  return np.stack([np.cross(y_hand, z_hand), y_hand, z_hand], axis=1)


def grasp_groups(cfg: PegHoleCfg, rng: np.random.Generator) -> list[list[Grasp]]:
  """Groups of equivalent grasp frames in the peg frame (6 groups, 16 frames).

  Each side-grasp group gets one random position along the peg axis.
  """
  groups: list[list[Grasp]] = []
  half_l = cfg.peg_length / 2
  for b in (0, 1):  # Side grasp approaching along +-axis b, squeezing along a.
    a = 1 - b
    for s in (1.0, -1.0):
      z_hand = s * _AXES[b]
      along = rng.uniform(-(half_l - PAD_HALF_WIDTH), half_l - PAD_HALF_WIDTH)
      pos = along * _AXES[2]
      groups.append([(pos, _frame(z_hand, sy * _AXES[a])) for sy in (1.0, -1.0)])
  for s in (1.0, -1.0):  # End grasp into the end at s * L/2, pointing inwards.
    z_hand = -s * _AXES[2]
    pos = s * (half_l - END_GRASP_DEPTH) * _AXES[2]
    groups.append(
      [(pos, _frame(z_hand, sy * _AXES[a])) for a in (0, 1) for sy in (1.0, -1.0)]
    )
  return groups
