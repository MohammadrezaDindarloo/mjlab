"""The reset deck: a fixed buffer of start states, saved as one .npz file.

One row = one start state (SGS "task configuration", paper Sec. 3.1 / App. A.2):

  arm_q     (N, 7)  arm joint positions [rad]
  finger_q  (N,)    opening of EACH finger [m] (both fingers are set to it)
  peg_pos   (N, 3)  peg centre in the env frame (robot base at the origin) [m]
  peg_quat  (N, 4)  peg orientation in the env frame, (w, x, y, z)
  family    (N,)    which generator made it (``Family``)

Velocities are zero for every entry. The hole pose is fixed by the task config
and stored in ``meta`` for reference. This file is the exchange format with
offline tools (FARO labels are matched to rows by index), so it only holds plain
arrays plus a JSON ``meta`` dict describing the geometry it was generated for.
"""

import json
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np


class Family(IntEnum):
  NEAR_GOAL = 0
  STABLE_GRASP = 1
  REACHING = 2


@dataclass
class ResetDeck:
  arm_q: np.ndarray
  finger_q: np.ndarray
  peg_pos: np.ndarray
  peg_quat: np.ndarray
  family: np.ndarray
  meta: dict = field(default_factory=dict)

  def __post_init__(self) -> None:
    n = len(self.arm_q)
    assert self.arm_q.shape == (n, 7), self.arm_q.shape
    assert self.finger_q.shape == (n,), self.finger_q.shape
    assert self.peg_pos.shape == (n, 3), self.peg_pos.shape
    assert self.peg_quat.shape == (n, 4), self.peg_quat.shape
    assert self.family.shape == (n,), self.family.shape

  def __len__(self) -> int:
    return len(self.arm_q)

  def save(self, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
      path,
      arm_q=self.arm_q.astype(np.float32),
      finger_q=self.finger_q.astype(np.float32),
      peg_pos=self.peg_pos.astype(np.float32),
      peg_quat=self.peg_quat.astype(np.float32),
      family=self.family.astype(np.int8),
      meta=np.array(json.dumps(self.meta)),
    )

  @classmethod
  def load(cls, path: str | Path) -> "ResetDeck":
    with np.load(path) as f:
      return cls(
        arm_q=f["arm_q"],
        finger_q=f["finger_q"],
        peg_pos=f["peg_pos"],
        peg_quat=f["peg_quat"],
        family=f["family"],
        meta=json.loads(str(f["meta"])),
      )

  @classmethod
  def concat(cls, decks: list["ResetDeck"]) -> "ResetDeck":
    return cls(
      arm_q=np.concatenate([d.arm_q for d in decks]),
      finger_q=np.concatenate([d.finger_q for d in decks]),
      peg_pos=np.concatenate([d.peg_pos for d in decks]),
      peg_quat=np.concatenate([d.peg_quat for d in decks]),
      family=np.concatenate([d.family for d in decks]),
      meta=decks[0].meta,
    )
