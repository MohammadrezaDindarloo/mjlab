"""Reset events.

``reset_from_deck`` starts each episode from one row of a reset deck
(resets/deck.py): it writes the arm joints, both finger joints and the peg pose,
all with zero velocity. The action terms reset *after* events, so the arm target
and gripper target start at exactly the deck state (nothing jumps on step 1).

Which row each env gets is chosen uniformly for now; the SGS sampler (sampling/)
will replace ``_choose``. ``self.index[env]`` remembers the row each env started
from, which is what the sampler will need to credit success to a deck entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.sgs_assembly.resets.deck import ResetDeck

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class reset_from_deck(ManagerTermBase):
  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
    super().__init__(env)
    p = cfg.params
    deck = ResetDeck.load(p["deck_path"])
    dev = env.device
    self.arm_q = torch.tensor(deck.arm_q, device=dev)
    self.finger_q = torch.tensor(deck.finger_q, device=dev)
    self.peg_pos = torch.tensor(deck.peg_pos, device=dev)
    self.peg_quat = torch.tensor(deck.peg_quat, device=dev)
    self.family = torch.tensor(deck.family, device=dev, dtype=torch.long)
    self.size = len(deck)
    self.index = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    robot = env.scene[p["robot_name"]]
    arm_ids, _ = robot.find_joints(p["arm_joints"], preserve_order=True)
    finger_ids, _ = robot.find_joints(p["finger_joints"], preserve_order=True)
    self.arm_ids = torch.tensor(arm_ids, device=dev)
    self.finger_ids = torch.tensor(finger_ids, device=dev)

  def _choose(self, n: int) -> torch.Tensor:
    return torch.randint(self.size, (n,), device=self.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    deck_path: str,
    robot_name: str,
    peg_name: str,
    arm_joints: tuple[str, ...],
    finger_joints: tuple[str, ...],
  ) -> None:
    del deck_path, arm_joints, finger_joints  # Used in __init__.
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    rows = self._choose(len(env_ids))
    self.index[env_ids] = rows
    n = len(env_ids)

    robot = env.scene[robot_name]
    joint_ids = torch.cat([self.arm_ids, self.finger_ids])
    pos = torch.cat(
      [self.arm_q[rows], self.finger_q[rows, None].expand(n, len(self.finger_ids))],
      dim=1,
    )
    robot.write_joint_state_to_sim(
      pos, torch.zeros_like(pos), joint_ids=joint_ids, env_ids=env_ids
    )

    root = torch.zeros(n, 13, device=env.device)
    root[:, :3] = self.peg_pos[rows] + env.scene.env_origins[env_ids]
    root[:, 3:7] = self.peg_quat[rows]
    env.scene[peg_name].write_root_state_to_sim(root, env_ids=env_ids)
