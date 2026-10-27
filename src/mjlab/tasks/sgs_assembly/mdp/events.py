"""Reset events: start every episode from a reset-deck entry, chosen by a sampler.

``reset_from_deck`` (mode "reset") does two things for the envs being reset:
  1. record: the episode that just ended credits its outcome (1 = it ended by the
     success termination, 0 = time-out or failure) to the deck entry it started
     from, in the sampler's history (sampling/sampler.py);
  2. choose: draw the next deck entry from the sampler ("uniform" baseline or
     "sgs") and write that state: arm joints, both fingers, peg pose, all at rest.
The action terms reset *after* events, so the arm and gripper targets start at
exactly the deck state (nothing jumps on step 1).

``sync_deck_outcomes`` (mode "step", runs every step on every GPU) merges the
recorded outcomes across GPUs, so with multi-GPU training every GPU's sampler
holds the same histories, as if one process saw all episodes. It also logs
per-family statistics (to W&B via RSL-RL).

``families`` (e.g. ("REACHING",)) restricts the draw to those deck families,
e.g. to watch or evaluate one family; None uses the whole deck.

Hang report: ``sync_deck_outcomes`` also re-arms a timer every step. If a process
makes no step for ``HANG_REPORT_SECONDS``, Python writes the stack of every thread
to ``$SGS_HANG_DIR/<job>_rank<r>.txt``. With multi-GPU, a frozen GPU process
leaves the others waiting at the next sync until NCCL aborts the job (10 min), so
this file is the only record of where it froze.
"""

from __future__ import annotations

import faulthandler
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TextIO

import torch
import torch.distributed as dist

from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.manager_base import ManagerTermBase
from mjlab.tasks.sgs_assembly.resets.deck import Family, ResetDeck
from mjlab.tasks.sgs_assembly.sampling.sampler import DeckSampler, SgsCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


HANG_REPORT_SECONDS = 300.0
_hang_file: TextIO | None = None


def _multi_gpu() -> bool:
  return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _arm_hang_report() -> None:
  """(Re)start the no-progress timer; see the module docstring."""
  global _hang_file
  if _hang_file is None:
    folder = Path(os.environ.get("SGS_HANG_DIR", "logs/hang_reports"))
    folder.mkdir(parents=True, exist_ok=True)
    rank = dist.get_rank() if _multi_gpu() else 0
    job = os.environ.get("SLURM_JOB_ID", "local")
    _hang_file = open(folder / f"{job}_rank{rank}.txt", "a")  # noqa: SIM115
  faulthandler.dump_traceback_later(HANG_REPORT_SECONDS, file=_hang_file)


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
    allowed = torch.ones(self.size, dtype=torch.bool, device=dev)
    if p.get("families"):
      codes = torch.tensor([int(Family[f]) for f in p["families"]], device=dev)
      allowed = torch.isin(self.family, codes)
    # Deck rows the sampler chooses from; sampler index k <-> deck row rows[k].
    self.rows = allowed.nonzero().flatten()
    if len(self.rows) == 0:
      raise ValueError(f"No deck rows of families {p['families']} in {p['deck_path']}")
    self.sampler = DeckSampler(
      len(self.rows), p.get("sampler", "uniform"), p.get("sgs", SgsCfg()), dev
    )
    # Per env: sampler index and deck row of the episode it is running.
    self.config = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    self.index = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
    # Outcomes waiting for the cross-GPU merge (multi-GPU only).
    self.pending: list[tuple[torch.Tensor, torch.Tensor]] = []
    robot = env.scene[p["robot_name"]]
    arm_ids, _ = robot.find_joints(p["arm_joints"], preserve_order=True)
    finger_ids, _ = robot.find_joints(p["finger_joints"], preserve_order=True)
    self.arm_ids = torch.tensor(arm_ids, device=dev)
    self.finger_ids = torch.tensor(finger_ids, device=dev)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    deck_path: str,
    robot_name: str,
    peg_name: str,
    arm_joints: tuple[str, ...],
    finger_joints: tuple[str, ...],
    families: tuple[str, ...] | None = None,
    sampler: Literal["uniform", "sgs"] = "uniform",
    sgs: SgsCfg | None = None,
  ) -> None:
    del deck_path, arm_joints, finger_joints, families, sampler, sgs  # In __init__.
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)

    # 1. Record the outcome of the episodes that just ended. Envs that have not
    #    run a step yet (the very first reset) have no outcome to record.
    ran = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(ran) > 0:
      success = env.termination_manager.get_term("success")[ran]
      self._record(self.config[ran], success)

    # 2. Choose the next deck entries and write their states.
    n = len(env_ids)
    config = self.sampler.sample(n)
    rows = self.rows[config]
    self.config[env_ids] = config
    self.index[env_ids] = rows

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

  def _record(self, config: torch.Tensor, success: torch.Tensor) -> None:
    if _multi_gpu():
      self.pending.append((config, success))  # Merged in sync().
    else:
      self.sampler.record(config, success)

  def sync(self) -> None:
    """Multi-GPU: every GPU records every GPU's outcomes, in rank order."""
    if not _multi_gpu():
      return
    dev = self.device
    if self.pending:
      config = torch.cat([c for c, _ in self.pending])
      success = torch.cat([s for _, s in self.pending])
    else:
      config = torch.zeros(0, dtype=torch.long, device=dev)
      success = torch.zeros(0, dtype=torch.bool, device=dev)
    self.pending = []
    world = dist.get_world_size()
    # all_gather needs equal sizes: share the counts, pad to the largest.
    count = torch.tensor([len(config)], device=dev)
    counts = [torch.zeros_like(count) for _ in range(world)]
    dist.all_gather(counts, count)
    size = int(max(c.item() for c in counts))
    if size == 0:
      return
    padded = torch.full((size, 2), -1, dtype=torch.long, device=dev)
    padded[: len(config), 0] = config
    padded[: len(config), 1] = success.long()
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded)
    merged = torch.cat(gathered)
    merged = merged[merged[:, 0] >= 0]
    self.sampler.record(merged[:, 0], merged[:, 1].bool())

  def family_stats(self) -> dict[str, float]:
    """Per family: mean success rate of tried entries, and sampling share."""
    family = self.family[self.rows]
    p = self.sampler.success_rate()
    tried = self.sampler.count > 0
    prob = self.sampler.probabilities()
    stats: dict[str, float] = {}
    for f in Family:
      mask = family == int(f)
      if not mask.any():
        continue
      name = f.name.lower()
      seen = mask & tried
      if seen.any():
        stats[f"SGS/success_rate/{name}"] = p[seen].mean().item()
      stats[f"SGS/sample_share/{name}"] = prob[mask].sum().item()
    stats["SGS/entries_tried"] = tried.float().mean().item()
    return stats


def sync_deck_outcomes(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  term_name: str = "reset_from_deck",
  log_every: int = 24,
) -> None:
  """Step event: merge outcomes across GPUs; every ``log_every`` steps (one PPO
  rollout by default) add per-family statistics to the training log."""
  del env_ids
  _arm_hang_report()
  term = env.event_manager.get_term_cfg(term_name).func
  assert isinstance(term, reset_from_deck)
  term.sync()
  if env.common_step_counter % log_every == 0:
    env.extras["log"].update(term.family_stats())
