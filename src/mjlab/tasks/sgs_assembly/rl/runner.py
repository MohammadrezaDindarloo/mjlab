"""PPO runner that also checkpoints the deck sampler's memory.

SGS remembers, per deck entry, its last H episode outcomes; these decide which
starts are practised. RSL-RL checkpoints only hold the networks and optimiser,
so a resumed SGS run would restart with empty histories (sampling uniformly
again), unlike the paper's uninterrupted SGS. This runner adds the sampler state
to every checkpoint (``infos["deck_sampler"]``) and restores it on load. On
multiple GPUs every GPU loads the same checkpoint, so all samplers stay equal.

It also logs to W&B through ``ResumableWandbLogWriter``, so a restarted run
continues the same W&B curve (train.sh sets WANDB_RUN_ID and WANDB_RESUME).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import wandb
from rsl_rl.utils.wandb_log_writer import WandbLogWriter

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.sgs_assembly.mdp.events import reset_from_deck

DECK_TERM = "reset_from_deck"


class ResumableWandbLogWriter(WandbLogWriter):
  """RSL-RL's W&B writer, but a resumed run may store a changed config.

  A resumed run's config differs from the first one (resume flag, checkpoint,
  remaining iterations), and W&B refuses to change logged config values unless
  asked to explicitly.
  """

  def store_config(self, env_cfg: Any, train_cfg: dict) -> None:
    env = env_cfg.to_dict() if hasattr(env_cfg, "to_dict") else asdict(env_cfg)
    wandb.config.update({"train_cfg": train_cfg, "env_cfg": env}, allow_val_change=True)


class SgsOnPolicyRunner(MjlabOnPolicyRunner):
  def __init__(self, env, train_cfg: dict, *args, **kwargs) -> None:
    if train_cfg.get("logger") == "wandb":
      train_cfg["logger"] = {
        "class_name": f"{__name__}:ResumableWandbLogWriter",
        "project_name": train_cfg["wandb_project"],
      }
    super().__init__(env, train_cfg, *args, **kwargs)

  def _deck_term(self) -> reset_from_deck | None:
    events = self.env.unwrapped.event_manager
    if DECK_TERM not in events.active_terms.get("reset", []):
      return None
    term = events.get_term_cfg(DECK_TERM).func
    return term if isinstance(term, reset_from_deck) else None

  def save(self, path: str, infos=None) -> None:
    term = self._deck_term()
    if term is not None:
      infos = {**(infos or {}), "deck_sampler": term.sampler.state_dict()}
    super().save(path, infos)

  def load(self, path: str, load_cfg=None, strict=True, map_location=None) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    term = self._deck_term()
    if term is not None and infos and "deck_sampler" in infos:
      state = infos["deck_sampler"]
      if state["outcomes"].shape != term.sampler.outcomes.shape:
        # E.g. playing a full-deck checkpoint on one family's rows only.
        print("[SGS] deck rows differ from the checkpoint: sampler memory not restored")
        return infos
      term.sampler.load_state_dict(state)
      n = int(term.sampler.count.sum())
      print(f"[SGS] restored deck sampler memory: {n} recorded outcomes")
    return infos
