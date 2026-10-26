"""PPO runner that also checkpoints the deck sampler's memory.

SGS remembers, per deck entry, its last H episode outcomes; these decide which
starts are practised. RSL-RL checkpoints only hold the networks and optimiser,
so a resumed SGS run would restart with empty histories (sampling uniformly
again), unlike the paper's uninterrupted SGS. This runner adds the sampler state
to every checkpoint (``infos["deck_sampler"]``) and restores it on load. On
multiple GPUs every GPU loads the same checkpoint, so all samplers stay equal.
"""

from __future__ import annotations

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.sgs_assembly.mdp.events import reset_from_deck

DECK_TERM = "reset_from_deck"


class SgsOnPolicyRunner(MjlabOnPolicyRunner):
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
      term.sampler.load_state_dict(infos["deck_sampler"])
      n = int(term.sampler.count.sum())
      print(f"[SGS] restored deck sampler memory: {n} recorded outcomes")
    return infos
