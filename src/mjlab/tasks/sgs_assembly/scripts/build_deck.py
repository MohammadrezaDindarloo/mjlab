"""Build a reset deck and save it as .npz (see resets/deck.py for the format).

  uv run python src/mjlab/tasks/sgs_assembly/scripts/build_deck.py
  uv run play Mjlab-SGS-PegHole-FR3-Deck --agent zero --viewer viser --num-envs 16

  uv run python src/mjlab/tasks/sgs_assembly/scripts/build_deck.py \
    --reaching 10922 --seed 0
  # -> logs/sgs_decks/fr3_peg_hole.npz, used by Mjlab-SGS-PegHole-FR3-Deck

Only the Reaching family exists so far; Near-Goal and Stable Grasp follow.
"""

import numpy as np
import tyro

from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import (
  DEFAULT_DECK_PATH,
  fr3_peg_hole_env_cfg,
)
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg
from mjlab.tasks.sgs_assembly.resets.deck import Family, ResetDeck
from mjlab.tasks.sgs_assembly.resets.generate import DeckBuilder


def main(reaching: int = 10922, seed: int = 0, out: str = DEFAULT_DECK_PATH) -> None:
  """Args:
  reaching: Number of Reaching states (paper: 32768 total over 3 families).
  seed: Random seed; the same seed gives the same deck.
  out: Output .npz path.
  """
  rng = np.random.default_rng(seed)
  builder = DeckBuilder(fr3_peg_hole_env_cfg(), PegHoleCfg())
  deck = builder.reaching(reaching, rng)
  deck.meta["seed"] = seed
  deck.save(out)
  counts = {f.name: int((deck.family == f).sum()) for f in Family}
  print(f"saved {len(deck)} states to {out}: {counts}")
  # Round-trip check.
  assert len(ResetDeck.load(out)) == len(deck)


if __name__ == "__main__":
  tyro.cli(main)
