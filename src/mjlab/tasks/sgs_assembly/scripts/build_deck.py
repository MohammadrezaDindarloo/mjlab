"""Build the reset deck and save it as .npz (format: resets/deck.py).

The deck holds the three SGS start-state families (paper App. A.2):
  NEAR_GOAL     peg held and already on its insertion path (0-110 %)
  STABLE_GRASP  peg held firmly above the hole
  REACHING      peg anywhere, gripper around it, random opening (may drop);
                the paper evaluates on this family
Defaults give the paper's 32768 states, balanced (10922 + 10923 + 10923), with
seed 0, in ~3 min. The file goes to logs/sgs_decks/ (not in git; rebuild it).

1. Build the deck (run from the repo root)
  uv run python src/mjlab/tasks/sgs_assembly/scripts/build_deck.py

   Options: --reaching N --stable-grasp N --near-goal N (0 skips a family),
   --seed S, --out PATH (other decks need a task cfg pointing to them).

2. Look at the start states (viser; press Reset in the panel for new starts)
  # all three families mixed, as in training:
  uv run play Mjlab-SGS-PegHole-FR3-Deck --agent zero --viewer viser --num-envs 16
  # one family at a time (same deck file, only that family is drawn):
  uv run play Mjlab-SGS-PegHole-FR3-Deck-NearGoal --agent zero --viewer viser \
    --num-envs 16
  uv run play Mjlab-SGS-PegHole-FR3-Deck-StableGrasp --agent zero --viewer viser \
    --num-envs 16
  uv run play Mjlab-SGS-PegHole-FR3-Deck-Reaching --agent zero --viewer viser \
    --num-envs 16

   With --agent zero the arm holds still and the gripper closes on step 1.
   Near-Goal starts that are already >= 25 mm deep succeed at once and reset.

3. Check the deck code
  uv run pytest src/mjlab/tasks/sgs_assembly/tests/test_sgs_deck.py
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


def main(
  reaching: int = 10922,
  stable_grasp: int = 10923,
  near_goal: int = 10923,
  seed: int = 0,
  out: str = DEFAULT_DECK_PATH,
) -> None:
  """Args:
  reaching: Number of Reaching states (paper: 32768 total, balanced over 3).
  stable_grasp: Number of Stable Grasp states.
  near_goal: Number of Near-Goal states.
  seed: Random seed; the same seed gives the same deck.
  out: Output .npz path.
  """
  rng = np.random.default_rng(seed)
  builder = DeckBuilder(fr3_peg_hole_env_cfg(), PegHoleCfg())
  parts = []
  if reaching:
    parts.append(builder.reaching(reaching, rng))
  if stable_grasp:
    parts.append(builder.stable_grasp(stable_grasp, rng))
  if near_goal:
    parts.append(builder.near_goal(near_goal, rng))
  deck = ResetDeck.concat(parts)
  deck.meta["seed"] = seed
  deck.save(out)
  counts = {f.name: int((deck.family == f).sum()) for f in Family}
  print(f"saved {len(deck)} states to {out}: {counts}")
  # Round-trip check.
  assert len(ResetDeck.load(out)) == len(deck)


if __name__ == "__main__":
  tyro.cli(main)
