"""Reset-deck checks (CPU only).

uv run pytest src/mjlab/tasks/sgs_assembly/tests/test_sgs_deck.py
"""

import numpy as np
import pytest

from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import fr3_peg_hole_env_cfg
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg
from mjlab.tasks.sgs_assembly.resets.deck import Family, ResetDeck
from mjlab.tasks.sgs_assembly.resets.generate import DeckBuilder, quat_to_mat

PEG = PegHoleCfg()


@pytest.fixture(scope="module")
def builder() -> DeckBuilder:
  return DeckBuilder(fr3_peg_hole_env_cfg(), PEG)


@pytest.fixture(scope="module")
def deck(builder) -> ResetDeck:
  return builder.reaching(20, np.random.default_rng(0))


def test_reaching_states_are_valid_and_aligned_to_a_grasp(builder, deck):
  assert (deck.family == Family.REACHING).all()
  half = np.array([PEG.peg_width / 2] * 2 + [PEG.peg_length / 2])
  for i in range(len(deck)):
    builder.set_state(
      deck.arm_q[i], [deck.finger_q[i]] * 2, deck.peg_pos[i], deck.peg_quat[i]
    )
    assert builder.is_valid()
    # Every grasp frame puts the TCP inside the peg's box (between the pads).
    tcp, _ = builder.ik.forward(deck.arm_q[i])
    in_peg = quat_to_mat(deck.peg_quat[i]).T @ (tcp - deck.peg_pos[i])
    assert np.all(np.abs(in_peg) <= half + 1e-4), in_peg
    assert PEG.peg_width / 2 < deck.finger_q[i] <= 0.04


def test_deck_round_trip(deck, tmp_path):
  path = tmp_path / "deck.npz"
  deck.save(path)
  loaded = ResetDeck.load(path)
  np.testing.assert_allclose(loaded.arm_q, deck.arm_q, atol=1e-6)
  np.testing.assert_allclose(loaded.peg_quat, deck.peg_quat, atol=1e-6)
  assert loaded.meta["hole_pos"] == pytest.approx([0.55, 0.15, 0.0])
