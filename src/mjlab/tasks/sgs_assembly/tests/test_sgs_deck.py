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


def test_stable_grasp_states_hold_the_peg_above_the_hole(builder):
  deck = builder.stable_grasp(10, np.random.default_rng(0))
  assert (deck.family == Family.STABLE_GRASP).all()
  half = np.array([PEG.peg_width / 2] * 2 + [PEG.peg_length / 2])
  for i in range(len(deck)):
    builder.set_state(
      deck.arm_q[i], [deck.finger_q[i]] * 2, deck.peg_pos[i], deck.peg_quat[i]
    )
    assert builder.is_valid()
    rot = quat_to_mat(deck.peg_quat[i])
    tcp, _ = builder.ik.forward(deck.arm_q[i])
    in_peg = rot.T @ (tcp - deck.peg_pos[i])
    # TCP inside the peg, in the upper half (end grasp on the top end).
    assert np.all(np.abs(in_peg) <= half + 1e-4), in_peg
    assert in_peg[2] * np.sign(rot[2, 2]) > 0
    # Lower peg end above the hole entrance, fingers just wider than the peg.
    tip_z = deck.peg_pos[i][2] - PEG.peg_length / 2 * abs(rot[2, 2])
    assert tip_z > builder.hole_pos[2] + PEG.hole_depth
    assert PEG.peg_width / 2 < deck.finger_q[i] < PEG.peg_width / 2 + 0.0031


def test_near_goal_states_lie_on_the_insertion_path(builder):
  deck = builder.near_goal(10, np.random.default_rng(0))
  assert (deck.family == Family.NEAR_GOAL).all()
  for i in range(len(deck)):
    builder.set_state(
      deck.arm_q[i], [deck.finger_q[i]] * 2, deck.peg_pos[i], deck.peg_quat[i]
    )
    assert builder.is_valid()
    rot = quat_to_mat(deck.peg_quat[i])
    assert abs(rot[2, 2]) == pytest.approx(1.0, abs=1e-6)  # Vertical.
    tip = deck.peg_pos[i] - PEG.peg_length / 2 * np.array([0.0, 0.0, 1.0])
    offset = tip - builder.hole_pos
    assert np.all(np.abs(offset[:2]) <= 0.0003 + 1e-9)  # On the hole axis.
    assert -1e-9 <= offset[2] / PEG.hole_depth <= 1.1 + 1e-9  # Path fraction.
