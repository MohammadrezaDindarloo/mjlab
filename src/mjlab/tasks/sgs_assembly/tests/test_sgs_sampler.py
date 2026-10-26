"""SGS sampler maths (CPU only), against the paper's Eq. 1-2 and App. B.2.

uv run pytest src/mjlab/tasks/sgs_assembly/tests/test_sgs_sampler.py
"""

import math

import pytest
import torch

from mjlab.tasks.sgs_assembly.sampling.sampler import DeckSampler, SgsCfg

CFG = SgsCfg()


def _sampler(n: int, mode: str = "sgs") -> DeckSampler:
  return DeckSampler(n, mode, CFG, device="cpu")  # type: ignore[arg-type]


def _set_rates(s: DeckSampler, rates: list[float]) -> None:
  """Give configuration i exactly rates[i] * H successes out of H episodes."""
  h = CFG.history
  for i, r in enumerate(rates):
    k = round(r * h)
    s.record(torch.full((h,), i), torch.arange(h) < k)


def test_untried_configurations_are_sampled_uniformly():
  s = _sampler(5)
  assert torch.allclose(s.probabilities(), torch.full((5,), 0.2))


def test_probabilities_match_the_paper_formula():
  rates = [0.0, 0.1, 0.5, 0.9, 1.0]
  s = _sampler(len(rates))
  _set_rates(s, rates)
  eps, k, t, temp = CFG.eps, CFG.concentration, CFG.target, CFG.temperature
  logits = []
  for p in rates:
    w = (p + eps) ** (k * t) * (1 - p + eps) ** (k * (1 - t))
    logits.append(math.log(max(w, eps) + eps) / max(temp, 1.0))
  z = sum(math.exp(x) for x in logits)
  expected = torch.tensor([math.exp(x) / z for x in logits])
  assert torch.allclose(s.probabilities(), expected.float(), atol=1e-6)


def test_half_solved_is_favoured_but_nothing_is_starved():
  s = _sampler(3)
  _set_rates(s, [0.0, 0.5, 1.0])
  p = s.probabilities()
  assert p[1] > p[0] and p[1] > p[2]
  assert p[0] == pytest.approx(p[2].item(), rel=1e-4)  # t = 0.5 is symmetric.
  assert p[1] / p[0] == pytest.approx(7.0, rel=0.05)  # ~7x, see the docstring.
  assert p.min() > 0


def test_history_keeps_only_the_last_h_outcomes():
  s = _sampler(1)
  h = CFG.history
  s.record(torch.zeros(h, dtype=torch.long), torch.ones(h, dtype=torch.bool))
  assert s.success_rate()[0] == 1.0
  # h/2 failures overwrite the oldest h/2 successes.
  s.record(torch.zeros(h // 2, dtype=torch.long), torch.zeros(h // 2, dtype=torch.bool))
  assert s.success_rate()[0] == pytest.approx(0.5)
  assert s.count[0] == h


def test_duplicate_configurations_in_one_call_get_separate_slots():
  s = _sampler(2)
  s.record(torch.tensor([1, 0, 1, 1]), torch.tensor([True, False, False, True]))
  assert s.count.tolist() == [1, 3]
  assert s.success_rate()[1] == pytest.approx(2 / 3)


def test_uniform_mode_ignores_success_but_still_records_it():
  s = _sampler(3, mode="uniform")
  _set_rates(s, [0.0, 0.5, 1.0])
  assert torch.allclose(s.probabilities(), torch.full((3,), 1 / 3))
  assert s.success_rate().tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_state_dict_round_trip_restores_the_memory():
  s = _sampler(3)
  _set_rates(s, [0.0, 0.5, 1.0])
  restored = _sampler(3)
  restored.load_state_dict(s.state_dict())
  assert torch.equal(restored.probabilities(), s.probabilities())
  assert torch.equal(restored.cursor, s.cursor) and torch.equal(restored.count, s.count)
