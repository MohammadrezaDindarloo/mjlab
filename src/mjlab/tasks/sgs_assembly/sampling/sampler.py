"""Deck samplers: uniform (baseline) and Success Guided Sampling (SGS).

SGS (paper Sec. 3.2, Eq. 1-2, App. B.2), for each of the N deck configurations i:

  history   last H Boolean outcomes (1 = the episode started from i succeeded)
  p_i       mean of that history                       (0 if i was never tried)
  w_i     = (p_i + eps)^(k t) * (1 - p_i + eps)^(k (1 - t))     Beta-shaped score
  w~_i    = max(w_i, eps)                                       floor
  l_i     = log(w~_i + eps)                                     logit
  P(i)    = softmax(l / T_eff),   T_eff = max(T, 1)             next configuration

w_i peaks at p_i = t: configurations solved about half the time are practised
most, while every configuration keeps a nonzero probability (so its estimate keeps
updating). With the paper's manipulation values (H=100, t=0.5, k=1, T=2,
eps=1e-4), a configuration with p = 0.5 is drawn ~7x more often than one with
p = 0 or p = 1. Untried configurations all have p = 0, so sampling starts uniform
(paper Fig. 7: "Early in training, SGS samples uniformly").

Both samplers keep the same outcome history, so a uniform run logs the same
success statistics as an SGS run; only ``probabilities`` differs.
"""

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class SgsCfg:
  """Paper values for manipulation (Sec. 3.2)."""

  history: int = 100  # H
  target: float = 0.5  # t, the success rate sampled most
  concentration: float = 1.0  # kappa, how sharply around t
  temperature: float = 2.0  # T
  eps: float = 1e-4  # epsilon


class DeckSampler:
  """Chooses deck configurations and records their episode outcomes."""

  def __init__(
    self,
    num_configs: int,
    mode: Literal["uniform", "sgs"],
    cfg: SgsCfg,
    device: str,
  ) -> None:
    self.n, self.mode, self.cfg = num_configs, mode, cfg
    h = cfg.history
    # Circular buffer of the last H outcomes per configuration.
    self.outcomes = torch.zeros(num_configs, h, dtype=torch.bool, device=device)
    self.cursor = torch.zeros(num_configs, dtype=torch.long, device=device)
    self.count = torch.zeros(num_configs, dtype=torch.long, device=device)

  def success_rate(self) -> torch.Tensor:
    """p_i: mean of the recorded outcomes (unused buffer slots are False)."""
    return self.outcomes.sum(dim=1) / self.count.clamp(min=1)

  def probabilities(self) -> torch.Tensor:
    """P(i) for the next draw."""
    if self.mode == "uniform":
      return torch.full((self.n,), 1.0 / self.n, device=self.outcomes.device)
    c = self.cfg
    p = self.success_rate()
    w = (p + c.eps) ** (c.concentration * c.target) * (1 - p + c.eps) ** (
      c.concentration * (1 - c.target)
    )
    logits = torch.log(torch.clamp(w, min=c.eps) + c.eps)
    return torch.softmax(logits / max(c.temperature, 1.0), dim=0)

  def sample(self, n: int) -> torch.Tensor:
    """n configuration indices, drawn i.i.d. from P."""
    return torch.multinomial(self.probabilities(), n, replacement=True)

  def record(self, configs: torch.Tensor, success: torch.Tensor) -> None:
    """Append one outcome per finished episode to its configuration's history.

    Several episodes in one call may share a configuration; each gets its own
    buffer slot (k-th duplicate -> cursor + k).
    """
    if len(configs) == 0:
      return
    order = torch.argsort(configs, stable=True)
    rows, ok = configs[order], success[order]
    # Rank of each entry among equal configurations: 0, 1, 2, ...
    new_group = torch.ones_like(rows, dtype=torch.bool)
    new_group[1:] = rows[1:] != rows[:-1]
    idx = torch.arange(len(rows), device=rows.device)
    group_start = torch.cummax(torch.where(new_group, idx, 0), dim=0).values
    rank = idx - group_start
    h = self.cfg.history
    self.outcomes[rows, (self.cursor[rows] + rank) % h] = ok
    added = torch.zeros_like(self.count).index_add_(0, rows, torch.ones_like(rows))
    self.cursor = (self.cursor + added) % h
    self.count = torch.clamp(self.count + added, max=h)
