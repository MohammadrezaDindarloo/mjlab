"""(Later) Samplers that pick which reset-deck entry starts each episode.

Planned: one interface, ``sample(n) -> deck indices`` plus ``update(indices,
success)``, with two implementations:
- uniform sampling (baseline),
- Success Guided Sampling (SGS): last H=100 success bits per entry, score
  w = (p+eps)^(k*t) * (1-p+eps)^(k*(1-t)), floor max(w, eps), logits log(w+eps),
  softmax at temperature T (manipulation: t=0.5, k=1, T=2, eps=1e-4).
An optional Beta prior per entry (from FARO labels, see ``feasibility/``) seeds p.
"""
