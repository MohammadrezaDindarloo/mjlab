"""Samplers that pick which reset-deck entry starts each episode.

``DeckSampler`` (sampler.py) in two modes behind one interface:
  - "uniform": every deck entry equally likely (the paper's Uniform baseline),
  - "sgs": Success Guided Sampling (paper Sec. 3.2, Eq. 1-2, App. B.2).
Both record each entry's last H episode outcomes. Planned: an optional Beta prior
per entry (from FARO labels, see ``feasibility/``) to initialise p.
"""
