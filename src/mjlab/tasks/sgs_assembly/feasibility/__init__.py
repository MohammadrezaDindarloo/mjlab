"""(Later) Interface to offline FARO feasibility labels.

Planned: export reset-deck states for the CasADi/Pinocchio tools (grasp wrench
feasibility QP, kinematic sequence optimisation), import their per-entry labels,
and turn them into Beta priors for the SGS sampler. No FARO code runs inside the
simulator; this package only converts data.
"""
