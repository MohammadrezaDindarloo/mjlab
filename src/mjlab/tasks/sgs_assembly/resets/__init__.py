"""(Later) Reset deck: a fixed, pre-generated buffer of start states.

Planned contents:
- A ``ResetState`` record (arm joints, gripper aperture, peg pose, family label)
  that is the single exchange format with offline tools (FARO feasibility labels).
- Generators for the three OmniReset families: Near-Goal (peg along its insertion
  path, fraction in [0, 1.1]), Stable Grasp (peg held above the hole), and Reaching
  (peg anywhere, gripper aligned to a grasp frame with random aperture).
- Save/load of a deck to disk (e.g. .npz) so decks are reproducible and shareable.
"""
