"""Point-cloud sampling checks (CPU only).

uv run pytest src/mjlab/tasks/sgs_assembly/tests/test_sgs_point_cloud.py
"""

import numpy as np
import pytest

from mjlab.scene import Scene
from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import fr3_peg_hole_env_cfg
from mjlab.tasks.sgs_assembly.mdp import point_cloud as pcm
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg


@pytest.fixture(scope="module")
def model():
  return Scene(fr3_peg_hole_env_cfg().scene, device="cpu").compile()


def test_part_counts_and_gripper_split(model):
  rng = np.random.default_rng(0)
  for part, n in pcm.POINTS.items():
    body_ids, local = pcm.sample_part(model, part, n, rng)
    assert body_ids.shape == (n,) and local.shape == (n, 3)
  body_ids, _ = pcm.sample_part(model, "gripper", pcm.POINTS["gripper"], rng)
  counts = {model.body(b).name: int((body_ids == b).sum()) for b in set(body_ids)}
  assert counts == pcm.GRIPPER_SPLIT


def test_peg_points_lie_on_the_peg_surface(model):
  peg = PegHoleCfg()
  half = np.array([peg.peg_width / 2] * 2 + [peg.peg_length / 2])
  _, local = pcm.sample_part(model, "peg", 500, np.random.default_rng(1))
  assert np.all(np.abs(local) <= half + 1e-9)
  assert np.all(np.isclose(np.abs(local), half, atol=1e-9).any(axis=1))
