"""MDP checks on the real GPU env (success predicate, Eq. 5 reward, stability).

uv run pytest src/mjlab/tasks/sgs_assembly/tests/test_sgs_mdp.py
"""

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import (
  HOLE_POS,
  fr3_peg_hole_env_cfg,
)
from mjlab.tasks.sgs_assembly.objects.peg_hole.peg_hole import PegHoleCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PEG = PegHoleCfg()
L = PEG.peg_length
# Peg centre offsets from the hole frame origin (hole bottom on the table) and
# the expected success for each.
CASES = {
  "seated": ((0.0, 0.0, L / 2), True),
  "beside": ((0.06, 0.0, L / 2), False),  # Lateral gate.
  "hovering": ((0.0, 0.0, L / 2 + PEG.hole_depth + 0.02), False),  # Depth gate.
  "10mm_in": ((0.0, 0.0, L / 2 + PEG.hole_depth - 0.01), False),  # Depth gate.
}


@pytest.fixture(scope="module")
def env():
  cfg = fr3_peg_hole_env_cfg()
  cfg.scene.num_envs = len(CASES)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  env.reset()
  yield env
  env.close()


def _hold(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Zero arm action (hold pose) with the gripper commanded open."""
  a = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
  a[:, -1] = 1.0
  return a


def test_success_gates_and_reward(env):
  peg = env.scene["peg"]
  hole_w = env.scene.env_origins + torch.tensor(HOLE_POS, device=env.device)
  offsets = torch.tensor([c[0] for c in CASES.values()], device=env.device)
  pose = torch.zeros(env.num_envs, 7, device=env.device)
  pose[:, :3] = hole_w + offsets
  pose[:, 3] = 1.0  # Upright.
  peg.write_root_link_pose_to_sim(pose)
  peg.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
  env.sim.forward()

  _, reward, terminated, _, _ = env.step(_hold(env))

  success = env.termination_manager.get_term("success").tolist()
  expected = [c[1] for c in CASES.values()]
  assert success == expected, dict(zip(CASES, success, strict=True))
  assert terminated.tolist() == expected
  # The only nonzero penalty is 1e-5 * ||a||^2 = 1e-5 (open-gripper action = 1).
  for i, ok in enumerate(expected):
    assert reward[i].item() == pytest.approx(100.0 if ok else 0.0, abs=1e-3)


@pytest.mark.parametrize("agent", ["zero", "random"])
def test_agents_run_without_nans_or_failures(env, agent):
  env.reset()
  robot = env.scene["robot"]
  for _ in range(75):  # 3 s at 25 Hz.
    a = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    if agent == "random":
      a = 2 * torch.rand_like(a) - 1
    obs, reward, _, _, _ = env.step(a)
    assert all(torch.isfinite(v).all() for v in obs.values())
    assert torch.isfinite(reward).all()
    for name in ("arm_joint_vel_exceeded", "peg_out_of_bounds"):
      assert not env.termination_manager.get_term(name).any(), name
  assert robot.data.joint_vel.abs().max() < 5.0
