"""Run the zero and random agents on the GPU env and print a stability summary.

For each agent: steps ``--seconds`` of sim time in ``--num-envs`` parallel envs
and reports NaNs, largest joint and peg speeds, and which terminations fired.

MuJoCo Warp prints solver warnings (e.g. "linesearch iterations limit reached")
from inside GPU kernels, so Python cannot count them. Each phase is wrapped in
"@@BEGIN <agent>" / "@@END" lines; count the warnings per phase with:

  uv run python src/mjlab/tasks/sgs_assembly/scripts/run_agents.py 2>&1 | awk \
    '/@@BEGIN/{a=$2; n=0} /linesearch iterations/{n++} /@@END/{print a": "n}'
"""

import torch
import tyro

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main(
  task: str = "Mjlab-SGS-PegHole-FR3", num_envs: int = 256, seconds: float = 10.0
) -> None:
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  steps = int(seconds / env.step_dt)
  shape = (num_envs, env.action_manager.total_action_dim)
  tm = env.termination_manager

  for agent in ("zero", "random"):
    env.reset()
    fired = {name: 0 for name in tm.active_terms}
    nan, max_qd, max_peg_v = False, 0.0, 0.0
    print(f"@@BEGIN {agent}", flush=True)
    for _ in range(steps):
      a = torch.zeros(shape, device=env.device)
      if agent == "random":
        a = 2 * torch.rand_like(a) - 1
      obs, reward, _, _, _ = env.step(a)
      nan |= not torch.isfinite(reward).all().item()
      groups = [v for v in obs.values() if isinstance(v, torch.Tensor)]
      nan |= not all(torch.isfinite(v).all().item() for v in groups)
      max_qd = max(max_qd, env.scene["robot"].data.joint_vel.abs().max().item())
      if "peg" in env.scene.entities:
        v = env.scene["peg"].data.root_link_lin_vel_w.norm(dim=1).max().item()
        max_peg_v = max(max_peg_v, v)
      for name in fired:
        fired[name] += int(tm.get_term(name).sum().item())
    torch.cuda.synchronize()
    print("@@END", flush=True)
    print(
      f"[{agent}] {num_envs} envs x {seconds} s ({steps * cfg.decimation} physics "
      f"steps each): NaN={nan}  max|qd|={max_qd:.2f} rad/s  "
      f"max|v_peg|={max_peg_v:.3f} m/s  terminations={fired}",
      flush=True,
    )
  env.close()


if __name__ == "__main__":
  tyro.cli(main)
