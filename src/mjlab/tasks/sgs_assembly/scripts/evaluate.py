"""Evaluate a trained checkpoint: success rate per deck family, no exploration noise.

Runs one episode per env from each family's starts (the -Deck-<Family> tasks, in
play mode: deterministic actions, no observation noise) and reports how every
first episode ended. Unlike the training curve SGS/success_rate/<family>, which
averages each start's last 100 episodes (~300 iterations), this is the policy as
it is now.

  uv run python src/mjlab/tasks/sgs_assembly/scripts/evaluate.py \
    --checkpoint logs/rsl_rl/sgs_fr3_peg_hole/<run>/model_1750.pt
"""

from dataclasses import asdict

import torch
import tyro

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

FAMILY_TASKS = {
  "near_goal": "Mjlab-SGS-PegHole-FR3-Deck-NearGoal",
  "stable_grasp": "Mjlab-SGS-PegHole-FR3-Deck-StableGrasp",
  "reaching": "Mjlab-SGS-PegHole-FR3-Deck-Reaching",
}


def evaluate(task: str, checkpoint: str, num_envs: int) -> dict[str, float]:
  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = num_envs
  # Play mode never times out; evaluate with the training episode length.
  env_cfg.episode_length_s = load_env_cfg(task).episode_length_s
  agent_cfg = load_rl_cfg(task)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(env_cfg, device="cuda:0"), clip_actions=agent_cfg.clip_actions
  )
  runner_cls = load_runner_cls(task)
  assert runner_cls is not None, f"{task} registers no runner class"
  runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
  runner.load(checkpoint, load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device="cuda:0")

  tm = env.unwrapped.termination_manager
  ended = {name: torch.zeros(num_envs, dtype=torch.bool) for name in tm.active_terms}
  done_once = torch.zeros(num_envs, dtype=torch.bool)
  obs = env.get_observations()
  with torch.no_grad():
    while not done_once.all():
      obs, _, dones, _ = env.step(policy(obs))
      first = dones.bool().cpu() & ~done_once
      for name in ended:
        ended[name] |= first & tm.get_term(name).bool().cpu()
      done_once |= first
  env.close()
  return {name: flag.float().mean().item() for name, flag in ended.items()}


def main(checkpoint: str, num_envs: int = 1024) -> None:
  results = {f: evaluate(t, checkpoint, num_envs) for f, t in FAMILY_TASKS.items()}
  names = list(next(iter(results.values())))
  print(f"\n{checkpoint}\n{'family':<14}" + "".join(f"{n:>24}" for n in names))
  for family, r in results.items():
    print(f"{family:<14}" + "".join(f"{r[n]:>24.3f}" for n in names))


if __name__ == "__main__":
  tyro.cli(main)
