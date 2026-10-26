from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  fr3_peg_hole_deck_env_cfg,
  fr3_peg_hole_env_cfg,
  fr3_robot_only_env_cfg,
)
from .rl_cfg import fr3_peg_hole_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-SGS-Robot-FR3",
  env_cfg=fr3_robot_only_env_cfg(),
  play_env_cfg=fr3_robot_only_env_cfg(play=True),
  rl_cfg=fr3_peg_hole_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-SGS-PegHole-FR3",
  env_cfg=fr3_peg_hole_env_cfg(),
  play_env_cfg=fr3_peg_hole_env_cfg(play=True),
  rl_cfg=fr3_peg_hole_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-SGS-PegHole-FR3-Deck",
  env_cfg=fr3_peg_hole_deck_env_cfg(),
  play_env_cfg=fr3_peg_hole_deck_env_cfg(play=True),
  rl_cfg=fr3_peg_hole_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

# One task per deck family, e.g. to watch it or (Reaching) to evaluate on it, as in
# the paper. They read the same deck file and only restrict which rows are drawn.
for _suffix, _family in (
  ("NearGoal", "NEAR_GOAL"),
  ("StableGrasp", "STABLE_GRASP"),
  ("Reaching", "REACHING"),
):
  register_mjlab_task(
    task_id=f"Mjlab-SGS-PegHole-FR3-Deck-{_suffix}",
    env_cfg=fr3_peg_hole_deck_env_cfg(families=(_family,)),
    play_env_cfg=fr3_peg_hole_deck_env_cfg(play=True, families=(_family,)),
    rl_cfg=fr3_peg_hole_ppo_runner_cfg(),
    runner_cls=ManipulationOnPolicyRunner,
  )
