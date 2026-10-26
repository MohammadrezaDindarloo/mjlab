from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  fr3_peg_hole_deck_env_cfg,
  fr3_peg_hole_env_cfg,
  fr3_robot_only_env_cfg,
)
from .rl_cfg import fr3_peg_hole_ppo_runner_cfg

# Point-cloud tasks use the plain mjlab runner: the manipulation runner exports
# an ONNX policy at every checkpoint, which the PointNet model does not support.
_PC_RL = fr3_peg_hole_ppo_runner_cfg("pointcloud")

register_mjlab_task(
  task_id="Mjlab-SGS-Robot-FR3",
  env_cfg=fr3_robot_only_env_cfg(),
  play_env_cfg=fr3_robot_only_env_cfg(play=True),
  rl_cfg=fr3_peg_hole_ppo_runner_cfg("pose"),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-SGS-PegHole-FR3",
  env_cfg=fr3_peg_hole_env_cfg(),
  play_env_cfg=fr3_peg_hole_env_cfg(play=True),
  rl_cfg=_PC_RL,
  runner_cls=MjlabOnPolicyRunner,
)

# Main training task: every episode starts from a reset-deck entry; the actor
# sees the point cloud (paper setup).
register_mjlab_task(
  task_id="Mjlab-SGS-PegHole-FR3-Deck",
  env_cfg=fr3_peg_hole_deck_env_cfg(),
  play_env_cfg=fr3_peg_hole_deck_env_cfg(play=True),
  rl_cfg=_PC_RL,
  runner_cls=MjlabOnPolicyRunner,
)

# Same deck and training, but episodes start from entries chosen by Success Guided
# Sampling (paper Sec. 3.2) instead of uniformly: the method being reproduced.
register_mjlab_task(
  task_id="Mjlab-SGS-PegHole-FR3-Deck-SGS",
  env_cfg=fr3_peg_hole_deck_env_cfg(sampler="sgs"),
  play_env_cfg=fr3_peg_hole_deck_env_cfg(play=True, sampler="sgs"),
  rl_cfg=_PC_RL,
  runner_cls=MjlabOnPolicyRunner,
)

# Same, but the actor sees exact peg/hole poses instead (debugging / comparison).
register_mjlab_task(
  task_id="Mjlab-SGS-PegHole-FR3-Deck-Pose",
  env_cfg=fr3_peg_hole_deck_env_cfg(obs_mode="pose"),
  play_env_cfg=fr3_peg_hole_deck_env_cfg(play=True, obs_mode="pose"),
  rl_cfg=fr3_peg_hole_ppo_runner_cfg("pose"),
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
    rl_cfg=_PC_RL,
    runner_cls=MjlabOnPolicyRunner,
  )
