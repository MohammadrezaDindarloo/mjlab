"""PPO settings. Started from the YAM lift-cube task; the SGS paper does not give
its PPO hyperparameters, so these are not tuned yet.

obs_mode "pointcloud": actor and critic are PointNetModels (rl/pointnet.py) that
read their 1D group plus the point_cloud group; "pose": plain MLPs on 1D groups.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.sgs_assembly.rl.pointnet import PointNetModelCfg


def fr3_peg_hole_ppo_runner_cfg(obs_mode: str = "pointcloud") -> RslRlOnPolicyRunnerCfg:
  model_cfg = PointNetModelCfg if obs_mode == "pointcloud" else RslRlModelCfg
  obs_groups = {"actor": ("actor",), "critic": ("critic",)}
  if obs_mode == "pointcloud":
    obs_groups = {
      "actor": ("actor", "point_cloud"),
      "critic": ("critic", "point_cloud"),
    }
  return RslRlOnPolicyRunnerCfg(
    obs_groups=obs_groups,
    actor=model_cfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
        # Floor on the exploration noise sigma (not in the paper). With
        # entropy_coef 0, sigma fell to 0.06 within 1750 iterations, before any
        # Reaching start was ever solved. Above 0.2 sigma is learned from the
        # reward only; 0.2 is +-0.01 rad per step (action scale 0.05 rad).
        "std_range": (0.2, 1e6),
      },
    ),
    critic=model_cfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # 0.005 let the exploration noise grow from 1.0 to ~2.5 within 80 iterations
      # while the sparse reward gave no signal yet (first uniform run).
      # The paper maximises the return only (no entropy term); 0.001 still let
      # sigma creep up while no successes gave a signal (1.0 -> 1.3 in 240 its).
      entropy_coef=0.0,
      num_learning_epochs=5,
      # Point clouds make each sample large (320 points through PointNet): with
      # 4 mini-batches the update ran out of memory at 16K envs; 16 fits (A40).
      num_mini_batches=16 if obs_mode == "pointcloud" else 4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="sgs_fr3_peg_hole",
    # Weights & Biases (the default logger): runs go to this project under your
    # default W&B entity; local run files land in ./wandb/ (git-ignored by mjlab).
    wandb_project="sgs-assembly",
    # Actions are clipped to [-1, 1] before the env: a joint target then moves at
    # most 0.05 rad per step (1.25 rad/s). Unclipped, noisy actions of +-5 made the
    # targets jump ~6 rad/s and most episodes ended in joint-speed failures.
    clip_actions=1.0,
    save_interval=100,
    # Checkpoints stay on disk only: uploading every one to W&B would be ~2000
    # uploads (~20 GB) per week-long run.
    upload_model=False,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
