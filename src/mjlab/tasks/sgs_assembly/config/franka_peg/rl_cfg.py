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
      entropy_coef=0.005,
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
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
