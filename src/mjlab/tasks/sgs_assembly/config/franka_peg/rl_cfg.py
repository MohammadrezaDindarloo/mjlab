"""PPO settings.

The SGS paper gives no PPO hyperparameters (it names PPO, keeps the number of
mini-batches fixed across scales, and builds on OmniReset's pipeline). Every value
below is therefore OmniReset's released training config (UWLab,
``omnireset/config/ur5e_robotiq_2f85/agents/rsl_rl_cfg.py``, ``Base_PPORunnerCfg``,
used by ``OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-v0``), except:
- ``num_mini_batches`` for point clouds (see below),
- the PointNet encoder (rl/pointnet.py): OmniReset's teacher sees state, not a
  point cloud, and SGS does not describe its network.

obs_mode "pointcloud": actor and critic are PointNetModels (rl/pointnet.py) that
read their 1D group plus the point_cloud group; "pose": plain MLPs on 1D groups.
"""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.sgs_assembly.rl.pointnet import PointNetModelCfg

# OmniReset: 4. Point clouds make each sample large (320 points through PointNet).
# Measured at 16384 envs on one A40 (46 GB), 2026-11-04: 4 -> out of memory;
# 8 -> 45.4 GB peak (no margin); 16 -> 27.4 GB; all ~30 s per iteration.
POINTCLOUD_MINI_BATCHES = 16


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
      hidden_dims=(512, 256, 128, 64),
      activation="elu",
      obs_normalization=True,
      # State-dependent noise: the network outputs a log std per action next to
      # the mean, so exploration can stay high where the policy is unsure.
      distribution_cfg={
        "class_name": "HeteroscedasticGaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
        "std_range": (0.001, 2.0),
      },
    ),
    critic=model_cfg(
      hidden_dims=(512, 256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      normalize_advantage_per_mini_batch=False,
      clip_param=0.2,
      entropy_coef=0.006,
      num_learning_epochs=5,
      num_mini_batches=POINTCLOUD_MINI_BATCHES if obs_mode == "pointcloud" else 4,
      learning_rate=1.0e-4,
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
    # No action clipping: OmniReset's runner config leaves clip_actions unset
    # (None). The OSC turns any action into torques within the actuator limits.
    clip_actions=None,
    save_interval=100,
    # Checkpoints stay on disk only: uploading every one to W&B would be ~2000
    # uploads (~20 GB) per week-long run.
    upload_model=False,
    num_steps_per_env=32,
    max_iterations=40_000,
  )
