"""PointNet encoder for the scene point cloud, as an RSL-RL model.

The SGS paper's teacher observes a point cloud (App. A.3.2) but does not describe
its network, and its code is not released. We use PointNet (Qi et al., CVPR 2017)
without the optional T-Net alignment (our points are already in the gripper
frame):

  each point (x, y, z, one-hot part)  --shared MLP-->  per-point features
  max over all points                  -->             one feature vector

The shared MLP sees every point the same way and the max ignores point order, so
the encoding does not depend on how points are listed, and a symmetric peg
turned by 90 deg gives the same features.

``PointNetModel`` = RSL-RL's ``MLPModel`` whose latent is
  [normalised 1D observation groups, PointNet(point-cloud group)]
so it works for both the actor (Gaussian action head) and the critic (value).
Observation groups shaped (N, P, F) are treated as point clouds; (N, D) as 1D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP
from tensordict import TensorDict

from mjlab.rl import RslRlModelCfg


@dataclass
class PointNetModelCfg(RslRlModelCfg):
  """RslRlModelCfg + the PointNet settings (passed to PointNetModel)."""

  class_name: str = "mjlab.tasks.sgs_assembly.rl.pointnet:PointNetModel"
  pointnet_cfg: dict[str, Any] = field(
    default_factory=lambda: {
      "hidden_dims": (64, 128),
      "xyz_scale": 10.0,
      # bfloat16 per-point MLP: ~2x faster PPO updates and less memory at 16K
      # envs (30.8 -> 14.3 s/iteration on an A40); the output stays float32.
      "bf16": True,
    }
  )


class PointNetEncoder(nn.Module):
  """(B, P, F) points -> (B, hidden_dims[-1]) feature."""

  def __init__(
    self,
    input_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (64, 128),
    activation: str = "elu",
    xyz_scale: float = 10.0,
    bf16: bool = False,
  ) -> None:
    super().__init__()
    # Run the per-point MLP in bfloat16 (autocast) to halve its memory/time; the
    # pooled feature is returned in float32.
    self.bf16 = bf16
    # Shared per-point MLP (acts on the last dim, i.e. on every point alike); the
    # last layer is also activated before the max, as in PointNet.
    self.point_mlp = MLP(
      input_dim,
      hidden_dims[-1],
      list(hidden_dims[:-1]),
      activation,
      last_activation=activation,
    )
    # Coordinates are in metres (|x| <~ 0.5); scale them to O(1) for the MLP.
    # The remaining features (part labels) are left as they are.
    self.xyz_scale = xyz_scale
    self.output_dim = int(hidden_dims[-1])

  def forward(self, points: torch.Tensor) -> torch.Tensor:
    x = torch.cat([points[..., :3] * self.xyz_scale, points[..., 3:]], dim=-1)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.bf16):
      per_point = self.point_mlp(x)  # (B, P, H)
      pooled = per_point.max(dim=1).values  # (B, H)
    return pooled.float()


class PointNetModel(MLPModel):
  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    pointnet_cfg: dict[str, Any] | None = None,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
  ) -> None:
    pointnet_cfg = dict(pointnet_cfg or {})
    active = obs_groups[obs_set]
    # Plain attributes may be set before nn.Module.__init__ (as in mjlab's CNN
    # model); MLPModel.__init__ needs the point latent size to build its MLP.
    self.pc_groups = [g for g in active if len(obs[g].shape) == 3]
    feat_dim = int(pointnet_cfg.get("hidden_dims", (64, 128))[-1])
    self.pc_latent_dim = feat_dim * len(self.pc_groups)
    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )
    self.encoders = nn.ModuleDict(
      {
        g: PointNetEncoder(obs[g].shape[-1], activation=activation, **pointnet_cfg)
        for g in self.pc_groups
      }
    )

  def _get_obs_dim(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str
  ) -> tuple[list[str], int]:
    """1D groups only (the base class rejects (N, P, F) point-cloud groups)."""
    groups = [g for g in obs_groups[obs_set] if len(obs[g].shape) == 2]
    return groups, sum(obs[g].shape[-1] for g in groups)

  def _get_latent_dim(self) -> int:
    return self.obs_dim + self.pc_latent_dim

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    latent_1d = super().get_latent(obs, masks, hidden_state)
    features = [self.encoders[g](obs[g]) for g in self.pc_groups]
    return torch.cat([latent_1d, *features], dim=-1)

  def as_jit(self) -> nn.Module:
    raise NotImplementedError("Export of the PointNet policy is not supported yet.")

  def as_onnx(self, verbose: bool) -> nn.Module:
    raise NotImplementedError("Export of the PointNet policy is not supported yet.")
