"""Watch the scene point cloud that the policy observes (viser, in the browser).

Two views of the SAME points (the "point_cloud" observation of env 0):
  1. On the scene: each point drawn where it really is in the world, coloured by
     part (gripper blue, peg orange, hole green). They should lie on the
     surfaces of the hand/fingers, the peg and the hole walls.
  2. "What the policy sees": the observation itself, i.e. the points in the
     gripper (TCP) frame, drawn next to the robot, enlarged 2x, with the gripper
     frame's axes (x red, y green, z blue; z = approach direction, drawn pointing
     down). Like a camera bolted to the wrist: the gripper stays still and the
     peg and hole move around it (the finger points move when it opens/closes,
     and a grasped peg stops moving because it moves with the gripper).
Checkboxes in the "Point cloud" panel toggle each view and the 1 mm training
noise (the policy gets noisy points during training, clean ones in evaluation).

  # Scripted pick-and-place (default), so the cloud changes as the peg moves:
  uv run python src/mjlab/tasks/sgs_assembly/scripts/view_point_cloud.py
  # Deck starts with the zero agent, e.g. one family:
  uv run python src/mjlab/tasks/sgs_assembly/scripts/view_point_cloud.py \
    --task Mjlab-SGS-PegHole-FR3-Deck-NearGoal --agent zero
Then open http://localhost:8080 (through your SSH tunnel) and press play.
"""

import importlib.util
from pathlib import Path

import numpy as np
import torch
import tyro
import viser

import mjlab.tasks  # noqa: F401  (populates the task registry)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.sgs_assembly.config.franka_peg.env_cfgs import POINT_CLOUD_NOISE_STD
from mjlab.tasks.sgs_assembly.mdp.point_cloud import POINTS, TCP_SITE
from mjlab.viewer import ViserPlayViewer


def _load_scripted_policy():
  """scripts/ is not a package (on purpose, see README), so load the file."""
  path = Path(__file__).parent / "scripted_insert.py"
  spec = importlib.util.spec_from_file_location("scripted_insert", path)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module.ScriptedInsertPolicy


COLORS = {"gripper": (60, 120, 255), "peg": (255, 140, 0), "hole": (40, 190, 80)}
VIEW_POS = np.array([0.55, -0.75, 0.6])  # Where the "policy view" is drawn.
VIEW_SCALE = 2.0
# The gripper frame's z axis is the approach direction (palm -> fingertips). Turn
# the view 180 deg about x so that z points DOWN, as when the gripper points at
# the table; otherwise the hole (ahead of the fingertips) is drawn above it.
VIEW_WXYZ = (0.0, 1.0, 0.0, 0.0)


class PointCloudVisualizer:
  """Wraps a policy; after each observation, redraws env 0's point cloud."""

  def __init__(self, env: ManagerBasedRlEnv, policy, server: viser.ViserServer):
    self.env, self.policy = env, policy
    # Set after the viewer is built (it owns the scene and its draw offset).
    self.viewer: ViserPlayViewer | None = None
    self.site = env.sim.mj_model.site(TCP_SITE).id
    colors = np.concatenate(
      [np.tile(COLORS[p], (n, 1)) for p, n in POINTS.items()]
    ).astype(np.uint8)
    zeros = np.zeros((len(colors), 3), dtype=np.float32)
    with server.gui.add_folder("Point cloud"):
      self.show_world = server.gui.add_checkbox("On the scene", True)
      self.show_policy = server.gui.add_checkbox("What the policy sees", True)
      self.noise = server.gui.add_checkbox("Add 1 mm training noise", False)
    self.world = server.scene.add_point_cloud(
      "/pc_world", zeros, colors, point_size=0.003, point_shape="circle"
    )
    # Gripper frame axes: x red, y green, z blue (approach, drawn pointing down).
    server.scene.add_frame(
      "/pc_policy",
      wxyz=VIEW_WXYZ,
      position=VIEW_POS,
      axes_length=0.05,
      axes_radius=0.002,
    )
    # point_size is in the points' own units and gets enlarged by ``scale`` too.
    self.local = server.scene.add_point_cloud(
      "/pc_policy/points",
      zeros,
      colors,
      point_size=0.003,
      point_shape="circle",
      scale=VIEW_SCALE,
    )
    server.scene.add_label(
      "/pc_policy/label",
      "what the policy sees: gripper frame, x2 (gripper fixed, scene moves)",
      position=(0.0, 0.0, -0.3),  # Frame is flipped: -z is up.
    )

  def __call__(self, obs) -> torch.Tensor:
    pc = obs["point_cloud"][0, :, :3].cpu().numpy()  # Gripper frame, env 0.
    if self.noise.value:
      pc = pc + np.random.normal(0.0, POINT_CLOUD_NOISE_STD, pc.shape)
    d = self.env.sim.data
    tcp = d.site_xpos[0, self.site].cpu().numpy()
    rot = d.site_xmat[0, self.site].reshape(3, 3).cpu().numpy()
    # The viewer draws the scene shifted by its own offset; apply the same shift.
    scene = getattr(self.viewer, "_scene", None)
    offset = getattr(scene, "_scene_offset", None)
    offset = np.zeros(3) if offset is None else np.asarray(offset)
    self.world.points = (tcp + pc @ rot.T + offset).astype(np.float32)
    self.local.points = pc.astype(np.float32)
    self.world.visible = self.show_world.value
    self.local.visible = self.show_policy.value
    return self.policy(obs)

  def reset(self, dones=None) -> None:
    if hasattr(self.policy, "reset"):
      self.policy.reset(dones)


def main(
  task: str = "Mjlab-SGS-PegHole-FR3", agent: str = "scripted", num_envs: int = 1
) -> None:
  """Args:
  task: Any SGS peg-in-hole task id (all have the point_cloud observation).
  agent: "scripted" (pick-and-place demo), "zero" or "random".
  num_envs: Parallel envs; the point cloud is drawn for env 0.
  """
  cfg = load_env_cfg(task, play=True)
  cfg.scene.num_envs = num_envs
  if agent == "scripted":
    cfg.episode_length_s = 20.0  # The script needs ~9 s per insertion.
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  shape = (num_envs, env.action_manager.total_action_dim)
  if agent == "scripted":
    policy = _load_scripted_policy()(env)
  elif agent == "zero":
    policy = lambda obs: torch.zeros(shape, device=env.device)  # noqa: E731
  else:
    policy = lambda obs: 2 * torch.rand(shape, device=env.device) - 1  # noqa: E731
  server = viser.ViserServer(label="mjlab point cloud")
  viewer_policy = PointCloudVisualizer(env, policy, server)
  viewer = ViserPlayViewer(wrapped, viewer_policy, viser_server=server)
  viewer_policy.viewer = viewer
  viewer.run()


if __name__ == "__main__":
  tyro.cli(main)
