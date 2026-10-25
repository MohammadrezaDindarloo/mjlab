"""Base assembly env config, generic over the robot and the objects.

``make_assembly_env_cfg`` builds the parts every assembly task shares:
- scene: table (ground plane) + robot + any extra entities (peg, hole, ...),
- sim: all solver/contact settings from ``PhysicsCfg``,
- action: joint-position targets for every robot actuator,
- observations: proprioception only (task configs add object terms),
- events: place every entity in its env at reset,
- termination: time-out.

Task-specific terms (reward, success, failures, object observations, peg
randomisation) are added by the task config in ``config/<robot>_<object>/``.
"""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.sgs_assembly.physics_cfg import PhysicsCfg
from mjlab.tasks.sgs_assembly.robots import RobotInfo
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig


def make_assembly_env_cfg(
  robot: RobotInfo,
  physics: PhysicsCfg,
  objects: dict[str, EntityCfg] | None = None,
  episode_length_s: float = 10.0,
) -> ManagerBasedRlEnvCfg:
  objects = objects or {}
  contact = physics.contact

  # The ground plane is the table top; its friction comes from physics_cfg too.
  table = TerrainEntityCfg(
    terrain_type="plane",
    collisions=(
      CollisionCfg(
        geom_names_expr=(".*",),
        contype=1,
        conaffinity=1,
        condim=contact.table.condim,
        priority=0,
        friction=contact.table.friction,
      ),
    ),
  )

  proprio = {
    "joint_pos": ObservationTermCfg(func=mjlab_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mjlab_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mjlab_mdp.last_action),
  }

  # Every entity (robot, objects) is written to its env's origin + init_state pose
  # at reset. Fixed-base entities (robot, hole) need this to leave the world origin.
  events: dict[str, EventTermCfg] = {
    f"reset_{name}_pose": EventTermCfg(
      func=mjlab_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {},
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg(name),
      },
    )
    for name in ("robot", *objects)
  }
  events["reset_robot_joints"] = EventTermCfg(
    func=mjlab_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=table,
      entities={"robot": robot.make_entity_cfg(contact), **objects},
      num_envs=1,
      env_spacing=2.0,
    ),
    sim=physics.solver.to_simulation_cfg(),
    decimation=physics.solver.decimation,
    episode_length_s=episode_length_s,
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=robot.action_scale,
        use_default_offset=True,
      )
    },
    observations={
      "actor": ObservationGroupCfg(dict(proprio), enable_corruption=False),
      "critic": ObservationGroupCfg(dict(proprio), enable_corruption=False),
    },
    events=events,
    terminations={
      "time_out": TerminationTermCfg(func=mjlab_mdp.time_out, time_out=True),
    },
    # SGS Eq. 5 is an undiscounted-scale sparse reward (+100 on success); mjlab
    # would otherwise multiply every reward term by step_dt (= 0.04 s).
    scale_rewards_by_dt=False,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=robot.root_body,
      distance=1.8,
      elevation=-20.0,
      azimuth=135.0,
    ),
  )
