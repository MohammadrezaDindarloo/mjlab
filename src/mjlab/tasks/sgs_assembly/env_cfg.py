"""Base assembly env config, generic over the robot and the objects.

``make_assembly_env_cfg`` builds what SGS shares across *all* manipulation tasks:
- scene: table (ground plane) + robot + any extra entities (peg, hole, ...),
- sim: all solver/contact settings from ``PhysicsCfg``,
- actions: delta joint-position arm + binary gripper (mdp/actions.py),
- observations: proprioception (actor) + joint velocities (critic only, App. C.3),
- rewards: Eq. 5 regularisers and the failure penalty,
- terminations: arm joint velocity > 2x limit (failure) and time-out,
- events: place every entity in its env at reset.

A task config adds its object observations, success term (+ success reward),
object-out-of-bounds failure, and object resets. A task that adds a failure
termination must also add its name to the ``failure`` reward's ``term_names``.
"""

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as mjlab_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.sgs_assembly.mdp import actions, observations, rewards, terminations
from mjlab.tasks.sgs_assembly.physics_cfg import PhysicsCfg
from mjlab.tasks.sgs_assembly.robots import RobotInfo
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.viewer import ViewerConfig

# Paper: "abnormal joint velocities" = above twice the configured limits.
JOINT_VEL_FAILURE_FACTOR = 2.0


def make_assembly_env_cfg(
  robot: RobotInfo,
  physics: PhysicsCfg,
  objects: dict[str, EntityCfg] | None = None,
  weights: rewards.SgsRewardWeights | None = None,
  episode_length_s: float = 10.0,
) -> ManagerBasedRlEnvCfg:
  objects = objects or {}
  weights = weights or rewards.SgsRewardWeights()
  contact = physics.contact
  arm_cfg = SceneEntityCfg("robot", joint_names=robot.arm_joints)
  ee_cfg = SceneEntityCfg("robot", site_names=(robot.ee_site,))

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

  # Actor: what the paper's teacher sees besides the scene (App. A.3.2).
  actor_terms = {
    "joint_pos": ObservationTermCfg(func=mjlab_mdp.joint_pos_rel),
    "ee_pose": ObservationTermCfg(
      func=observations.ee_pose_in_base, params={"robot_cfg": ee_cfg}
    ),
    "actions": ObservationTermCfg(func=mjlab_mdp.last_action),
  }
  # Critic: additionally joint velocities (privileged, App. C.3).
  critic_terms = {
    **actor_terms,
    "joint_vel": ObservationTermCfg(func=mjlab_mdp.joint_vel_rel),
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

  terms = {
    "time_out": TerminationTermCfg(func=mjlab_mdp.time_out, time_out=True),
    "arm_joint_vel_exceeded": TerminationTermCfg(
      func=terminations.joint_vel_exceeded,
      params={
        "limits": robot.arm_joint_vel_limits,
        "factor": JOINT_VEL_FAILURE_FACTOR,
        "asset_cfg": arm_cfg,
      },
    ),
  }

  # Eq. 5 without the success term (added by the task, which defines S).
  reward_terms = {
    "failure": RewardTermCfg(
      func=rewards.terminated_by,
      weight=-weights.failure,
      params={"term_names": ("arm_joint_vel_exceeded",)},
    ),
    "action": RewardTermCfg(func=rewards.action_l2_clamped, weight=-weights.action),
    "action_rate": RewardTermCfg(
      func=rewards.action_rate_l2_clamped, weight=-weights.action_rate
    ),
    "arm_joint_vel": RewardTermCfg(
      func=rewards.joint_vel_l2_clamped,
      weight=-weights.arm_joint_vel,
      params={"asset_cfg": arm_cfg},
    ),
  }

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
      "arm": actions.DeltaJointPositionActionCfg(
        entity_name="robot",
        joint_names=robot.arm_joints,
        scale=robot.arm_action_scale,
      ),
      "gripper": actions.BinaryGripperActionCfg(
        entity_name="robot",
        joint_name=robot.gripper_joint,
        open_pos=robot.gripper_open,
        closed_pos=robot.gripper_closed,
      ),
    },
    observations={
      "actor": ObservationGroupCfg(actor_terms, enable_corruption=False),
      "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
    },
    events=events,
    rewards=reward_terms,
    terminations=terms,
    # Eq. 5 is a per-step reward (+100 on success); mjlab would otherwise multiply
    # every reward term by step_dt (= 0.04 s).
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
