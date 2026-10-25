"""Franka Research 3 (FR3) arm with the Franka Hand.

Sources (MuJoCo Menagerie, commit in ``xmls/MENAGERIE_COMMIT``, files unmodified):
- ``xmls/fr3/fr3.xml``: FR3 arm. Menagerie's FR3 has no gripper.
- ``xmls/franka_hand/mjx_hand.xml``: Franka Hand (same hardware on Panda and FR3).
  We use the MJX version because its fingers collide through small boxes instead
  of meshes: cheap on GPU, and the fingertip pad is a single named box per finger.

``get_spec`` composes the two in Python and strips the XML actuators, keyframe and
<option>, because mjlab adds its own actuators (from ``ARTICULATION``) and keyframe
(from ``HOME_KEYFRAME``), and takes solver options only from ``physics_cfg.py``.
"""

from functools import partial
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.tasks.sgs_assembly.physics_cfg import ContactCfg
from mjlab.tasks.sgs_assembly.robots import RobotInfo
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF composition.
##

_XML_DIR = Path(__file__).parent / "xmls"
FR3_XML = _XML_DIR / "fr3" / "fr3.xml"
HAND_XML = _XML_DIR / "franka_hand" / "mjx_hand.xml"

# Geom names. The hand XML only names one box per finger; get_spec names the rest.
PAD_GEOMS = r"(left|right)_finger_pad"  # Fingertip pad: the grasping surface.
FINGER_BOX_GEOMS = r"(left|right)_finger_box_[123]"  # Rest of each finger.
HAND_GEOMS = r"hand_capsule"  # Palm.
ARM_GEOMS = r"fr3_link[0-7]_collision"  # Convex-hull meshes of the arm links.

EE_SITE = "gripper"  # TCP between the fingertips, 0.1 m below the hand base.

GRAVITY_COMPENSATION = True  # Mimic the Franka controller (see get_spec).

# Box classes in mjx_hand.xml -> our names. Class 4 is the fingertip pad.
_FINGER_BOX_CLASSES = {
  "fingertip_pad_collision_1": "box_1",
  "fingertip_pad_collision_2": "box_2",
  "fingertip_pad_collision_3": "box_3",
  "fingertip_pad_collision_4": "pad",
}


def get_spec() -> mujoco.MjSpec:
  arm = mujoco.MjSpec.from_file(str(FR3_XML))
  hand = mujoco.MjSpec.from_file(str(HAND_XML))

  for spec in (arm, hand):
    for act in list(spec.actuators):
      spec.delete(act)
    for key in list(spec.keys):
      spec.delete(key)
  arm.option.integrator = mujoco.mjtIntegrator.mjINT_EULER  # Back to default.

  # Both files define materials named "black"/"white": rename the hand's.
  for mat in hand.materials:
    mat.name = f"hand_{mat.name}"
  for geom in hand.geoms:
    if geom.material:
      geom.material = f"hand_{geom.material}"

  # Name every finger collision geom so friction and contacts can target them.
  hand.geom("hand_capsule")  # Fails loudly if upstream renames it.
  for side in ("left", "right"):
    for geom in hand.body(f"{side}_finger").geoms:
      if geom.classname.name in _FINGER_BOX_CLASSES:
        geom.name = f"{side}_finger_{_FINGER_BOX_CLASSES[geom.classname.name]}"

  # The hand body already carries the 0.107 m flange offset and the -45 deg yaw,
  # so attach it directly in the link7 frame (as in menagerie's panda.xml).
  arm.attach(hand, prefix="", frame=arm.body("fr3_link7").add_frame())

  # The real Franka controller compensates gravity internally, so a joint target
  # is reached without steady-state sag. Without this, the PD loop needs an error
  # to hold the arm up (about 7 mm at the TCP in the home pose).
  if GRAVITY_COMPENSATION:
    for body in arm.bodies:
      body.gravcomp = 1.0
  return arm


##
# Actuators. Joint-position control through MuJoCo <position> actuators.
##

# Arm gains are menagerie's (kp, kv); effort limits are the FR3 torque limits.
# Armature, joint damping and frictionloss stay as identified in fr3.xml.
ARM_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=("fr3_joint[12]",),
    stiffness=4500.0,
    damping=450.0,
    effort_limit=87.0,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("fr3_joint[34]",),
    stiffness=3500.0,
    damping=350.0,
    effort_limit=87.0,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=("fr3_joint[567]",),
    stiffness=2000.0,
    damping=200.0,
    effort_limit=12.0,
  ),
)

# Franka Hand: one actuated finger; finger_joint2 follows through the XML's
# equality constraint. Force limit = the hand's 70 N continuous grasp force, so
# grasp normal forces (and hence friction limits) are realistic. Stiffness is our
# choice: closing on a 16 mm peg from a fully-closed target gives ~16 N; targets
# below 0 (allowed, ctrl is unlimited) squeeze harder, up to 70 N.
GRIPPER_FORCE_LIMIT = 70.0  # [N]
GRIPPER_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("finger_joint1",),
  stiffness=2000.0,
  damping=100.0,
  effort_limit=GRIPPER_FORCE_LIMIT,
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=(*ARM_ACTUATORS, GRIPPER_ACTUATOR),
  soft_joint_pos_limit_factor=0.95,
)

##
# Keyframe and limits.
##

FINGER_OPEN = 0.04  # [m] per finger, i.e. 80 mm aperture.

# Menagerie "home": flange pointing down, TCP at about (0.55, 0, 0.52) m.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "fr3_joint[1235]": 0.0,
    "fr3_joint4": -1.57079,
    "fr3_joint6": 1.57079,
    "fr3_joint7": -0.7853,
    "finger_joint[12]": FINGER_OPEN,
  },
  joint_vel={".*": 0.0},
)

# FR3 datasheet joint velocity limits [rad/s] (differ from the Panda's).
# Check these against your robot's datasheet / libfranka version.
JOINT_VEL_LIMITS = {
  "fr3_joint[1-4]": 2.62,
  "fr3_joint5": 5.26,
  "fr3_joint6": 4.18,
  "fr3_joint7": 5.26,
}

# target = home + scale * action. 0.5 rad per unit action for the arm (as in Isaac
# Lab's Franka tasks); for the gripper, action -1 closes fully from open.
ACTION_SCALE = {
  "fr3_joint[1-7]": 0.5,
  "finger_joint1": FINGER_OPEN,
}

##
# Collisions. Which geoms collide is decided here; the numbers come from ContactCfg.
##


def collision_cfg(contact: ContactCfg, arm_collisions: bool) -> CollisionCfg:
  """Gripper (pads, finger boxes, palm) always collides; arm links optionally.

  Priority is 0 everywhere so MuJoCo combines friction with max (as in SGS).
  Geoms not listed (visual meshes, finger/hand collision meshes) are disabled.
  """
  names = (PAD_GEOMS, FINGER_BOX_GEOMS, HAND_GEOMS)
  if arm_collisions:
    names += (ARM_GEOMS,)
  return CollisionCfg(
    geom_names_expr=names,
    contype=1,
    conaffinity=1,
    condim={PAD_GEOMS: contact.pad.condim, ".*": contact.robot.condim},
    priority=0,
    friction={PAD_GEOMS: contact.pad.friction, ".*": contact.robot.friction},
  )


def get_fr3_robot_cfg(contact: ContactCfg, arm_collisions: bool = False) -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(collision_cfg(contact, arm_collisions),),
    spec_fn=get_spec,
    articulation=ARTICULATION,
  )


FR3_INFO = RobotInfo(
  make_entity_cfg=partial(get_fr3_robot_cfg, arm_collisions=False),
  action_scale=ACTION_SCALE,
  arm_joints=("fr3_joint[1-7]",),
  gripper_joints=("finger_joint[12]",),
  joint_vel_limits=JOINT_VEL_LIMITS,
  ee_site=EE_SITE,
  pad_geoms=PAD_GEOMS,
  root_body="fr3_link0",
)
