"""PAL Kangaroo constants: the model with 7-DoF arms and grippers.

A leg is a hip-ankle slide (``leg_*_length_joint``) with a passive knee linkage
closed on it by an equality constraint (``leg_*_femur_joint``,
``leg_*_knee_joint``). The model, gains and action scales are those of pal_mjlab.
"""

import math
from pathlib import Path

import mujoco
import torch

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

KANGAROO_GRIPPERS_XML: Path = (
  MJLAB_SRC_PATH
  / "asset_zoo"
  / "robots"
  / "pal_kangaroo"
  / "xmls"
  / "kangaroo_grippers.xml"
)
assert KANGAROO_GRIPPERS_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(KANGAROO_GRIPPERS_XML))


##
# Actuator config.
##

# Stiffness and damping put each joint's armature at this natural frequency and
# damping ratio.
NATURAL_FREQ = 3.5 * 2.0 * 3.1415926535  # 3.5 Hz
DAMPING_RATIO = 2.0


def _actuator(
  target_names_expr: tuple[str, ...],
  effort_limit: float,
  armature: float,
  frictionloss: float | None = None,
  viscous_damping: float | None = None,
) -> BuiltinPositionActuatorCfg:
  return BuiltinPositionActuatorCfg(
    target_names_expr=target_names_expr,
    stiffness=round(armature * NATURAL_FREQ**2, 3),
    damping=round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3),
    effort_limit=effort_limit,
    armature=armature,
    frictionloss=frictionloss,
    viscous_damping=viscous_damping,
  )


# Reflected inertia of the arm and waist motors: rotor inertia * gear ratio^2.
ARMATURE_S_PLUS = 3.3e-05 * 121**2
ARMATURE_S_MINUS = 2.99e-05 * 101**2
ARMATURE_XS = 1.99e-05 * 101**2

KANGAROO_GRIPPERS_ACTUATORS = (
  _actuator(("leg_.*_1_joint",), 80.0, 0.25, 0.9, 1.65),
  _actuator(("leg_.*_2_joint",), 230.0, 0.5, 4.0, 6.49),
  _actuator(("leg_.*_3_joint",), 139.0, 0.2, 2.34, 2.705),
  _actuator(("leg_.*_4_joint",), 140.0, 0.1, 1.25, 2.8),
  _actuator(("leg_.*_5_joint",), 82.0, 0.05, 0.8, 1.0),
  # The armature of the leg length joint is the actuator inertia times the
  # velocity ratio, which is not constant: its average value is used.
  _actuator(("leg_.*_length_joint",), 1100.0, 4.1),
  _actuator(
    ("arm_.*_1_joint", "arm_.*_2_joint", "pelvis_1_joint", "pelvis_2_joint"),
    50.0,
    ARMATURE_S_PLUS,
  ),
  _actuator((r"arm_.*_(?![1267]_joint)\d+_joint",), 25.0, ARMATURE_S_MINUS),
  _actuator((r"arm_.*_(?![12345]_joint)\d+_joint",), 25.0, ARMATURE_XS),
)

##
# Keyframe config.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.9),
  joint_pos={
    "leg_left_1_joint": -0.012,
    "leg_right_1_joint": 0.012,
    "leg_.*_2_joint": 0.054,
    "leg_left_3_joint": 0.04,
    "leg_right_3_joint": -0.04,
    "leg_.*_length_joint": 0.6,
    "leg_.*_4_joint": -0.053,
    "leg_.*_5_joint": 0.0,
    "leg_.*_femur_joint": 0.9,
    "leg_.*_knee_joint": 1.8,
    "arm_left_1_joint": 0.24,
    "arm_right_1_joint": -0.24,
    "arm_.*_2_joint": 1.32,
    "arm_left_3_joint": 1.57,
    "arm_right_3_joint": -1.57,
    "arm_.*_4_joint": 0.8,
    "pelvis_1_joint": 0.0,
    "pelvis_2_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_FOOT_REGEX = ".*_foot.*_collision"

# All collision geoms, self collisions included. The feet get condim 3 and custom
# friction, everything else condim 1.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={_FOOT_REGEX: 3, ".*_collision": 1},
  priority={_FOOT_REGEX: 1, ".*_collision": 0},
  friction={_FOOT_REGEX: (0.6,)},
)

##
# Final config.
##

KANGAROO_GRIPPERS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=KANGAROO_GRIPPERS_ACTUATORS,
  soft_joint_pos_limit_factor=0.9,
)


def get_kangaroo_grippers_robot_cfg() -> EntityCfg:
  """Get a fresh Kangaroo grippers robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=KANGAROO_GRIPPERS_ARTICULATION,
  )


# Natural frequency [Hz] of the gains the arm and waist action scales are computed
# for. At the robot's 3.5 Hz gains these scales are 12 times smaller: too little
# exploration per unit of action noise, and too high an action-rate cost per radian.
ARM_WAIST_ACTION_SCALE_FREQ_HZ = 1.0

KANGAROO_GRIPPERS_ACTION_SCALE: dict[str, float] = {}
for a in KANGAROO_GRIPPERS_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None and a.armature is not None
  for n in a.target_names_expr:
    if not n.startswith("leg_"):
      s = a.armature * (2.0 * math.pi * ARM_WAIST_ACTION_SCALE_FREQ_HZ) ** 2
    KANGAROO_GRIPPERS_ACTION_SCALE[n] = 0.25 * e / s

##
# Closed-chain leg.
##

# The femur (hip to knee) and the tibia (knee to ankle) are both this long, so they
# close an isosceles triangle on the slide and their joint angles follow from the
# leg length alone.
KANGAROO_LEG_LINK_LENGTH = 0.38  # [m]


def kangaroo_passive_leg_joints(
  joint_pos: dict[str, torch.Tensor], joint_vel: dict[str, torch.Tensor]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  """Position and velocity of the passive femur and knee joints that close the leg.

  Args:
    joint_pos: Joint positions by joint name; holds both ``leg_*_length_joint``.
    joint_vel: Joint velocities by joint name, likewise.

  Returns:
    (position, velocity) of each ``leg_*_femur_joint`` and ``leg_*_knee_joint``,
    by name.
  """
  state = {}
  for side in ("left", "right"):
    length = joint_pos[f"leg_{side}_length_joint"]
    rate = joint_vel[f"leg_{side}_length_joint"]
    sin = (length / (2.0 * KANGAROO_LEG_LINK_LENGTH)).clamp(0.0, 0.999)
    femur = torch.asin(sin)
    femur_vel = rate / (2.0 * KANGAROO_LEG_LINK_LENGTH * torch.sqrt(1.0 - sin * sin))
    state[f"leg_{side}_femur_joint"] = (femur, femur_vel)
    state[f"leg_{side}_knee_joint"] = (2.0 * femur, 2.0 * femur_vel)
  return state


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_kangaroo_grippers_robot_cfg())

  viewer.launch(robot.spec.compile())
