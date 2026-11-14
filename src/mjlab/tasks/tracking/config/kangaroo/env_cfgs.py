"""PAL Kangaroo flat tracking environment configurations."""

from dataclasses import dataclass, fields

import torch

from mjlab.asset_zoo.robots import (
  KANGAROO_ACTION_SCALE,
  KANGAROO_GRIPPERS_ACTION_SCALE,
  get_kangaroo_grippers_robot_cfg,
  get_kangaroo_robot_cfg,
)
from mjlab.asset_zoo.robots.pal_kangaroo.kangaroo_constants import (
  KANGAROO_ACTUATORS,
  KANGAROO_GRIPPERS_ACTUATORS,
  kangaroo_passive_leg_joints,
)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


class KangarooMotionCommand(MotionCommand):
  """A motion command for motions that hold the actuated joints only."""

  def full_joint_state(
    self, joint_pos: torch.Tensor, joint_vel: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Set the passive femur and knee joints from the leg length.

    Left at zero they would open the knee linkage, and a reset would leave the
    equality constraint to snap it shut.
    """
    pos, vel = super().full_joint_state(joint_pos, joint_vel)
    names = self.robot.joint_names
    passive = kangaroo_passive_leg_joints(
      dict(zip(names, pos.unbind(-1), strict=True)),
      dict(zip(names, vel.unbind(-1), strict=True)),
    )
    for name, (passive_pos, passive_vel) in passive.items():
      pos[..., names.index(name)] = passive_pos
      vel[..., names.index(name)] = passive_vel
    return pos, vel


@dataclass(kw_only=True)
class KangarooMotionCommandCfg(MotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> KangarooMotionCommand:
    return KangarooMotionCommand(self, env)


def pal_kangaroo_flat_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Kangaroo (4-DoF arms) flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_ACTION_SCALE

  base_cmd = cfg.commands["motion"]
  assert isinstance(base_cmd, MotionCommandCfg)
  motion_cmd = KangarooMotionCommandCfg(
    **{f.name: getattr(base_cmd, f.name) for f in fields(base_cmd)}
  )
  cfg.commands["motion"] = motion_cmd
  # The motions hold the actuated joints: the femur and knee joints are passive.
  motion_cmd.joint_names = tuple(
    name for a in KANGAROO_ACTUATORS for name in a.target_names_expr
  )
  motion_cmd.anchor_body_name = "pelvis_2_link"
  # The G1's tracked bodies mapped to Kangaroo. The G1 knee has no counterpart:
  # the leg is a hip-ankle slide. The hand is the tip of the rigid forearm.
  motion_cmd.body_names = (
    "base_link",
    "pelvis_2_link",
    "leg_left_3_link",
    "leg_left_5_link",
    "leg_right_3_link",
    "leg_right_5_link",
    "arm_left_2_link",
    "arm_left_4_link",
    "arm_left_tip_link",
    "arm_right_2_link",
    "arm_right_4_link",
    "arm_right_tip_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot(0|2|4|6|8|10)_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_left_5_link",
    "leg_right_5_link",
    "arm_left_tip_link",
    "arm_right_tip_link",
  )

  cfg.viewer.body_name = "pelvis_2_link"

  # The knee linkage and the self collisions need more room than the G1.
  cfg.sim.nconmax = 50
  cfg.sim.njmax = 1024

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def pal_kangaroo_grippers_flat_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Kangaroo grippers flat terrain tracking configuration.

  The 4-DoF configuration with the 7-DoF arms: their wrist is the hand.
  """
  cfg = pal_kangaroo_flat_tracking_env_cfg(play=play)

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, KangarooMotionCommandCfg)
  motion_cmd.joint_names = tuple(
    name for a in KANGAROO_GRIPPERS_ACTUATORS for name in a.target_names_expr
  )
  hand = {
    "arm_left_tip_link": "arm_left_7_link",
    "arm_right_tip_link": "arm_right_7_link",
  }
  motion_cmd.body_names = tuple(hand.get(b, b) for b in motion_cmd.body_names)
  ee_body_pos = cfg.terminations["ee_body_pos"].params
  ee_body_pos["body_names"] = tuple(hand.get(b, b) for b in ee_body_pos["body_names"])

  return cfg
