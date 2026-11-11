"""PAL Kangaroo grippers flat tracking environment configurations."""

from dataclasses import dataclass, fields

import torch

from mjlab.asset_zoo.robots import (
  KANGAROO_GRIPPERS_ACTION_SCALE,
  get_kangaroo_grippers_robot_cfg,
)
from mjlab.asset_zoo.robots.pal_kangaroo.kangaroo_constants import (
  kangaroo_passive_leg_joints,
)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


class KangarooMotionCommand(MotionCommand):
  """A motion command that keeps the passive knee linkage closed at a reset."""

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Write the reference state with the femur and knee joints set from the leg
    length.

    The base class perturbs and clips every joint on its own, which would open the
    linkage and leave the equality constraint to snap it shut. Here the linkage
    follows the clipped leg length, and is not clipped itself.
    """
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    joint_vel = joint_vel.clone()
    names = self.robot.joint_names
    passive = kangaroo_passive_leg_joints(
      dict(zip(names, joint_pos.unbind(-1), strict=True)),
      dict(zip(names, joint_vel.unbind(-1), strict=True)),
    )
    for name, (pos, vel) in passive.items():
      joint_pos[:, names.index(name)] = pos
      joint_vel[:, names.index(name)] = vel
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)


@dataclass(kw_only=True)
class KangarooMotionCommandCfg(MotionCommandCfg):
  def build(self, env: ManagerBasedRlEnv) -> KangarooMotionCommand:
    return KangarooMotionCommand(self, env)


def pal_kangaroo_grippers_flat_tracking_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create PAL Kangaroo grippers flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_kangaroo_grippers_robot_cfg()}

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
  joint_pos_action.scale = KANGAROO_GRIPPERS_ACTION_SCALE

  base_cmd = cfg.commands["motion"]
  assert isinstance(base_cmd, MotionCommandCfg)
  motion_cmd = KangarooMotionCommandCfg(
    **{f.name: getattr(base_cmd, f.name) for f in fields(base_cmd)}
  )
  cfg.commands["motion"] = motion_cmd
  motion_cmd.anchor_body_name = "pelvis_2_link"
  # The G1's tracked bodies mapped to Kangaroo. The G1 knee has no counterpart:
  # the leg is a hip-ankle slide.
  motion_cmd.body_names = (
    "base_link",
    "pelvis_2_link",
    "leg_left_3_link",
    "leg_left_5_link",
    "leg_right_3_link",
    "leg_right_5_link",
    "arm_left_2_link",
    "arm_left_4_link",
    "arm_left_7_link",
    "arm_right_2_link",
    "arm_right_4_link",
    "arm_right_7_link",
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot(0|2|4|6|8|10)_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "leg_left_5_link",
    "leg_right_5_link",
    "arm_left_7_link",
    "arm_right_7_link",
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
