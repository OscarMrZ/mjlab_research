"""Tests for multi-motion tracking functionality."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML
from mjlab.tasks.tracking.mdp.commands import (
  MotionCommand,
  MotionLoader,
  _resolve_motion_file,
)
from mjlab.utils.forward_kinematics import ForwardKinematics


@pytest.fixture(scope="module")
def fk() -> ForwardKinematics:
  """Forward kinematics of the G1 to its first three bodies."""
  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  return ForwardKinematics(
    model,
    torch.arange(1, model.nbody),
    torch.from_numpy(model.jnt_qposadr[1:]),
    torch.tensor([0, 1, 2]),
  )


@pytest.fixture
def create_motion_file(tmp_path, fk):
  """Factory to create motion NPZ files of random root and joint states of the
  robot of ``fk``."""

  def _create(name: str, num_frames: int) -> str:
    root_pos, root_lin_vel, root_ang_vel = torch.randn(3, num_frames, 3)
    root_quat = torch.nn.functional.normalize(torch.randn(num_frames, 4), dim=-1)
    joint_pos, joint_vel = torch.randn(2, num_frames, len(fk.joint_names))
    path = tmp_path / f"{name}.npz"
    np.savez(
      path,
      joint_pos=joint_pos.numpy(),
      joint_vel=joint_vel.numpy(),
      root_pos_w=root_pos.numpy(),
      root_quat_w=root_quat.numpy(),
      root_lin_vel_w=root_lin_vel.numpy(),
      root_ang_vel_w=root_ang_vel.numpy(),
      joint_names=fk.joint_names,
    )
    return str(path)

  return _create


# _resolve_motion_file tests


def test_resolve_single_path(create_motion_file):
  path = create_motion_file("motion", num_frames=100)
  assert _resolve_motion_file(path) == [path]


def test_resolve_list_of_paths(create_motion_file):
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
  ]
  assert _resolve_motion_file(paths) == paths


def test_resolve_glob_pattern(create_motion_file, tmp_path):
  for i in range(3):
    create_motion_file(f"motion_{i}", num_frames=50)
  pattern = str(tmp_path / "motion_*.npz")
  result = _resolve_motion_file(pattern)
  assert len(result) == 3


def test_resolve_glob_no_match(tmp_path):
  pattern = str(tmp_path / "nonexistent_*.npz")
  with pytest.raises(ValueError, match="No files matched"):
    _resolve_motion_file(pattern)


# MotionLoader tests


def test_loader_single_motion(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=100)
  loader = MotionLoader(path, fk)

  assert loader.time_step_total == 100
  assert loader.joint_pos.shape == (100, len(fk.joint_names))
  assert loader.root_pos_w.shape == (100, 3)


def test_loader_multiple_motions_concatenated(create_motion_file, fk):
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
    create_motion_file("c", num_frames=75),
  ]
  loader = MotionLoader(paths, fk)

  # Should be concatenated: 50 + 100 + 75 = 225 frames
  assert loader.time_step_total == 225
  assert loader.joint_pos.shape == (225, len(fk.joint_names))
  assert loader.root_pos_w.shape == (225, 3)


def test_loader_motion_ends(create_motion_file, fk):
  """Verify motion_ends tracks cumulative boundaries."""
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
    create_motion_file("c", num_frames=75),
  ]
  loader = MotionLoader(paths, fk)

  # motion_ends should be cumulative: [50, 150, 225]
  assert loader.motion_ends.tolist() == [50, 150, 225]


def test_loader_concatenation_preserves_data(create_motion_file, fk):
  """Verify that concatenated data comes from the correct motion files."""
  paths = [
    create_motion_file("a", num_frames=50),
    create_motion_file("b", num_frames=100),
  ]
  loader = MotionLoader(paths, fk)

  # First 50 frames from motion_a, frames 50-149 from motion_b.
  joint_pos = [torch.from_numpy(np.load(path)["joint_pos"]) for path in paths]
  assert torch.equal(loader.joint_pos[:50], joint_pos[0])
  assert torch.equal(loader.joint_pos[50:], joint_pos[1])


def test_loader_rejects_other_joints(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=20)
  data = dict(np.load(path))
  data["joint_names"] = data["joint_names"][::-1]
  np.savez(path, **data)
  with pytest.raises(ValueError, match="robot's"):
    MotionLoader(path, fk)


def test_full_joint_state_spreads_the_motion_joints():
  """The joints the motions hold land on their robot columns, the others at zero."""
  command = SimpleNamespace(
    joint_ids=torch.tensor([0, 2, 3]), robot=SimpleNamespace(num_joints=5)
  )
  joint_pos, joint_vel = torch.rand(2, 4, 3) + 1.0
  pos, vel = MotionCommand.full_joint_state(command, joint_pos, joint_vel)  # ty: ignore[invalid-argument-type]
  assert pos.shape == vel.shape == (4, 5)
  assert torch.equal(pos[:, [0, 2, 3]], joint_pos)
  assert torch.equal(vel[:, [0, 2, 3]], joint_vel)
  assert not pos[:, [1, 4]].any() and not vel[:, [1, 4]].any()


# Held clips


def _rest(path: str, frames: tuple[int, ...], lying: tuple[int, ...] = ()) -> None:
  """Put ``frames`` of the motion file at rest, upright at standing height, and
  those of ``lying`` on their side."""
  clip = dict(np.load(path))
  clip["root_pos_w"][:, 2] = 0.9
  for frame in frames:
    side = frame in lying
    clip["root_quat_w"][frame] = [0.5**0.5, 0.5**0.5, 0, 0] if side else [1, 0, 0, 0]
    for name in ("root_lin_vel_w", "root_ang_vel_w", "joint_vel"):
      clip[name][frame] = 0.0
  np.savez(path, **clip)


def test_held_clips_hold_a_still_frame_without_velocity(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=20)
  # Frames 3 and 11 are the only still, upright ones: 7 is still, but lying down.
  _rest(path, frames=(3, 7, 11), lying=(7,))
  motion = MotionLoader(path, fk)

  loader = MotionLoader(path, fk, held_clips=16, held_steps=5)

  assert loader.time_step_total == loader.joint_pos.shape[0] == 20 + 16 * 5
  assert loader.held_starts.tolist() == list(range(20, 100, 5))
  assert loader.motion_ends.tolist() == list(range(20, 105, 5))
  assert len(loader.motion_names) == 17
  held = zip(loader.held_starts.tolist(), loader.motion_names[1:], strict=True)
  for start, name in held:
    source = int(name.removeprefix("held_"))
    assert source in (3, 11)
    clip = slice(start, start + 5)
    assert torch.equal(loader.root_pos_w[clip], motion.root_pos_w[source].expand(5, 3))
    assert torch.equal(
      loader.root_quat_w[clip], motion.root_quat_w[source].expand(5, 4)
    )
    assert torch.equal(loader.joint_pos[clip], motion.joint_pos[source].expand(5, -1))
    assert not loader.root_lin_vel_w[clip].any()
    assert not loader.root_ang_vel_w[clip].any()
    assert not loader.joint_vel[clip].any()
  # The motions themselves are as they were, and there are none held unless asked.
  assert torch.equal(loader.joint_pos[:20], motion.joint_pos)
  assert motion.held_starts.numel() == 0


def test_held_clips_need_a_still_frame(create_motion_file, fk):
  path = create_motion_file("motion", num_frames=20)

  with pytest.raises(ValueError, match="no still, upright frame"):
    MotionLoader(path, fk, held_clips=2, held_steps=5)
