"""Convert a pal_mjlab Kangaroo grippers motion library to a packed motion directory.

The library is a folder with ``clips.json`` (``fps``, ``joint_names`` and the
clips' ``name``, ``start`` and ``length``), ``qpos.npy`` and ``qvel.npy``: one row
per frame in MuJoCo's free-joint layout, with the actuated joints only. The packed
directory is the one ``pack_motions`` writes and ``MotionLoader`` reads. The frames
are kept as they are, with one change: the root angular velocity goes from the
root frame to the world frame.

Example:
  uv run python -m mjlab.scripts.kangaroo_library_to_motions \
    --input ~/kangaroo/libraries/smoke --output ~/kangaroo/packed/smoke
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from mjlab.asset_zoo.robots.pal_kangaroo.kangaroo_constants import get_spec
from mjlab.utils.lab_api.math import quat_apply


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", required=True, help="motion library folder")
  parser.add_argument("--output", required=True, help="output packed directory")
  args = parser.parse_args()

  library = os.path.expanduser(args.input)
  with open(os.path.join(library, "clips.json")) as fh:
    meta = json.load(fh)
  if meta["fps"] != 50:
    raise SystemExit(f"[convert] the library is at {meta['fps']} fps, not 50")
  clips = meta["clips"]
  frames = np.concatenate(
    [np.arange(c["start"], c["start"] + c["length"]) for c in clips]
  )
  qpos = torch.from_numpy(np.load(os.path.join(library, "qpos.npy"))[frames])
  qvel = torch.from_numpy(np.load(os.path.join(library, "qvel.npy"))[frames])

  joint_pos = dict(zip(meta["joint_names"], qpos[:, 7:].unbind(-1), strict=True))
  joint_vel = dict(zip(meta["joint_names"], qvel[:, 6:].unbind(-1), strict=True))
  # The library's joints, in the robot's order.
  joint_names = [joint.name for joint in get_spec().joints if joint.name in joint_pos]
  if len(joint_names) != len(joint_pos):
    raise SystemExit("[convert] the library has joints the robot does not")

  fields = {
    "joint_pos": torch.stack([joint_pos[name] for name in joint_names], dim=-1),
    "joint_vel": torch.stack([joint_vel[name] for name in joint_names], dim=-1),
    "root_pos_w": qpos[:, :3],
    "root_quat_w": qpos[:, 3:7],
    "root_lin_vel_w": qvel[:, :3],
    "root_ang_vel_w": quat_apply(qpos[:, 3:7], qvel[:, 3:6]),
  }

  out = os.path.expanduser(args.output)
  os.makedirs(out, exist_ok=True)
  with open(os.path.join(out, "joint_names.json"), "w") as fh:
    json.dump(joint_names, fh)
  with open(os.path.join(out, "motion_names.json"), "w") as fh:
    json.dump([c["name"] for c in clips], fh)
  lengths = np.array([c["length"] for c in clips], dtype=np.int64)
  np.save(os.path.join(out, "motion_ends.npy"), np.cumsum(lengths))
  for name, value in fields.items():
    np.save(os.path.join(out, f"{name}.npy"), value.numpy().astype(np.float32))
  print(f"[convert] {len(clips)} clips, {len(frames)} frames -> {out}", flush=True)


if __name__ == "__main__":
  main()
