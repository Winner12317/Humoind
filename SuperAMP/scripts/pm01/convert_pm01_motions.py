"""Convert the retargeted PM01 clips to this repository's motion NPZ layout.

    python scripts/pm01/convert_pm01_motions.py

Input : ``<workspace>/pm01_retarget/pm01_npz_50hz/{WalkandRun,Recovery}/*.npz``
        (``fps``, ``root_pos``, ``root_rot`` xyzw, ``dof_pos`` J00..J23, ``dof_names``),
        the 50 Hz set whose frames line up one-to-one with the G1 clips in
        ``src/assets/motions/g1/amp``.
Output: ``src/assets/motions/pm01/amp/{WalkandRun,Recovery}/*.npz`` with the keys the
        AMP loaders read (``fps``, ``joint_pos``, ``joint_vel``, ``body_pos_w``,
        ``body_quat_w`` wxyz, ``body_lin_vel_w``, ``body_ang_vel_w``), bodies and
        joints in the order of the PM01 model in ``src/assets/robots/engineai_pm01``.

The clips are already at 50 Hz, so nothing is resampled.  Velocities come from the
same finite differences ``scripts/csv_to_npz.py`` uses for the G1 clips (its
``MotionLoader._compute_velocities``), and body states from the same mjlab forward
pass.  The welded head joint (J23, zero in every clip) is dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from scripts.csv_to_npz import MotionLoader
from src.assets.robots import get_pm01_robot_cfg
from src.assets.robots.engineai_pm01.pm01_constants import PM01_JOINT_NAMES, PM01_WELDED_JOINT_NAMES

GROUPS = ("WalkandRun", "Recovery")
MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")


class CompactMotion(MotionLoader):
  """A clip that is already at the output rate; only the velocity code is reused."""

  def __init__(self, path: Path, device: str, joint_range: np.ndarray):
    with np.load(path, allow_pickle=False) as data:
      fps = float(np.asarray(data["fps"]).reshape(-1)[0])
      root_pos, root_rot, dof_pos = data["root_pos"], data["root_rot"], data["dof_pos"]
      names = [str(name) for name in data["dof_names"]]
    if names != [*PM01_JOINT_NAMES, *PM01_WELDED_JOINT_NAMES]:
      raise ValueError(f"{path}: unexpected joint order {names}")
    welded = dof_pos[:, len(PM01_JOINT_NAMES):]
    if np.abs(welded).max() > 1e-6:
      raise ValueError(f"{path}: the welded joints move in this clip (max {np.abs(welded).max():.4f} rad)")
    dof_pos = dof_pos[:, : len(PM01_JOINT_NAMES)]
    excess = np.maximum(joint_range[:, 0] - dof_pos, dof_pos - joint_range[:, 1]).max()
    if excess > 1e-5:
      raise ValueError(f"{path}: joint positions leave the model's range by {excess:.2e} rad")
    dof_pos = np.clip(dof_pos, joint_range[:, 0], joint_range[:, 1])

    self.motion_file = str(path)
    self.input_fps = self.output_fps = fps
    self.input_dt = self.output_dt = 1.0 / fps
    self.current_idx = 0
    self.device = device
    self.line_range = None
    self.output_frames = self.input_frames = dof_pos.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

    def tensor(values: np.ndarray) -> torch.Tensor:
      return torch.as_tensor(values, dtype=torch.float32, device=device)

    self.motion_base_poss = tensor(root_pos)
    self.motion_base_rots = tensor(root_rot[:, [3, 0, 1, 2]])  # xyzw -> wxyz
    self.motion_dof_poss = tensor(dof_pos)
    self._compute_velocities()


def convert(sim: Simulation, scene: Scene, motion: CompactMotion, output: Path) -> None:
  robot: Entity = scene["robot"]
  joint_ids = robot.find_joints(PM01_JOINT_NAMES, preserve_order=True)[0]
  log: dict[str, list] = {key: [] for key in MOTION_KEYS}
  scene.reset()
  for _ in range(motion.output_frames):
    (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), _ = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = base_rot
    root_states[:, 7:10] = base_lin_vel
    root_states[:, 10:] = base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = dof_pos
    joint_vel[:, joint_ids] = dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())
    torch.testing.assert_close(robot.data.body_link_lin_vel_w[0, 0], base_lin_vel[0])
    torch.testing.assert_close(robot.data.body_link_ang_vel_w[0, 0], base_ang_vel[0])

  output.parent.mkdir(parents=True, exist_ok=True)
  np.savez(str(output), fps=[motion.output_fps], **{key: np.stack(values) for key, values in log.items()})


def main(
  input_dir: str = str(REPO.parent / "pm01_retarget" / "pm01_npz_50hz"),
  output_dir: str = str(REPO / "src" / "assets" / "motions" / "pm01" / "amp"),
  device: str = "cuda:0",
) -> None:
  """Convert every clip under input_dir/{WalkandRun,Recovery}."""
  source, target = Path(input_dir), Path(output_dir)
  clips = [path for group in GROUPS for path in sorted((source / group).glob("*.npz"))]
  if not clips:
    raise FileNotFoundError(f"No clips under {source}/{{{','.join(GROUPS)}}}")

  scene = Scene(
    SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1, entities={"robot": get_pm01_robot_cfg()}),
    device=device,
  )
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / 50.0
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  if tuple(robot.joint_names) != PM01_JOINT_NAMES:
    raise RuntimeError(f"PM01 model joint order changed: {robot.joint_names}")
  joint_range = robot.data.joint_pos_limits[0].cpu().numpy()

  for path in clips:
    motion = CompactMotion(path, sim.device, joint_range)
    if motion.output_fps != 50.0:
      raise ValueError(f"{path}: expected a 50 Hz clip, found {motion.output_fps} Hz")
    output = target / path.parent.name / path.name
    convert(sim, scene, motion, output)
    print(f"{path.parent.name}/{path.stem}: {motion.output_frames} frames -> {output}")
  print(f"bodies ({len(robot.body_names)}): {', '.join(robot.body_names)}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
