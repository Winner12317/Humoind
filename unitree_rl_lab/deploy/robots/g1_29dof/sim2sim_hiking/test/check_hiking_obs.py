"""Compare g1_ctrl's Hiking observation vectors with the training-side reference.

Generates a random but realistic input sequence (Unitree-convention joints and
pelvis IMU, depth images with misses), runs ``hiking_obs_test`` (unitree_rl_lab's
own ObservationManager with the exported deploy.yaml) and rebuilds every vector
with ``G1_Perceptive_WMAC/src/wmac_perceptive/unitree_interface.py``.

    python check_hiking_obs.py --framework <G1_Perceptive_WMAC> --binary build/hiking_obs_test \
        --deploy-yaml ../../config/policy/hiking/<name>/params/deploy.yaml
"""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--deploy-yaml", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    args = parser.parse_args()
    sys.path.insert(0, str(args.framework))
    from src.wmac_perceptive import unitree_interface as ui

    rng = np.random.default_rng(0)
    n, pixels = args.steps, 36 * 64
    quat = rng.normal(size=(n, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    gyro = rng.uniform(-2, 2, (n, 3))
    q_u = ui.UNITREE_JOINT_SIGN * ui.default_joint_positions() + rng.uniform(-.4, .4, (n, 29))
    dq_u = rng.uniform(-4, 4, (n, 29))
    action = rng.normal(size=(n, 29))
    depth = rng.uniform(.2, 4., (n, pixels))
    depth[rng.random((n, pixels)) < .05] = np.inf
    depth[rng.random((n, pixels)) < .01] = np.nan
    with tempfile.TemporaryDirectory() as folder:
        inputs, output = Path(folder) / "inputs.bin", Path(folder) / "obs.bin"
        with open(inputs, "wb") as f:
            np.array([n, pixels], np.int32).tofile(f)
            np.concatenate((quat, gyro, q_u, dq_u, action, depth), 1).astype(np.float32).tofile(f)
        print(subprocess.run([str(args.binary), str(args.deploy_yaml), str(inputs), str(output)],
                             check=True, capture_output=True, text=True).stdout.strip())
        raw = np.fromfile(output, np.int32, 2)
        got = np.fromfile(output, np.float32, offset=8).reshape(raw[0], raw[1])
    # Single-precision inputs, as the C++ side received them.
    quat, gyro, q_u, dq_u, action, depth = (x.astype(np.float32).astype(float)
                                           for x in (quat, gyro, q_u, dq_u, action, depth))
    iface = ui.UnitreeG1PolicyInterface()
    worst = {"proprio": 0., "depth": 0.}
    previous = np.zeros(29)
    for k in range(n):
        torso_quat, torso_gyro = ui.torso_from_pelvis(quat[k], gyro[k], q_u[k, 12:15], dq_u[k, 12:15])
        command = got[k, 7 * 3 + 2 * 24: 8 * 3 + 2 * 24]       # joystick filter output, newest sample
        # Depths are >= 0.2 m, so the 0.1 m ray rule (applied by the simulator) never triggers here.
        frame = ui.depth_frame(depth[k].reshape(36, 64))
        kwargs = dict(torso_quat=torso_quat, torso_gyro=torso_gyro, q_unitree=q_u[k], dq_unitree=dq_u[k],
                      command=command, depth_frame=frame)
        if k == 0:
            iface.reset(**kwargs)
            # The reset fill also used this command, and the newest sample carries the same values.
        iface.step(**kwargs, action=previous)
        if k == 0:
            # g1_ctrl's reset fill recorded the command the joystick filter had at reset time.
            iface._proprio[:-1, 6:9] = got[0, 48:51]
        expected = iface.policy_input()[0]
        if k > 0:
            # Earlier command samples are whatever the filter produced then (already checked above).
            for h in range(8):
                expected[48 + h * 3: 51 + h * 3] = got[k, 48 + h * 3: 51 + h * 3]
        worst["proprio"] = max(worst["proprio"], float(np.abs(got[k, :768] - expected[:768]).max()))
        scale = np.maximum(1., np.abs(expected[768:]))
        worst["depth"] = max(worst["depth"], float((np.abs(got[k, 768:] - expected[768:]) / scale).max()))
        previous = action[k]
    print(f"compared {n} steps x {got.shape[1]} values: max |proprio error| {worst['proprio']:.2e}, "
          f"max relative depth error {worst['depth']:.2e}")
    if worst["proprio"] > 1e-4 or worst["depth"] > 1e-5:
        raise SystemExit("MISMATCH")
    print("OK: g1_ctrl builds the same policy input as training")


if __name__ == "__main__":
    main()
