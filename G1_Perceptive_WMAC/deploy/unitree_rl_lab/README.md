# Unitree G1 controller integration

This patch integrates the published `sensor_transform` policy with Unitree's 29-DoF G1 controller. It adds the Hiking stand and policy states, reconstructs torso-frame gyro and gravity from the pelvis IMU and waist encoders, and subscribes to a depth image over DDS. The policy's `params/deploy.yaml` supplies the joint sign conventions, gains and observation order. The patch applies to [unitreerobotics/unitree_rl_lab at 4960b84](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3). The included upstream-derived changes retain Unitree's [Apache 2.0 license](LICENSE-unitree-rl-lab.txt).

## Build on Linux x86-64

Start in `Humoind/G1_Perceptive_WMAC` after downloading the policy files. The upstream build uses its bundled Linux x86-64 ONNX Runtime. Install Unitree SDK2 and the C++ dependencies in the [upstream setup instructions](https://github.com/unitreerobotics/unitree_rl_lab/tree/4960b84732b0c2ec593dccbfe963fda1bcd7b1e3#setup). On an ARM onboard computer, this upstream build needs the corresponding ARM libraries and CMake paths. A Linux x86-64 host can run the controller over Ethernet.

```bash
project_dir="$PWD"
integration="$project_dir/deploy/unitree_rl_lab"
policy="$project_dir/policies/sensor_transform"
controller="$HOME/unitree_rl_lab-hiking"

git clone https://github.com/unitreerobotics/unitree_rl_lab.git "$controller"
git -C "$controller" checkout --detach 4960b84732b0c2ec593dccbfe963fda1bcd7b1e3
git -C "$controller" apply --check "$integration/g1-hiking.patch"
git -C "$controller" apply "$integration/g1-hiking.patch"

robot_dir="$controller/deploy/robots/g1_29dof"
dest="$robot_dir/config/policy/hiking/sensor_transform"
mkdir -p "$dest"
cp -R "$policy/exported" "$policy/params" "$dest/"
cmake -S "$robot_dir" -B "$robot_dir/build"
cmake --build "$robot_dir/build" --parallel 2
```

If several exports are installed under `config/policy/hiking`, the upstream loader picks the last directory in lexical order. Set `FSM.Hiking.policy_dir` to `config/policy/hiking/sensor_transform` to select this export explicitly.

## Offline observation check

The fixture uses synthetic IMU, joint and depth samples. It feeds the same inputs into the C++ controller observation manager and the project's NumPy interface, without connecting to DDS or sending motor commands. Run with the variables above and a Python environment containing NumPy.

```bash
cmake -S "$integration/test" -B "$controller/build-hiking-test" \
  -DUNITREE_RL_LAB_ROOT="$controller"
cmake --build "$controller/build-hiking-test" --parallel 2
python "$integration/test/check_hiking_obs.py" \
  --framework "$project_dir" \
  --binary "$controller/build-hiking-test/hiking_obs_test" \
  --deploy-yaml "$dest/params/deploy.yaml"
```

## Depth input on the robot

Supply a camera-side publisher on DDS domain 0, topic `rt/hiking/depth`. The message is an organized `sensor_msgs/msg/PointCloud2` of width 64, height 36. Each pixel contains one little-endian float32 field named `z`, in metres of **image-plane depth**, with `point_step=4`, `row_step=256`, and non-finite or nonpositive for a missing return. Use the rays and calibrated projection in `params/depth_camera.npz` and `params/deployment_contract.json`. The neural network handles cropping and blurring. The controller takes the newest image every 20 ms.

This export uses the v3 `g1_hiking_calibrated_v1` projection derived from the 848 × 480 calibration. A 480 × 270 camera stream needs an explicit mapping from its own intrinsics to the v3 rays. The v4/v5 Hiking-pinhole lookup targets a different projection. This integration supplies the depth subscriber. The real-camera publisher and acquisition latency still require integration on the robot. The controller can keep the last frame after a stream interruption, so verify reception and freshness independently before motor operation.

With the stock Unitree foot model, this checkpoint walked on flat ground and fell in every uneven-terrain MuJoCo trial (five runs per terrain setting). The offline observation check verifies interface consistency; physical operation is untested.
