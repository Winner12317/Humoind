# EngineAI PM01 部署文件

本次目标是 PM01 EDU 的 SuperMimic + AMP v3，训练至 **50,400 次迭代**。训练权重在 [policy/wamp/v3_50400.pt](../policy/wamp/v3_50400.pt)。推理用的 [ONNX、MNN 和逐关节配置](policy_wamp_v3/) 来自这一个检查点。其余四个检查点在 [版本列表](README.md) 中。默认执行 `./run.sh` 会选择 v1，验证这个版本请显式指定 `policy_wamp_v3`。

## 下载与核对

在已有的 Humoind 仓库根目录执行。需要 Git LFS。

```bash
git lfs install --local
GIT_LFS_SKIP_SMUDGE=1 git pull --ff-only
git lfs pull --include="policy/wamp/v3_50400.pt"
sha256sum -c pm01_sim2sim/policy_wamp_v3/SHA256SUMS
```

## 先在 MuJoCo 复现

已经安装 SuperAMP 环境时，在 Linux 上运行：

```bash
cd pm01_sim2sim
./run.sh --policy-dir policy_wamp_v3 --headless
./run.sh --policy-dir policy_wamp_v3
```

也可以用独立的 Python 3.11 环境安装 `numpy==2.4.6 mujoco==3.12.0 onnxruntime==1.31.0`，再用 `python sim2sim.py --policy-dir policy_wamp_v3 --headless`。键盘在运行脚本的终端输入，W/S 控制前后速度，Q/E 转向，空格停步。R 和 F 只在仿真中重置姿态。

| 50,400 版验证项目 | 结果 |
|---|---|
| ONNX 对原 PyTorch，4,096 组输入 | 最大动作绝对误差 5.25 × 10⁻⁶ |
| MNN 对 ONNX，982 帧仿真观测，EngineAI SDK 自带的 MNN 2.9.5，Linux x86-64 CPU | 最大动作绝对误差 6.68 × 10⁻⁶ |
| 58 秒行走及 10 次推扰 | ONNX 和 MNN 均零跌倒 |
| MuJoCo 独立生成的 8 个倒地姿势 | ONNX 和 MNN 均成功 8/8，平均 2.71 秒 |
| 本地授权参考动作中的 8 个姿势 | ONNX 和 MNN 均成功 8/8，平均约 2.56 秒 |

公开仓库默认可以在没有动作库时运行。用 `--recovery-poses generic` 固定使用仿真生成的姿势。已有获授权的参考动作时，可用 `--recovery-poses reference` 验证参考组。以上数字均为同一随机种子的一轮仿真。

[一般倒地姿势 ONNX 报告](results/wamp_v3_50400_report_sdk_generic.json) · [参考姿势 ONNX 报告](results/wamp_v3_50400_report_sdk_reference.json) · [MNN 闭环报告](results/wamp_v3_50400_report_sdk_mnn_generic.json) · [MNN 输出对照](results/wamp_v3_50400_mnn_parity.json)

## 对接真机控制器

**目前已验证桌面推理和 MuJoCo 闭环。真机仍需接入适配该策略的 EngineAI 控制器。** 官方 Native SDK 的 PM01 `rl_lab` 状态使用 100 Hz、15 帧历史及不同的观测顺序。Robot Session SDK 的官方 PM01 行走示例还少输出一个关节。仅替换示例的模型文件无法运行这份策略。仓库中的 Python 键盘程序只操作 MuJoCo。

控制器实现可依据 [policy.json](policy_wamp_v3/policy.json) 和 [sim2sim.py](sim2sim.py) 对照：

- 每 20 ms 在 `LINK_BASE` 坐标系读取陀螺仪和投影重力；每帧依次拼接角速度、投影重力、速度指令、相对于默认姿态的关节角、关节速度和上一帧未缩放的网络动作。
- 保留从旧到新的四帧，每帧 78 个数，总输入 312 维。首次进入策略时，上一动作为零，四帧均填充同一份初始观测。
- 按配置中 `joint_names` 的顺序映射 23 维输出，目标关节角为 `default_joint_pos + action_scale × action`。使用该检查点的逐关节增益。头部偏航由独立通道控制。
- 实机应先用机器人的关节和 IMU 实测记录做无电机输出的回放，对齐观测、关节顺序、时间戳和目标关节角，再在带保护的测试条件下验证。当前版本尚无 ARM 或真机运行记录。

[EngineAI Native SDK（所对照版本）](https://github.com/engineai-robotics/engineai_robotics_native_sdk/tree/6065bb219e3c15cbf93ab240fcfd737d6f007816) · [EngineAI Robot Session SDK 行走示例](https://github.com/engineai-robotics/engineai_robotics_sdk/blob/98827ab8b18e6aaea2e4e43197f0dbc38d60fb20/cpp/examples/scenarios/joint_bridge_stand_and_walk.cpp)

## 重新导出模型

从 `pm01_sim2sim/` 运行，在单独目录生成 ONNX 和配置。导出需要 SuperAMP 训练环境。MNN 文件使用 MNNConvert 3.3.0（`83333199`），保留 FP32 并关闭 GELU 近似。ONNX 和 MNN 输出均包含观测归一化与动作限幅。

```bash
./run.sh export --checkpoint ../policy/wamp/v3_50400.pt --out /tmp/pm01-v3-50400
cd /tmp/pm01-v3-50400
MNNConvert -f ONNX --modelFile policy.onnx --MNNModel policy.mnn \
  --targetVersion 2.9 --useGeluApproximation 0 --bizCode MNNTest
```
