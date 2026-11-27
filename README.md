# Humoind

使用 SuperAMP 在 MuJoCo/mjlab 中训练 Unitree G1 和 EngineAI PM01。SuperAMP 支持 PPO + AMP 和 WMAC。PM01 使用由 G1 参考动作重定向并修正足部接触后的动作。

- **[PM01 真机对接与仿真验证（v3_50400）](pm01_sim2sim/PM01_DEPLOYMENT.md)** · [五个 PM01 策略及对应 ONNX](pm01_sim2sim/README.md)
- [SuperAMP 训练框架](SuperAMP/README_zh.md)和 [PM01 训练配置](SuperAMP/docs/pm01/README.md)
- [PM01 重定向、接触修正和逐帧查看器](pm01_retarget/README.md)
- [平地 G1 的 WMAC＋AMP 复现手册](docs/平地%20G1%20的%20WMAC＋AMP%20复现手册.md)
- [G1 视觉越野行走训练与快速开始](G1_Perceptive_WMAC/reproduce/QUICKSTART.md)

## 获取代码

```bash
git lfs install
git clone --depth 1 --recurse-submodules --shallow-submodules https://github.com/Winner12317/Humoind.git
cd Humoind
git lfs pull
git submodule foreach --recursive 'git lfs pull'
```

已有仓库可先运行 `git pull --ff-only`，再运行 `git submodule update --init --recursive` 和以上两条 `git lfs pull` 命令。

`engineai_rl_lab`、`unitree_mujoco` 和 `Human-Humanoid-Tools` 使用固定提交的子模块。带本地控制器和仿真改动的 [unitree_rl_lab](unitree_rl_lab/) 直接收录在此仓库。原始 G1 视觉策略位于 [policy/sensorwm](policy/sensorwm/)，部署导出在 [G1_Perceptive_WMAC/policies/sensor_transform](G1_Perceptive_WMAC/policies/sensor_transform/)。

`Human-Humanoid-Tools` 固定在上游提交 `13b6dd3024193aeb24eb4bf7ea5c73a975a38f5b`。安装重定向所需的预设时，从仓库根目录运行以下命令。依赖安装见 [hhtools 快速开始](https://github.com/Roboparty/Human-Humanoid-Tools/blob/13b6dd3024193aeb24eb4bf7ea5c73a975a38f5b/README_cn.md)。

```bash
cp -a pm01_retarget/hhtools_presets/pm01 Human-Humanoid-Tools/configs/robots/
(cd Human-Humanoid-Tools && uv run python scripts/install_builtin_robots.py --only g1_29dof)
```

公开版本保留源码、策略、模型、配置、实测日志和可分发的结果。项目的本地 Python 环境、构建产物和编辑器索引按需重建。相机标定的公开副本保留内参和时序，隐去设备序列号、固件更新标识和 USB 端口。

## 准备训练数据

重新训练或重导出需要本地获授权的参考动作。数据来源、目录、校验值和重定向步骤见[动作来源和本地恢复说明](SuperAMP/docs/pm01/MOTION_SOURCES.md)。PM01 已导出的 ONNX 和 MNN 可直接下载用于离线推理。仿真中的键盘行走和推扰测试可独立运行。

## 安装远端训练环境

Linux x86_64 上，从仓库根目录运行。

```bash
cd SuperAMP
bash scripts/install_env.sh
```

安装脚本会创建或复用 `superamp` 环境，安装锁定的 PyTorch CUDA 12.6 依赖并应用 mjlab 补丁。安装结束后执行脚本打印的激活命令。远端训练需要已安装兼容的 NVIDIA 驱动。使用 G1 任务时按上面的步骤恢复参考动作。更多安装选项见 [SuperAMP 说明](SuperAMP/README_zh.md#远端一键安装训练环境)。

本仓库记录加入 PM01 适配后的工作树。原 G1 复现手册对应加入 PM01 之前的独立快照。

## 训练数据署名

Training data includes Motion Data by [Bones Studio](https://bones.studio/). Use of the underlying dataset is subject to the [BONES-SEED Dataset License Agreement](https://bones.studio/info/seed-license). The get-up reference derives from [Ubisoft La Forge LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).
