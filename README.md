# Humoind

使用 SuperAMP 在 MuJoCo/mjlab 中训练 Unitree G1 和 EngineAI PM01。SuperAMP 支持 PPO + AMP 和 WMAC。PM01 使用由 G1 参考动作重定向并修正足部接触后的动作。

- [SuperAMP 训练框架](SuperAMP/README_zh.md)和 [PM01 训练配置](SuperAMP/docs/pm01/README.md)
- [PM01 重定向、接触修正和逐帧查看器](pm01_retarget/README.md)
- [平地 G1 的 WMAC＋AMP 复现手册](docs/平地%20G1%20的%20WMAC＋AMP%20复现手册.md)

## 获取代码

```bash
git clone --recurse-submodules https://github.com/Winner12317/Humoind.git
cd Humoind
```

`Human-Humanoid-Tools` 固定在上游提交 `13b6dd3024193aeb24eb4bf7ea5c73a975a38f5b`。安装重定向所需的预设时，从仓库根目录运行以下命令。依赖安装见 [hhtools 快速开始](https://github.com/Roboparty/Human-Humanoid-Tools/blob/13b6dd3024193aeb24eb4bf7ea5c73a975a38f5b/README_cn.md)。

```bash
cp -a pm01_retarget/hhtools_presets/pm01 Human-Humanoid-Tools/configs/robots/
(cd Human-Humanoid-Tools && uv run python scripts/install_builtin_robots.py --only g1_29dof)
```

## 准备训练数据

动作文件、训练日志、权重和虚拟环境保存在本地。训练前，将有权使用的 G1 原始 NPZ 和 CSV 分别恢复到 `SuperAMP/src/assets/motions/g1/amp/` 和 `SuperAMP/motion_data_csv/amp/`。本项目的 G1 参考文件对应 [AMP_mjlab 的基线提交](https://github.com/ccrpRepo/AMP_mjlab/tree/6c7a2947fccc973e4af8e6d90e550400f1b6fcfc)。

若已有本项目生成的 PM01 动作，可直接恢复 `SuperAMP/src/assets/motions/pm01/amp/`。也可按[重定向步骤](pm01_retarget/README.md#regenerating)重新生成 `pm01_retarget/pm01_npz_50hz/`，然后在 SuperAMP 环境中运行 `python scripts/pm01/convert_pm01_motions.py`。

## 安装远端训练环境

Linux x86_64 上，从仓库根目录运行。

```bash
cd SuperAMP
bash scripts/install_env.sh
```

安装脚本会创建或复用 `superamp` 环境，安装锁定的 PyTorch CUDA 12.6 依赖并应用 mjlab 补丁。安装结束后执行脚本打印的激活命令。远端训练需要已安装兼容的 NVIDIA 驱动，并按上面的步骤准备相应机器人的参考动作。更多安装选项见 [SuperAMP 说明](SuperAMP/README_zh.md#远端一键安装训练环境)。

本仓库记录加入 PM01 适配后的工作树。原 G1 复现手册对应加入 PM01 之前的独立快照。
