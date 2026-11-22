# 平地 G1 的 WMAC＋AMP 复现手册

# 平地 G1 的 WMAC＋AMP 复现手册



版本为 2026\-11\-20。正式启动命令对应当前 `local_4096_ppo_amp_budget` 训练，使用 `supermimic_official`、居中 AMP、v3 平滑成本、4096 环境，以及每轮两次、每批 25600 对的判别器训练。该配置的 checkpoint 协议为 v12。



任务是平地 G1 的速度指令行走和倒地起身。正式运行使用 `learned_v1` 想象足滑。当前源码还支持基于脚底胶囊几何的 InstantAMP 完整梯度模式，第 9 章给出独立启动命令，对应 checkpoint 协议 v13。恢复重置覆盖对照属于诊断。本文的正式训练从随机初始化开始。



**带走完整 Markdown 就可以复现。** 两个手册文件内容相同，正文包含启动命令，附录包含完整源码补丁。另备一份 `REPRODUCE_FROM_GITHUB.sh`，脚本内含相同载荷，可独立运行。



另一台电脑使用 Linux x86\_64、可用的 NVIDIA 驱动和访问公开 GitHub、软件包源的网络。驱动脚本下载公开 AMP\_mjlab 的固定提交，从文档提取整合补丁，随后在本机生成镜像数据和新权重。公司盘、CodeHub 和原电脑均无需访问。



公式全部采用普通文本代码块，图解采用文本流程图，表格采用简单Markdown，不依赖LaTeX、Mermaid、HTML图片或作者本地图片。推荐通过飞书“导入为在线文档”导入本 `.md`，导入/导出后附录内容仍必须完整，脚本会核对SHA256。[飞书官方导入说明](https://www.feishu.cn/hc/zh-CN/articles/360033241654-%E4%B8%8A%E4%BC%A0%E6%88%96%E5%AF%BC%E5%85%A5%E6%9C%AC%E5%9C%B0%E6%96%87%E4%BB%B6%E5%92%8C%E6%96%87%E4%BB%B6%E5%A4%B9)。



## 第1章 目标、边界与交付文件



|项目|本文固定目标|
|---|---|
|机器人/任务|G1、29动作，平面速度指令行走＋倒地恢复|
|Actor|4×96=384维时间优先历史，29维输出，MLP|
|WM|71维raw状态、35维context、29动作；135输入、70残差输出＋原2维接触代价分支|
|AMP|grounded\_v2：203维/帧、406维/pair；单个LSGAN D|
|风格奖励|真实策略分数 EMA 居中，固定尺度 1 的 tanh，范围 0–0\.1|
|判别器预算|每轮 2 批，每批各抽 25600 对真实策略帧与 25600 对专家帧|
|v3 平滑成本|二阶动作差、关节速度、解析 PD 请求力矩、指令功率|
|正式想象足滑|`learned_v1`，读世界模型接触代价头|
|可选想象足滑|`geometry_instantamp_full_support_v1`，仅平地恢复想象阶段启用|
|数据|原17条行走＋17条镜像；原1条恢复不改动，全部原NPZ为50Hz|
|默认规模|4096环境、H32、UTD0\.4、400000更新、TF32|
|结果目录|默认本地 `$HOME/amp_wmac_smooth_ws/results`|
|训练起点|四个网络随机初始化，无resume/init\-policy|



完整复现包括源码、数据生成、训练参数和启动流程。v3 系数来自短训和物理量级对照，长期起身表现由实际训练和回放评估决定。



## 第2章 来源、固定版本与方法范围



|来源|固定版本|采用范围|
|---|---|---|
|AMP\_mjlab|6c7a2947fccc973e4af8e6d90e550400f1b6fcfc|唯一公开补丁基线，原G1/动作数据/物理任务|
|官方SuperMimic|abc124c1bd5976341034668af3d12c087f2f1f55|模型训练/短想象BPTT/价值拟合思想及兼容网络、优化器、更新预算|
|InstantAMP|9fc8031055a18d58e84d9de90639660e7aa6b491|EMA 居中 tanh 奖励及可选的胶囊净空软接地足滑|
|HoST|70bb580949a336a920833700e4b5dc3bf7fe87ce|二阶动作差奖励思想，不采用动作幅度课程和控制器|
|HumanUP|7516e0f27e6f4d1e7365cf64ea577a78247bd8cb|速度/力矩/功率成本思想，明确适配为当前解析PD负荷代理|
|行走镜像来源|92ca7bb8a3d8ea959d934fc13b5aae7e30317757|已内嵌通用镜像实现，不需要获取私人项目|



官方 SuperMimic 用参考动作定义跟踪奖励。本项目使用平地任务奖励和单个 LSGAN 运动先验，沿用本地世界模型的状态表示、接触分支、监督和采样节奏。四项平滑成本和可选足滑几何分支由固定公式计算。



源码快照由公开基线＋本文补丁定义，不能用作者工作区某个Git HEAD替代。重建后Git HEAD仍是固定公开基线，补丁产生未提交差异是预期；文件哈希清单定义整合后的准确内容。许可证和来源随载荷保留。



## 第3章 新电脑从零一键复现



### 3\.0 唯一需要携带的文件与系统前提



将本文件完整保存为 `$HOME/Downloads/AMP_SUPERMIMIC_REPRODUCTION_FEISHU_zh.md`。允许文件名不同，但执行时的 `--document` 必须指向这份完整文件。必须保留附录B所有数据块；只有讲解正文、只有网页链接或删过附录的节选不能重建。



首次安装从公开 GitHub、Miniforge、conda\-forge、PyPI 和 PyTorch 下载依赖。请预先安装显卡驱动。脚本会在专用目录创建独立环境。Torch 及其 CUDA 依赖合计下载数 GB。



在 Ubuntu 的 Bash 终端安装系统依赖。



```Bash
set -eo pipefail
sudo apt-get update
sudo apt-get install -y python3 git curl ca-certificates build-essential libegl1 libgl1 libglfw3
nvidia-smi
mkdir -p "$HOME/Downloads"
```



Linux x86\_64、Python3\.11和CUDA12\.6轮子是本文安装范围；默认4096大网络已在48GB RTX6000 Ada验证。`nvidia-smi`成功只是驱动可见性检查，真正的GPU兼容要以脚本小规模物理烟测为准。小网络验证占用少，不代表正式训练只需同样显存。



### 3\.1 从本文生成驱动脚本



复制以下**整个代码块**到Bash。脚本本身较短，源码数据从同一份Markdown附录提取，不读取当前电脑工程。



```Bash
cat > "$HOME/reproduce_amp_wmac_smooth.sh" <<'AMP_REPRO_SCRIPT'
#!/usr/bin/env bash
# Reconstruct this exact flat AMP/WMAC snapshot from public GitHub and the
# payload embedded in the complete manual (or in the standalone script).
set -eo pipefail

amp_die() { printf '[AMP reproduce] ERROR: %s\n' "$*" >&2; exit 1; }
amp_note() { printf '[AMP reproduce] %s\n' "$*"; }

AMP_REPRO_DOCUMENT=""
AMP_REPRO_WORKSPACE="$HOME/amp_wmac_smooth_ws"
amp_repro_train=0
amp_repro_source_only=0
amp_repro_skip_smoke=0
while (( $# )); do
  case "$1" in
    --document)
      (( $# >= 2 )) || amp_die '--document needs a Markdown path'
      AMP_REPRO_DOCUMENT="$2"; shift 2 ;;
    --workspace)
      (( $# >= 2 )) || amp_die '--workspace needs a local directory'
      AMP_REPRO_WORKSPACE="$2"; shift 2 ;;
    --train) amp_repro_train=1; shift ;;
    --source-only) amp_repro_source_only=1; shift ;;
    --skip-gpu-smoke) amp_repro_skip_smoke=1; shift ;;
    --help|-h)
      printf '%s\n' \
        'Usage: bash reproduce_from_github.sh [--document complete_manual.md]' \
        '       [--workspace local_directory] [--train] [--skip-gpu-smoke]' \
        '       [--source-only]' \
        'Default: reconstruct current smooth-reward WMAC, install, CPU tests, mirror motions, GPU tiny smoke.' \
        '--train: additionally start the 4096-env full training run.' \
        '--source-only: only reconstruct/hash-check source; no install or training.' \
        'The standalone script embeds its own payload; a copied manual driver' \
        'needs --document pointing to the complete Markdown including Appendix B.'
      exit 0 ;;
    *) amp_die "Unknown argument: $1" ;;
  esac
done
(( !amp_repro_train || !amp_repro_source_only )) || amp_die '--train and --source-only cannot be combined'
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || amp_die 'Supported platform: Linux x86_64'
for amp_command in python3 git; do
  command -v "$amp_command" >/dev/null || amp_die "Missing $amp_command. Run: sudo apt-get install -y python3 git"
done

AMP_WORKSPACE="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$AMP_REPRO_WORKSPACE")"
[[ "$AMP_WORKSPACE" != / && "$AMP_WORKSPACE" != "$HOME" ]] || amp_die 'Use a dedicated workspace directory'
AMP_PROJECT_DIR="$AMP_WORKSPACE/AMP_mjlab_wmac_smooth_rewards"
AMP_LOG_ROOT="$AMP_WORKSPACE/results"
AMP_CONDA_ROOT="$AMP_WORKSPACE/tools/miniforge3"
AMP_CONDA_ENV="$AMP_CONDA_ROOT/envs/amp-wmac-smooth"
AMP_REPRO_SELF="$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "${BASH_SOURCE[0]}")"
if [[ -z "$AMP_REPRO_DOCUMENT" ]]; then
  AMP_REPRO_DOCUMENT="$AMP_REPRO_SELF"
fi
export AMP_WORKSPACE AMP_PROJECT_DIR AMP_LOG_ROOT AMP_CONDA_ROOT AMP_CONDA_ENV AMP_REPRO_DOCUMENT
mkdir -p "$AMP_WORKSPACE"

# Verify the complete payload BEFORE cloning or installing anything.
python3 - <<'PY'
from pathlib import Path
import base64
import hashlib
import json
import lzma
import os

document = Path(os.environ['AMP_REPRO_DOCUMENT']).expanduser()
assert document.is_file(), f'Missing complete Markdown: {document}'
reading = False
encoded = []
for line in document.read_text().splitlines():
    line = line.strip()
    if line.startswith('# '):
        line = line[2:].strip()
    if line == 'AMP_WMAC_B64_BEGIN':
        assert not reading, 'Missing end marker in Appendix B'
        reading = True
    elif line == 'AMP_WMAC_B64_END':
        assert reading, 'Missing begin marker in Appendix B'
        reading = False
    elif reading:
        encoded.append(''.join(line.split()))
assert not reading, 'Incomplete final Appendix B block'
assert encoded, ('No embedded source payload. Use a complete Markdown containing '
                 'all Appendix B data, or use the standalone script.')
blob = base64.b64decode(''.join(encoded), validate=True)
assert hashlib.sha256(blob).hexdigest() == 'c02add523776b414ae4a98ee5ca2b6bc139ab48f5721e2d12940a739369e01f3', 'Payload checksum mismatch'
payload = json.loads(lzma.decompress(blob))
expected = {'amp_wmac_smooth_flat.patch', 'source_manifest.json',
            'requirements-linux-py311-cu126.txt', 'bootstrap_env.sh',
            'verify_reproduction.py', 'default_config.json'}
assert payload['format'] == 'amp-wmac-flat-doc-payload-v3'
assert set(payload['files']) == expected
out = Path(os.environ['AMP_WORKSPACE'])/'.reproduction_payload'
assert not out.is_symlink(), f'Refuse payload directory symlink: {out}'
out.mkdir(exist_ok=True)
for name in sorted(expected):
    target = out/name
    data = payload['files'][name].encode()
    assert not target.is_symlink(), f'Refuse payload symlink: {target}'
    if target.exists():
        assert target.read_bytes() == data, f'Existing different payload file: {target}'
    else:
        target.write_bytes(data)
patch = out/'amp_wmac_smooth_flat.patch'
assert hashlib.sha256(patch.read_bytes()).hexdigest() == 'b914633f2270891813c54c7844d147c64a3b603b34c8a227853ee348380c366f'
print('[AMP reproduce] Complete source payload checksum OK')
PY

if [[ ! -e "$AMP_PROJECT_DIR" ]]; then
  amp_note 'Cloning the public AMP repository; no private remote is used'
  git init -q "$AMP_PROJECT_DIR"
  git -C "$AMP_PROJECT_DIR" remote add origin https://github.com/ccrpRepo/AMP_mjlab.git
  git -C "$AMP_PROJECT_DIR" fetch --depth 1 origin 6c7a2947fccc973e4af8e6d90e550400f1b6fcfc
  git -C "$AMP_PROJECT_DIR" checkout --detach FETCH_HEAD
else
  [[ -d "$AMP_PROJECT_DIR/.git" && ! -L "$AMP_PROJECT_DIR" ]] || amp_die "Existing path is not a regular Git checkout: $AMP_PROJECT_DIR"
fi
[[ "$(git -C "$AMP_PROJECT_DIR" rev-parse HEAD)" == 6c7a2947fccc973e4af8e6d90e550400f1b6fcfc ]] || amp_die 'Existing checkout HEAD is not the fixed public base; select a fresh --workspace'

python3 - <<'PY'
from pathlib import Path
import os
workspace = Path(os.environ['AMP_WORKSPACE'])
out = Path(os.environ['AMP_PROJECT_DIR'])/'docs/reproduction'
assert not out.parent.is_symlink(), f'Refuse docs directory symlink: {out.parent}'
assert not out.is_symlink(), f'Refuse helper directory symlink: {out}'
out.mkdir(parents=True, exist_ok=True)
expected = {'amp_wmac_smooth_flat.patch', 'source_manifest.json',
            'requirements-linux-py311-cu126.txt', 'bootstrap_env.sh',
            'verify_reproduction.py', 'default_config.json'}
for name in sorted(expected):
    source = workspace/'.reproduction_payload'/name
    target = out/source.name
    assert not target.is_symlink(), f'Refuse helper file symlink: {target}'
    if target.exists():
        assert target.read_bytes() == source.read_bytes(), f'Existing different helper: {target}'
    else:
        target.write_bytes(source.read_bytes())
PY
cd "$AMP_PROJECT_DIR"
amp_patch="$AMP_PROJECT_DIR/docs/reproduction/amp_wmac_smooth_flat.patch"
if git apply --check --whitespace=nowarn "$amp_patch" 2>/dev/null; then
  git diff --quiet && git diff --cached --quiet || amp_die 'Refuse to patch an already modified checkout; select a fresh --workspace'
  git apply --whitespace=nowarn "$amp_patch"
elif git apply --reverse --check --whitespace=nowarn "$amp_patch" 2>/dev/null; then
  amp_note 'Patch is already applied; checking every manifest file'
else
  amp_die 'Checkout conflicts with the fixed patch; no reset/clean performed. Select a fresh --workspace'
fi
python3 docs/reproduction/verify_reproduction.py
if (( amp_repro_source_only )); then
  amp_note "Source-only reconstruction complete: $AMP_PROJECT_DIR"
  amp_note 'No environment installation or training was performed'
  exit 0
fi

for amp_command in curl g++ make nvidia-smi; do
  command -v "$amp_command" >/dev/null || amp_die "Missing $amp_command. Install system packages and a working NVIDIA driver first; see manual section 3.0"
done
python3 - <<'PY'
import ctypes.util
missing = [name for name in ('EGL', 'GL', 'glfw') if not ctypes.util.find_library(name)]
assert not missing, f'Missing {missing}; run sudo apt-get install -y libegl1 libgl1 libglfw3'
PY
nvidia-smi

python3 - <<'PY'
from pathlib import Path
import os
workspace = Path(os.environ['AMP_WORKSPACE'])
for relative in ('downloads', 'cache', 'results', 'tools', 'tools/miniforge3',
                 'tools/miniforge3/envs', 'tools/miniforge3/envs/amp-wmac-smooth'):
    path = workspace/relative
    assert not path.is_symlink(), f'Refuse output/environment directory symlink: {path}'
PY
mkdir -p "$AMP_WORKSPACE/downloads" "$AMP_WORKSPACE/cache" "$AMP_LOG_ROOT"
export XDG_CACHE_HOME="$AMP_WORKSPACE/cache"
export PIP_CACHE_DIR="$AMP_WORKSPACE/cache/pip"
export CONDA_PKGS_DIRS="$AMP_WORKSPACE/cache/conda_pkgs"
export TORCH_HOME="$AMP_WORKSPACE/cache/torch"
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
export PIP_NO_INPUT=1
unset PIP_EXTRA_INDEX_URL PIP_CONSTRAINT PIP_BUILD_CONSTRAINT PIP_REQUIREMENT
amp_installer="$AMP_WORKSPACE/downloads/Miniforge3-25.9.1-0-Linux-x86_64.sh"
if [[ ! -x "$AMP_CONDA_ROOT/bin/conda" ]]; then
  [[ ! -e "$AMP_CONDA_ROOT" ]] || amp_die "Incomplete Conda directory exists: $AMP_CONDA_ROOT; use a fresh workspace"
  curl -fL --retry 3 -o "$amp_installer" https://github.com/conda-forge/miniforge/releases/download/25.9.1-0/Miniforge3-25.9.1-0-Linux-x86_64.sh
  python3 - "$amp_installer" <<'PY'
from pathlib import Path
import hashlib
import sys
assert hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() == '07f64c1d908ae036e9f6a81f97704899311c0ae677d83980d664b9781d4cc5fc', 'Miniforge installer checksum mismatch'
PY
  mkdir -p "$(dirname -- "$AMP_CONDA_ROOT")"
  bash "$amp_installer" -b -p "$AMP_CONDA_ROOT"
fi
if [[ ! -x "$AMP_CONDA_ENV/bin/python" ]]; then
  [[ ! -e "$AMP_CONDA_ENV" ]] || amp_die "Incomplete environment exists: $AMP_CONDA_ENV; use a fresh workspace"
  "$AMP_CONDA_ROOT/bin/conda" create --prefix "$AMP_CONDA_ENV" --override-channels \
    -c https://conda.anaconda.org/conda-forge python=3.11.17 pip -y
fi

# Write a portable activation entry point using proper shell quoting.
python3 - <<'PY'
from pathlib import Path
import os
import shlex
keys = ('AMP_WORKSPACE', 'AMP_PROJECT_DIR', 'AMP_LOG_ROOT', 'AMP_CONDA_ROOT',
        'AMP_CONDA_ENV', 'XDG_CACHE_HOME', 'PIP_CACHE_DIR', 'CONDA_PKGS_DIRS', 'TORCH_HOME')
lines = ['# Source this file from Bash; do not execute it as a subprocess.']
lines += [f'export {key}={shlex.quote(os.environ[key])}' for key in keys]
lines += [f'cd -- {shlex.quote(os.environ["AMP_PROJECT_DIR"])}',
          f'source {shlex.quote(str(Path(os.environ["AMP_PROJECT_DIR"])/"scripts/activate_env.sh"))}']
target = Path(os.environ['AMP_WORKSPACE'])/'activate_local.sh'
assert not target.is_symlink(), f'Refuse activation symlink: {target}'
target.write_text('\n'.join(lines)+'\n')
PY
source "$AMP_WORKSPACE/activate_local.sh"
export PYTHONPATH="$AMP_PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
bash docs/reproduction/bootstrap_env.sh
python -m pip check
CUDA_VISIBLE_DEVICES='' python docs/reproduction/verify_reproduction.py --environment
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -q
python scripts/prepare_wmac_mirrored_motions.py
python - <<'PY'
from pathlib import Path
import os
import tempfile
import torch
root = Path(os.environ['AMP_LOG_ROOT'])
with tempfile.TemporaryDirectory(prefix='local_checkpoint_check_', dir=root) as tmp:
    first, final = Path(tmp)/'first.pt', Path(tmp)/'final.pt'
    torch.save({'x': torch.arange(3)}, first)
    os.replace(first, final)
    assert torch.equal(torch.load(final, weights_only=True)['x'], torch.arange(3))
print('[AMP reproduce] Local checkpoint write/replace/load OK:', root.resolve())
PY
if (( !amp_repro_skip_smoke )); then
  bash scripts/train_smooth_weights_v3.sh --imagined-recovery-foot-slip learned_v1 \
    --run-name repro_smoke --log-root "$AMP_LOG_ROOT" \
    --num-envs 8 --iterations 3 --no-compile \
    --model-hidden 64 64 --policy-hidden 64 32 --critic-hidden 32 16 \
    --discriminator-hidden 32 16 --wm-updates 1 --wm-batch-size 8 \
    --actor-batch-size 8 --imagination-horizon 4 \
    --critic-epochs 1 --critic-minibatches 2 \
    --discriminator-updates 1 --discriminator-batch-size 16 \
    --save-interval 3 --log-interval 1 --evaluation-interval 0
fi
amp_note 'Reconstruction and requested checks completed. No company storage was accessed.'
printf 'New terminal activation: source %q\n' "$AMP_WORKSPACE/activate_local.sh"
amp_full_args=(--num-envs 4096 --imagined-recovery-foot-slip learned_v1
               --discriminator-updates 2 --discriminator-batch-size 25600
               --log-root "$AMP_LOG_ROOT" --run-name local_4096_ppo_amp_budget)
printf 'Full training: bash scripts/train_smooth_weights_v3.sh'
printf ' %q' "${amp_full_args[@]}"
printf '\n'
if (( amp_repro_train )); then
  bash scripts/train_smooth_weights_v3.sh "${amp_full_args[@]}"
fi
AMP_REPRO_SCRIPT
bash "$HOME/reproduce_amp_wmac_smooth.sh" \
  --document "$HOME/Downloads/AMP_SUPERMIMIC_REPRODUCTION_FEISHU_zh.md"
```



默认依次执行：载荷校验→公开GitHub固定提交clone→应用补丁→源码与资产哈希校验→创建独立Conda环境→安装锁定依赖→安装history补丁→CPU测试→本地生成镜像→小规模GPU烟测。**默认不会自动启动40万轮正式训练。**



第一次完整安装成功之后，在新终端执行：



```Bash
source "$HOME/amp_wmac_smooth_ws/activate_local.sh"
```



Bash子进程中的activate不会自动改变外层终端，所以正式训练前仍需要上面的source。



### 3\.2 脚本选项和确定的输出位置



|参数|效果|
|---|---|
|`--document 完整md路径`|从这份文档读取附录；独立自包含脚本可省略|
|`--workspace 专用本地目录`|整体改变项目/环境/结果位置，默认下面所列目录|
|`--source-only`|仅公开下载、应用补丁和源码校验，不安装环境/不训练|
|`--skip-gpu-smoke`|仍安装环境、跑CPU和准备数据，只跳过GPU短测；不能据此声称GPU已验证|
|`--train`|全部准备完成后启动正式4096环境训练，不是更换默认算法|



```Plain Text
workspace = $HOME/amp_wmac_smooth_ws
project   = $HOME/amp_wmac_smooth_ws/AMP_mjlab_wmac_smooth_rewards
results   = $HOME/amp_wmac_smooth_ws/results
Conda     = $HOME/amp_wmac_smooth_ws/tools/miniforge3
env       = $HOME/amp_wmac_smooth_ws/tools/miniforge3/envs/amp-wmac-smooth
activate  = $HOME/amp_wmac_smooth_ws/activate_local.sh
```



不用挂载任何共享盘。新clone不会携带作者机器的logs符号链接，脚本还会拒绝输出目录符号链接。所有权重、日志、缓存默认写到专用workspace。



只想先验证文档是否完整，可执行：



```Bash
bash "$HOME/reproduce_amp_wmac_smooth.sh" \
  --document "$HOME/Downloads/AMP_SUPERMIMIC_REPRODUCTION_FEISHU_zh.md" \
  --workspace "$HOME/amp_wmac_source_check" --source-only
```



这只重建源码；之后要完整安装可重跑同一workspace并去掉 `--source-only`，已经匹配的源码会被核验。不同载荷/代码修改不匹配时脚本明确停止，不覆盖你的改动；请选择另一个新workspace。



### 3\.3 若选择独立自包含脚本



如果你携带的是 `REPRODUCE_FROM_GITHUB.sh`，它自身已包含相同源码数据，可直接执行：



```Bash
bash "$HOME/Downloads/REPRODUCE_FROM_GITHUB.sh" --workspace "$HOME/amp_wmac_smooth_ws"
```



这是另一种等价载体，不要求同时拿到作者电脑的patch、requirements、assets、权重或任何文件夹。本文路线已经内嵌全部必要迁移文本，勿将第3\.1节驱动与其不含附录的节选单独使用。



## 第4章 安装的内容、锁定范围与复查



脚本固定Miniforge `25.9.1-0`，安装器SHA256为 `07f64c1d908ae036e9f6a81f97704899311c0ae677d83980d664b9781d4cc5fc`；创建Python `3.11.17`环境。核心依赖为Torch `2.7.0+cu126`、torchvision `0.22.0+cu126`、mjlab `1.2.0`、mujoco\-warp `3.8.1`、warp\-lang `1.12.0`、MuJoCo `3.12.0`、SciPy `1.17.0`、rsl\-rl\-lib发行包 `5.0.1`。完整版本闭包由载荷中的 `requirements-linux-py311-cu126.txt` 定义，不用执行未固定版本的pip升级命令替换它。



环境校验包含两项与观测和 AMP 直接相关的依赖。



1. 本地 `rsl_rl` 是带AMP的源码fork，必须在PYTHONPATH最前；发行包5\.0\.1用于满足mjlab依赖元数据。不要再用另一份 `pip install -e rsl_rl` 把发行版元数据替换成不兼容版本。激活脚本自动选择项目根目录。

2. 必须将仓库 `mjlab_patch/mjlab/managers/observation_manager.py` 安装到该独立环境。`history_ordering="time"` 是Actor/Critic配置，不是仅把一个384维数组reshape就能替代的功能。校验要求安装文件与补丁逐字节一致。

    

```Bash
source "$HOME/amp_wmac_smooth_ws/activate_local.sh"
python -m pip check
python docs/reproduction/verify_reproduction.py --environment
```



检查覆盖公开基线 HEAD、298 个源码/测试/许可证/资产文件、18 个原动作文件、依赖版本、实际模块导入位置、history 补丁及附录 A 正式配置。安装器、源码与载荷均有 SHA256 校验。不同 GPU 架构、驱动和底层库可能导致训练结果有所差异。



本文从头路线不需要迁移旧权重，不需要作者安装好的yao\-rl，不执行公司盘挂载。若安装失败，保留脚本错误与日志，不跳过校验后宣称复现完成；可在同一未损坏workspace重试，或选择新的专用目录，不删除自己的项目目录。



## 第5章 正式配置、训练预算与优化器



### 5\.1 正式训练模式与关键参数



```Plain Text
--profile supermimic_official
--amp-reward-mapping instantamp_centered_tanh_v1
--smooth-reward-mode host_humanup_v1
--imagined-recovery-foot-slip learned_v1
--discriminator-updates 2
--discriminator-batch-size 25600
```



裸 profile 的兼容默认是 `clipped_lsgan_v1`、`smooth_reward_mode=none`、`learned_v1`，判别器批量为 512。第 9 章的 v3 启动器选择居中 AMP 和平滑成本，正式命令另显式指定足滑和判别器预算。







|部分|当前supermimic\_official值|说明|
|---|---|---|
|num\_envs|4096|并行真实环境|
|iterations|400000|本框架训练更新次数|
|collect\_steps / utd|0 / 0\.4|启用UTD累加器，通常2、3步交替|
|warmup\_steps|8|起始采集真实窗口|
|model/critic\_warmup\_iterations|0 / 0|无额外模型/价值预热轮|
|replay\_steps|1000|向量控制步容量，约409\.6万条环境转移槽位|
|validation\_env\_fraction|0\.1|独立环境子集作held\-out验证，不是每次随机扣10%batch|
|wm\_window|7|对应官方8个存储状态帧中的7个计分转移|
|wm\_updates / wm\_batch\_size|10 / 2048|每更新十次模型拟合|
|actor\_batch\_size / actor\_updates|4096 / 1|从Replay抽取起点|
|imagination\_horizon|32|每起点展开32控制步，即0\.64秒|
|critic\_epochs / critic\_minibatches|5 / 4|复用一次randperm分组|
|discriminator\_updates / batch|2 / 25600|第 9 章正式命令显式指定，裸 profile 的批量为 512|
|model\_hidden|2048,2048,2048|主模型GELU＋LayerNorm，普通Linear输出初始化|
|policy\_hidden|2048,1024,512,256|GELU＋LayerNorm，最后29维线性输出，无内部tanh|
|critic\_hidden|256,128|GELU＋LayerNorm，最后1维输出|
|discriminator\_hidden|1024,512,256|AMP的ReLU trunk，最后1维线性分数，无sigmoid|
|接触头|256,128|保留SiLU，2维softplus非负代价|
|Actor/WM/Critic LR|2e\-4|新官方模式常数学习率|
|D LR|1e\-3|250000更新余弦衰减至3%＝3e\-5|
|gamma / gae\_lambda|0\.99 / 0\.95|Actor\-Critic配置|
|exploration\_noise / action\_limit|0\.1 / 20|真实采样高斯噪声；外层动作裁剪|
|precision / compile|tf32 / true|浮点参数，支持的矩阵运算允许TF32；首轮含编译|
|save\_interval / log\_interval|400 / 4|约每1000/10个采样tick|
|evaluation\_interval|2000|32环境×250步独立物理评估|
|checkpoint\_replay|false|默认不保存庞大Replay|



完整全部字段及未启用的兼容字段见附录A与`default_config.json`。不要依赖`WmacConfig()`默认构造：必须选择`for_profile('supermimic_official')`或CLI的同名profile；不传profile的训练入口默认是balanced。





|新模式参数|本文目标值|
|---|---|
|amp\_score\_ema\_decay / amp\_score\_scale|0\.99 / 1\.0|
|smooth\_action\_second\_coef|0\.0175|
|smooth\_joint\_velocity\_coef|0\.000625|
|smooth\_torque\_coef|0\.000625，请求力矩L2成本|
|smooth\_power\_coef|0\.00003125，指令功率L2成本|



### 5\.2 为什么UTD是0\.4，不把真实rollout改成PPO的24步



官方`replay_ratio=16`按模型batch、训练次数和存储状态帧核算学习频率。固定默认值下：



```Plain Text
官方每个真实向量采样tick对应的训练更新数
= num_envs * replay_ratio / (wm_num_batches * wm_batch_size * stored_frames)
= 4096 * 16 / (10 * 2048 * 8)
= 0.4

本框架每次更新平均采样步数 = 1 / UTD = 2.5
```



当前累加器每轮加2\.5，取整数并保留余数，产生2、3、2、3……；没有每轮强行采24步。官方常每个tick插入一组状态，本Replay存`state,next_state`；K=7是对应8帧的适配，不是少训练了一步。



400000更新约对应1000000采样tick，加初始8tick及续训重填Replay等额外数据。每tick含4096个环境转移，所以只比较“训练了多少轮”没有共同物理预算。PPO应保留自身rollout长度，再比较相同真实测试效果下的总wall time、累计`sim_steps`与实际环境转移数。



### 5\.3 预热、编译与首轮速度



初始8个真实控制步用于让Replay能抽连续窗口，不是额外500轮模型预训练。新版`model_warmup_iterations=0`、`critic_warmup_iterations=0`。首次WM、想象步、FK/奖励或Critic调用可能触发torch\.compile，耗时会明显高于稳定更新。续训换形状、batch或编译缓存也会重新编译。



时间日志分为collect、model、disc、actor\_critic、save、evaluation；默认GPU异步执行时阶段时间不等于精确CUDA核耗时，`--benchmark-sync`才插入阶段同步。正式性能比较应把环境构建、初次编译、保存、评估按统一规则计入，而不把小网络短测的0\.1秒当默认大网络成绩。



### 5\.4 Muon＋Adam具体怎样分组



本实现用PyTorch2\.10的Muon数学回移到当前Torch2\.7，避免为单个优化器升级整个仿真环境。源码和许可证见`src/wmac/optimizers.py`、`THIRD_PARTY_NOTICES.md`及`licenses/`；官方参考为[learner\_utils\.py固定版本](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/learner_utils.py)。



分组规则是遍历每个nn\.Sequential，只取其中Linear层列表的`[1:-1]`权重交给Muon。第一层、最后层、所有bias、LayerNorm等参数交给Adam。不是“所有二维参数都是Muon”。共享参数按id去重，检查点保存参数名和算法标记，防止把Adam矩误载入Muon。



```Plain Text
Muon：momentum=0.95，Nesterov，5次Newton-Schulz，eps=1e-7
NS系数=(3.4445,-4.7750,2.0315)
Muon权重衰减=0.1；学习率匹配match_rms_adamw
Adam：eps=1e-8，Actor betas=(0.9,0.95)，WM/Critic=(0.9,0.999)，weight_decay=0
梯度：先全局norm裁剪1，再对Actor元素做±0.1裁剪
```



Muon先形成动量混合梯度，将矩阵转成BF16，在Frobenius范数归一化后作多项式迭代，得到近似正交化更新。对梯度矩阵X的一次迭代如下；这不是严格SVD，不能宣称5步后每个奇异值精确为1。



```Plain Text
A = X * X^T
X <- a*X + (b*A+c*A^2)*X
调整步长 = lr * 0.2 * sqrt(max(weight.shape))
W_next = (1-lr*0.1)*W - 调整步长*X_final
```



D仍用独立Adam：trunk weight\_decay=1e\-3，输出head=0\.1，betas默认0\.9/0\.999。它没有改成Muon，也没有跟随Actor的常数LR。



### 5\.5 新官方模式与旧profile的保护逻辑



旧profile保留验证误差门控、动作偏移回滚等既有行为。新官方模式不因有限但较大的held\-out误差或action\-shift阈值拒绝更新；仍跳过非有限loss/梯度或失败的优化器步骤，并保留日志；新增居中AMP还要求真实策略分数EMA已初始化。配置中出现`max_action_shift=0.25`、`max_validation_model_error=20`等兼容字段，不表示新版仍执行这些回滚门控。默认`actor_trust_coef=0`、`real_critic_coef=0`，没有额外真实PPO surrogate或真实TD辅助项。





### 5\.6 当前实现的等价提速与计算预算



在 CUDA 的官方模式下，采集和模型拟合的日志标量先在 GPU 聚合，按阶段集中回读。Critic 在第一个 epoch 按原索引取出小批次，后续 epoch 复用，同样执行 5×4 次优化。Replay 采样在确认数据全部有效时跳过冗余替换，普通回退路径继续处理无效转移与边界。动作有限值保护、梯度检查与优化器拒绝机制继续参与执行。



判别器的 25600 对预算会增加判别器计算量。比较不同批量的速度时，在 `metrics.csv` 分开读 `disc_seconds`、`collect_seconds`、`model_seconds`、`actor_critic_seconds`，并记录物理交互数 `sim_steps` 和成功率。



## 第6章 状态、观测、动作与AMP输入协议



本节针对 `supermimic_official` 使用的 `Unitree-G1-AMP-Flat-Aligned`。它沿用本仓库的平地 G1、速度指令、AMP 风格约束与倒地恢复规则，不引入相机、复杂地形或 Hiking 动作库。复现时应先调用 G1 配置工厂，再应用训练入口中的覆盖；只抄公共配置中的初值，会得到错误的动作缩放、身体集合和恢复比例。以本文固定快照的函数名和源码校验清单为准。



### 6\.1 一个控制步如何流动



物理步长是 `0.005` 秒，`decimation=4`，一次策略控制持续 `0.02` 秒，即 50 Hz。Actor 每次产生 29 维动作，物理仿真执行四个子步，随后计算终止、奖励和新观测。`CaptureEnv` 在自动重置前保存实际后继，所以终止后继与新回合初态是两件事。



主要入口是 `src/tasks/amp_loco/amp_env_cfg.py`、`src/tasks/amp_loco/config/g1/env_cfgs.py` 与 `src/wmac/env.py`。平地配置把地形设为 plane、移除 height scan 和 terrain curriculum；当前章节中的高度均以世界地面 z=0 为基准，不使用感知项目的逐点地形高度查询。



### 6\.2 Actor 输入：384 = 4 × 96



单帧观察的顺序必须与下表完全一致。区间采用 Python 左闭右开记法，不能按“所有时刻的角速度放一起”重排。



|单帧区间|维数|内容与坐标系|训练时附加均匀噪声|
|---|---|---|---|
|`[0:3]`|3|骨盆 IMU 坐标系角速度，rad/s|每维 `[-0.2,0.2]`|
|`[3:6]`|3|世界向下单位向量在 root 坐标系中的投影|每维 `[-0.05,0.05]`|
|`[6:9]`|3|速度指令 `(vx,vy,wz)`|无|
|`[9:38]`|29|当前关节位置减默认位置，rad|每维 `[-0.01,0.01]`|
|`[38:67]`|29|当前关节速度减默认速度，rad/s|每维 `[-0.5,0.5]`|
|`[67:96]`|29|上一步执行的原始归一化动作|无|



物理配置见 `amp_env_cfg.py`；模型端相同布局见 `src/wmac/state.py`。这里的关节位置项调用 `joint_pos_rel`，没有传 `biased=True`，所以观测本身不叠加 encoder bias。噪声加在相对位置、相对速度等观察量上，并没有改变真实物理状态。指令和上一动作也没有被统一加噪声。



历史顺序为 `[o(t-3), o(t-2), o(t-1), o(t)]`，从最旧到最新，形成 `[B,4,96]` 后按时间展开成 `[B,384]`。重置后的首次有效帧填满该环境的四帧历史；这不同于三帧零加一帧初态。`history_ordering="time"` 位于 `amp_env_cfg.py`，配套补丁位于 `mjlab_patch/mjlab/managers/observation_manager.py`。补丁的 `compute(update_history=False)` 可返回缓存，重复查询不会再推进一次历史。



本文MLP收到的384维还会经过自己的RunningNorm。载荷保留Hiking MoE兼容代码，但本文训练命令选择MLP，不能拿关闭归一化的MoE协议替代本文配置。维数、排列和是否归一化必须与checkpoint一起恢复。



想象轨迹从 Replay 中对应的四帧真实历史开始，每预测一步，就用 `actor_frame()` 构造下一帧，丢掉最旧帧并追加新帧。`runner.py` 为相同槽位产生均匀噪声；当前 rollout 的指令来自起点 batch，在32步想象内保持固定。Actor 不直接看到35维上下文，也不直接看到世界位置、根部线速度或参考动作相位。



### 6\.3 世界模型的71维状态、35维上下文和 Critic 的136维输入



`StateAdapter.state()` 从 free joint 的 qpos/qvel 直接取状态，避免把可能滞后一物理子步的派生缓冲当作真实根状态。MuJoCo free joint 的角速度需旋转到世界系后，才放进以下状态。四元数始终是 wxyz。



|state 区间|含义|
|---|---|
|`[0:3]`|root 世界位置 xyz，m|
|`[3:7]`|root 世界姿态四元数 wxyz|
|`[7:10]`|root 世界线速度 xyz，m/s|
|`[10:13]`|root 世界角速度 xyz，rad/s|
|`[13:42]`|29维真实关节位置，rad|
|`[42:71]`|29维真实关节速度，rad/s|



state 中的关节角是绝对关节坐标，Actor 中的关节角才减默认姿态。它们不能互换。`state.py` 给出实际拼接，`analyze_state()` 利用已编译模型的父子关系、关节轴和参考位置执行 Torch FK，输出各身体世界位置、姿态、线速度与角速度。



|context 区间|含义|
|---|---|
|`[0]`|当前脚部几何摩擦系数的平均值|
|`[1:30]`|29维 encoder bias，rad|
|`[30:33]`|torso 的局部质心相对默认值的 xyz 偏移，m|
|`[33]`|此环境是否属于永久分配的延迟恢复子集，0或1|
|`[34]`|当前连续异常计数，整数值用浮点 Tensor 保存|



上下文见 `StateAdapter.context()`。摩擦启动随机化范围为0\.3至1\.2；bias为每关节±0\.015 rad；torso质心偏移 x/y 为±0\.025 m、z为±0\.03 m。这些是物理参数，不是每帧重新采样的观察噪声。想象中前34维随轨迹保留，只有恢复计数被更新。



Critic 使用当前物理状态和特权上下文，输入不是384维 Actor 历史，也不是原 PPO 里任意拼接的观察。`state.py` 的准确布局为：



|Critic 区间|内容|
|---|---|
|`[0:1]`|root高度|
|`[1:5]`|去掉世界 yaw 后的root四元数|
|`[5:8]`|yaw坐标系中的root线速度|
|`[8:11]`|yaw坐标系中的root角速度|
|`[11:40]`、`[40:69]`|关节位置、关节速度|
|`[69:98]`|上一原始动作|
|`[98:101]`|指令vx、vy、wz|
|`[101:136]`|上表35维context|



世界模型并非把raw state原样送入MLP：`invariant_features()`编码为height1＋去yaw姿态的旋转矩阵前两列6＋body线速度3＋body角速度3＋q29＋dq29，仍为71维；加动作29和context35得到135。raw状态存四元数4维，网络输入姿态用6维，网络输出姿态增量又用rotvec3维，三者不能混称同一排列。



heading 只去掉偏航，并不把倾倒姿态修成直立。Critic 对不同行进朝向具有统一解释，同时仍能看到倾斜、高度、摩擦与恢复计数。



### 6\.4 29维动作到真实驱动



动作遵循 G1 原生非 free\-joint 顺序，完整顺序如下。部署时应以导出元数据的 `joint_names` 再核对，不能按 Python actuator 配置分组的出现顺序自行拼向量。



```Plain Text
0 left_hip_pitch_joint       1 left_hip_roll_joint       2 left_hip_yaw_joint
3 left_knee_joint            4 left_ankle_pitch_joint    5 left_ankle_roll_joint
6 right_hip_pitch_joint      7 right_hip_roll_joint      8 right_hip_yaw_joint
9 right_knee_joint          10 right_ankle_pitch_joint  11 right_ankle_roll_joint
12 waist_yaw_joint          13 waist_roll_joint        14 waist_pitch_joint
15 left_shoulder_pitch_joint 16 left_shoulder_roll_joint 17 left_shoulder_yaw_joint
18 left_elbow_joint         19 left_wrist_roll_joint   20 left_wrist_pitch_joint
21 left_wrist_yaw_joint     22 right_shoulder_pitch_joint 23 right_shoulder_roll_joint
24 right_shoulder_yaw_joint 25 right_elbow_joint       26 right_wrist_roll_joint
27 right_wrist_pitch_joint 28 right_wrist_yaw_joint
```



官方对齐的 MLP 输出无内部 tanh，物理采样执行 `a=clip(actor(o)+0.1*Normal(0,1),-20,20)`；想象动作也在相同外层范围裁剪。这里的20是归一化动作范围，不是20 rad关节限位。



`JointPositionAction` 先形成 `q_des=q_default+scale*a`，实际下发的目标再减去 encoder bias。默认速度目标为0；驱动按PD误差计算力矩，再经过力矩\-速度限制与摩擦模型。不能把网络输出直接当力矩，也不能只复制XML中原来的位置驱动参数；G1 Python 配置会建立对应执行器。



当前使用 `KNEES_BENT_KEYFRAME`：root初高0\.78 m，hip\_pitch=\-0\.312、knee=0\.669、ankle\_pitch=\-0\.363，左右 shoulder\_pitch=0\.2，left/right shoulder\_roll=\+0\.2/\-0\.2，左右 elbow=0\.6，其余关节默认0，默认关节速度全0。配置中的 HOME\_KEYFRAME 初高0\.8 m不是这条训练路径的默认姿态。动作偏置使用默认姿态，回合初始的实际状态则从动作库随机帧恢复，二者职责不同。



|关节组|stiffness|damping|nominal effort limit|动作scale|
|---|---|---|---|---|
|hip\_pitch、hip\_roll、knee|99\.098428|6\.308802|139|0\.350661|
|hip\_yaw、waist\_yaw|40\.179239|2\.557890|88|0\.547546|
|shoulder三轴、elbow、wrist\_roll|14\.250623|0\.907223|25|0\.438577|
|wrist\_pitch、wrist\_yaw|8\.611032|0\.548195|10|0\.290325|
|waist\_pitch、waist\_roll|28\.501246|1\.814446|50|0\.438577|
|ankle\_pitch、ankle\_roll|28\.501246|1\.814446|50|0\.438577|



scale的实际公式是 `0.25*effort_limit/stiffness`，不是所有关节统一0\.25。见 `src/assets/robots/unitree_g1/g1_constants.py`，自定义驱动执行见 `unitree_actuators.py`。其中 effort limit 还会与具体电机转速曲线共同限幅，表中 nominal 值不等于任何转速都可输出的力矩。



运行后最可靠的提取办法，是从 `env.scene['robot'].joint_names` 获取自然顺序，再按每个 actuator 对应的 target joint，将 `env.sim.mj_model.actuator_gainprm[:,0]` 和 `-actuator_biasprm[:,2]` 重排成相同顺序。默认位置取 `robot.data.default_joint_pos[0]`，scale取 `env.action_manager.get_term('joint_pos').scale`。现有 `mjlab.rl.exporter_utils.get_base_metadata()` 已实现这套映射，导出的 ONNX 包含这些字段。



### 6\.5 203维 AMP：保留局部运动，还补充地面姿态线索



AMP单帧的13个身体严格依次为：pelvis；left\_hip\_roll\_link、left\_knee\_link、left\_ankle\_roll\_link；right\_hip\_roll\_link、right\_knee\_link、right\_ankle\_roll\_link；left\_shoulder\_roll\_link、left\_elbow\_link、left\_wrist\_yaw\_link；right\_shoulder\_roll\_link、right\_elbow\_link、right\_wrist\_yaw\_link。锚点为 torso\_link，不是pelvis。



`src/amp_features.py` 的特征按类型成块拼接：



|AMP区间|内容|
|---|---|
|`[0:39]`|13个身体相对torso位置，经torso完整姿态逆旋转|
|`[39:117]`|torso相对身体姿态矩阵的前两列，每身体6维|
|`[117:156]`|各身体世界线速度，分别转进该身体自己的局部系|
|`[156:195]`|各身体世界角速度，分别转进该身体自己的局部系|
|`[195:198]`|root/pelvis局部系的投影重力|
|`[198:199]`|pelvis世界z减地面高度0|
|`[199:201]`|左、右脚sole参考点的世界高度|
|`[201:203]`|左、右脚sole参考点的世界竖直速度|



前两列按现有Tensor布局展开为 `R00,R01,R10,R11,R20,R21`。身体集合的顺序、每身体的六个矩阵数与四大特征块的顺序都属于检查点协议；即使总维数同为203，错序后也不能复用判别器。只比较网络输入宽度不足以发现这类错误，复现时应同时保存body\_names和schema。速度不是“减去torso速度的相对速度”，而是世界速度用每个身体自己的坐标系表达。复现时把所有速度都转到torso坐标系，会改变判别器输入定义。



sole参考点在 ankle\_roll\_link 局部系的偏移为 `(0.04,0,-0.035)` m。其位置是 `p_ankle+R_ankle*offset`，速度是 `v_ankle+omega_ankle×(R_ankle*offset)`；脚高和竖直速度不能直接取 ankle 原点。它们属于运动学线索，不是真实接触力、接触布尔标签或整只脚的最小间隙。



原来195维会消去整体平移与部分姿态差异，新增8维使判别器能看到离地高度和倾倒状态。专家、真实策略与想象轨迹都调用同一特征构造器。判别器输入是相邻两帧拼接的406维；先分别归一化单帧再拼接，保持梯度可穿过输入。真实AMP观察不加随机噪声。





### 6\.6 本地生成左右镜像动作



`scripts/prepare_wmac_mirrored_motions.py` 读取公开基线的 `WalkandRun` 和 `Recovery`。每条行走片段以首帧骨盆的世界坐标 `y=y0` 为镜像平面，交换左右身体与关节。令 `M=diag(1,-1,1)`，位置的 y 分量变为 `2*y0-y`，线速度变为 `M*v`，角速度变为 `-M*omega`，姿态旋转变为 `M*R*M`。关节位置围绕各自参考角反射，关节符号由 G1 MJCF 的转轴计算，关节速度沿用同一置换和符号。镜像保留 50 Hz、原帧序与原时长。



脚本生成 17 条原走路、17 条反向配对和 1 条逐字节复制的恢复片段，保存到 `.wmac_cache/motions/g1_mirrored`。校验清单包含原文件、镜像文件、机器人映射及哈希。环境重置和 AMP 专家池通过训练配置读取同一份生成数据。实现见 `src/wmac/mirroring.py::mirror_motion` 和 `src/wmac/motions.py::prepare_mirrored_dataset`。



## 第7章 AMP、SuperMimic思想与梯度推导



本节以 `_smooth_tuned` 当前平地训练为准。主运行使用 `supermimic_official + instantamp_centered_tanh_v1 + host_humanup_v1`，Actor 为 MLP，采用 v3 平滑系数、两次各 25,600 对的判别器批次及 `learned_v1` 想象足滑代价。InstantAMP 几何足滑是可独立启用的选项。公式使用普通文本代码块，可随 Markdown 导入飞书。



AMP 用参考动作片段定义运动风格，SuperMimic 用真实模拟器数据拟合共享动力学，再用短时想象回报的梯度训练策略。当前 WMAC 将两者结合。官方 SuperMimic 的固定参考源码同时包含纯 BPTT 和带价值引导的 Actor\-Critic。我们对齐的是后者兼容的训练节奏、优化器和价值估计细节，仍使用本项目的平地速度指令、AMP 分布奖励和局部残差动力学。 [SuperMimic 论文 v2](https://arxiv.org/abs/2610.28480v2) · [官方策略学习器](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/policy_learner.py#L83-L199)



### 7\.1 四个网络分别解决什么



Actor回答“当前该输出什么动作”；世界模型预测“执行动作后状态怎样变化”；判别器评价“相邻状态的运动模式与参考分布是否相似”；Critic估计“短时想象结束之后还可能得到多少回报”。这四个职责不能混在一个损失里同时更新。



```Plain Text
真实交互：Actor → 动作 → MuJoCo/Warp → Replay（保存数据，不保留物理计算图）
                              │
                  ┌───────────┴────────────┐
                  ▼                        ▼
           用真实连续窗口拟合 WM      用专家/真实策略片段拟合 D

策略步骤：真实起点 → Actor → 冻结参数的 WM → 预测状态 → FK/任务/AMP奖励
                        ▲                         │
                        └──── 更新4帧观测历史 ────┘
                                     │
                    H32回报＋冻结参数Critic的末端价值
                                     │
                                  只更新Actor
                                     │
                    detach特征、回报和旧价值，拟合Critic
```



以下记Actor参数为theta，WM为phi，D为omega，Critic为nu；`stopgrad(x)`表示数值保留、导数截断。真实采样不需要可微物理引擎，Actor的梯度沿学习到的WM传播，因此“链路可微”不等于“导数已被证明等于真实物理导数”。



### 7\.2 AMP学习的是运动分布，不是逐帧参考跟踪



AMP 将参考片段转成相邻帧对，用两类转移的分布差异学习风格。当前动作 NPZ 为 50 Hz，专家取 `i,i+1`，真实转移也是 20 ms 控制步。增加 D 的抽样预算改变每单位真实交互对专家数据的使用次数，帧对跨度始终为 20 ms。



原 legacy 每帧为 13 个身体各 15 维，共 195 维。每个身体含相对 anchor 的位置 3 维、相对旋转矩阵前两列 6 维、body 局部线速度 3 维、局部角速度 3 维。当前 `grounded_v2` 保留这 195 维，再添加骨盆局部重力 3 维、骨盆高度 1 维、左右脚底参考点高度 2 维和竖直速度 2 维，每帧得到 203 维，相邻对共 406 维。这些补充量提供朝向和离地线索。专家、真实策略和预测状态使用同一特征约定，四元数为 wxyz。



令Phi为运动特征、N为当前AMP归一化，判别器输入和损失为：



```Plain Text
z_E = concat(N(Phi(s_E)), N(Phi(s_E_next)))
z_P = concat(N(Phi(s_P)), N(Phi(s_P_next)))

L_D = 0.5 * { E_E[(D_omega(z_E)−1)^2] + E_P[(D_omega(z_P)+1)^2] }
      + 10 * E_E[ ||grad_z D_omega(z_E)||_2^2 ]
```



D输出实数，不先经过sigmoid；±1是LSGAN回归标签，不是概率。忽略梯度惩罚、权重衰减和有限容量，在某个输入z处对d求最优，可得：



```Plain Text
0 = p_E(z)*(d−1) + p_P(z)*(d+1)
D_star(z) = [p_E(z)−p_P(z)] / [p_E(z)+p_P(z)]
```



分布相同时理想分数为 0。这里的 p\_E、p\_P 由实际抽样形成，恢复比例、片段权重和 Replay 数据都会影响它们。实际 D 还受正则和容量约束，分数用来提供相对风格信号。



GP是**归一化专家输入上的零中心梯度惩罚**：没有`(||grad||−1)^2`，没有专家/策略之间的随机插值，不能与WGAN\-GP混称。当前代码先在`no_grad`内构造真实/专家特征，再仅对专家输入重新开启梯度：



```Python
expert = expert.detach().requires_grad_(True)
policy_score, expert_score = discriminator(policy), discriminator(expert)
g = torch.autograd.grad(expert_score.sum(), expert, create_graph=True)[0]
gp = 10.0 * g.square().sum(-1).mean()
```



`create_graph=True` 保留 g 对 D 参数的依赖。因为 `d(g²)/domega = 2*g*d(g)/domega`，GP 涉及 D 参数与输入的混合二阶导。真实和专家数据在这一步都是固定输入，优化器只更新 D。将 `expert_score.sum()` 改成均值会让 g 先除以批次大小，再平方会改变 GP 强度。D 是逐样本的 ReLU MLP，先求所有分数的和再求梯度，得到的是每一对输入各自的分数梯度。



|每单位真实交互的专家抽样预算|真转移数 / 外层轮|D 更新数 × 每次专家对数|专家帧对抽样数 / 真实转移|
|---|---|---|---|
|原 PPO\-AMP，4096 环境|98,304|20 × 24,576|5|
|`supermimic_official` 默认，4096 环境|平均 10,240|2 × 512|0\.1|
|当前平地运行，4096 环境|平均 10,240|2 × 25,600|5|



PPO 按 24 个真实控制步、5 个 epoch、每 epoch 4 个小批次计算。当前配置仅把每次 D 更新的专家对和真实策略对扩大到 25,600，对数按抽样次数统计，同一帧对可能重复。专家帧对由每组的专家池抽取，真实策略对来自 Replay。D 的更新频率、回放分布、镜像走路样本和给 Actor 的梯度机制仍按 WMAC 配置执行。D 用独立优化器，因此这项对齐也会增加 D 阶段的计算量。实现入口为 `src/wmac/runner.py::ExpertPairs/update_discriminator`、`src/wmac/replay.py::sample_starts` 和 `rsl_rl/algorithms/amp_ppo.py::update`。



### 7\.3 PPO\-AMP与当前AMP反传路径为何不同



原PPO\-AMP先在真实轨迹上算标量奖励、优势，再用策略概率比更新：



```Plain Text
rho_t(theta) = pi_theta(a_t|o_t) / pi_old(a_t|o_t)
J_PPO = mean min(rho_t*Ahat_t, clip(rho_t,1−epsilon,1+epsilon)*Ahat_t)

基本梯度项 = Ahat_t * rho_t * grad_theta log pi_theta(a_t|o_t)
```



PPO 的奖励和优势来自已采集的轨迹。在未被裁剪且参与目标的分支，策略梯度由动作对数概率加权得到，D 的 `predict_amp_reward` 可以在 `no_grad` 下执行。当前 WMAC 直接最大化模型轨迹的回报，需要奖励、D、FK 和 WM 对输入的导数。这个差异也解释了为什么过去的奖励裁剪能影响想象路径，即使原 PPO 可以照常使用这种标量奖励。对应实现为 `rsl_rl/modules/discriminator.py::predict_amp_reward` 和 `src/wmac/runner.py::_amp_reward`。



旧裁剪LS风格奖励仍作为旧checkpoint兼容模式保留：



```Plain Text
r_quad(d) = c * max(1 − (d−1)^2/4, 0), c=0.1
−1<d<3: dr/dd = c*(1−d)/2
d<−1或d>3: dr/dd = 0
```



PyTorch在精确边界−1/3保留内侧次梯度，不能将闭区间边界也一概写成零。该函数在d\>1时斜率为负，d=1时为零。奖励在某些样本上被裁成零，不代表PPO的全部策略梯度为零；但它确实会切断WMAC该样本的即时AMP直接路径。



### 7\.4 当前居中、有界AMP奖励及其导数



当前从 InstantAMP 固定正尺度分支借鉴“真实策略分数 EMA 居中＋tanh 映射”。判别器仍按第 2 节的 LSGAN 和 GP 训练，映射复用这一个 D 的分数。 [InstantAMP 固定提交](https://github.com/RoboticsResearchHubVN/InstantAMP/blob/9fc8031055a18d58e84d9de90639660e7aa6b491/instantamp/amp.py#L106-L148)



```Plain Text
mu_robot 首次由一次成功D更新的真实策略分数均值初始化
随后：mu_robot ← 0.99*mu_robot + 0.01*mean(D(real_replay_pairs))

z = (d − stopgrad(mu_robot))/s
r_AMP(d) = (c/2)*(1+tanh(z))
c=0.1，s=1.0

dr_AMP/dd = c/(2s) * [1−tanh(z)^2]
```



因此奖励在 \[0, 0\.1\]，在策略均值处为 0\.05、斜率 0\.05。乘外部风格权重 0\.25 后，单步风格贡献上限为 0\.025，最大分数斜率为 0\.0125。scale 1 以当前 LSGAN 标签 ±1 的半跨度作为固定尺度。tanh 远端仍会饱和，浮点实现中导数可以精确为 0。映射单调增加，想象状态进入真实训练分布以外时，其高分也可能影响 Actor。



EMA 复用 D 训练中已计算的**真实策略、更新前分数**。优化器成功后更新 EMA，再按既有规则更新 AMP 归一化。两次 D 更新都成功且分数有效时，EMA 每轮推进两次。专家均值仅供诊断。EMA 作为只读 buffer 进入 Actor 的 BPTT，第一次有效更新初始化它后，Actor 的门控才允许开始。断点保留此统计及其初始化标记。



实际代码中的普通自动求导如下，**AMP映射没有自制backward或直通梯度**：



```Python
z = (score - self.robot_mean.detach()) / self.scale
reward = self.coefficient * 0.5 * (1 + torch.tanh(z))
```



令 `z_1=N(Phi(s_t))`、`z_2=N(Phi(s_(t+1)))`。对第 k 个动作的即时 AMP 项，用乘积形式可展开成两条链：



```Plain Text
d r_AMP,t / d a_k = (dr/dd) * [
    (partial D/partial z_1) * (partial z_1/partial s_t) * (d s_t/d a_k)
  + (partial D/partial z_2) * (partial z_2/partial s_(t+1)) * (d s_(t+1)/d a_k)]

其中 partial z_i/partial s = (partial N/partial Phi) * (partial Phi/partial s)
```



预测该转移前已固定的状态不会受本步动作影响，所以 k=t 时即时项沿第二端反传。若 k\<t，更早的动作可以影响相邻对的两端。完整 BPTT 还要对未来所有奖励、历史反馈和生存权重求导。冻结 D 参数时不生成 `grad_omega`，同时保留 D 对两个输入的导数。用 PPO 的 `predict_amp_reward` 替换想象中的 `_amp_reward` 会因其 `no_grad` 切断此路径。



AMP归一化在本配置中为`(x−mean)/sqrt(var+1e−4)`，再裁剪到±10。统计值在Actor步骤固定，但输入导数保留；被归一化裁剪的坐标仍可能没有导数。不能把tanh斜率恢复解释成整个状态或Actor梯度处处可靠。D更新不包BF16 autocast，输入/GP张量为FP32；默认生产配置TF32允许相应矩阵运算使用TF32数值路径。若显式选择BF16，Actor阶段D可在BF16 autocast中计算，分数转FP32再做奖励映射。



`direct_style_grad=False`只把即时style张量detach，前向风格值仍在总回报里：它还可通过生存权重的导数和学到的末端价值产生间接影响。当前居中tanh模式的 `style_zero_gradient_fraction` 仅统计分数→奖励映射的零斜率，不是整个Actor Jacobian是否为零。legacy模式的同名诊断按score≤−1或≥3计数，包含精确裁剪边界；它不等同于边界实际自动求导值。



### 7\.5 SuperMimic 如何用真实数据学习动力学



共享动力学描述同一台 G1 如何响应动作。行走、起身和不同速度指令的真实转移可以一起训练这一模型。每个真实窗口可重复用于动力学拟合，策略更新则在张量化模型内生成批量短轨迹并自动求导，梯度步骤无需为每个想象动作再调用真实物理。短视野控制误差累积，Critic 为短轨迹后的长期回报提供估计。训练中持续补入新策略的真实样本，覆盖策略访问状态的变化。 [SuperMimic 论文](https://arxiv.org/abs/2610.28480v2) · [官方动力学学习器](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/learners/wm_learner.py#L101-L166)



WMAC 的原始状态为 71 维，包含世界位置 3、wxyz 四元数 4、世界线速度和角速度各 3、29 维关节位置和 29 维关节速度。模型的状态编码为高度 1＋去 heading 后的旋转矩阵前两列 6＋根 body 坐标的线速度 3 和角速度 3＋关节位置 29＋关节速度 29，仍是 71 维。再拼 29 维动作和 35 维上下文，共 135 维。主模型输出 70 维局部残差，使用三层 2048 宽的 Linear → LayerNorm → GELU。既有接触成本分支在同一归一化输入上运行独立的 256/128 宽 SiLU MLP，预测两个非负原始成本。



```Plain Text
WM输入 = [invariant_features(s_t), a_t, context_t]
局部残差 = [dp(3), drot(3), dv(3), dw(3), dq(29), dqd(29)]
root_p_next = root_p + rotate(heading(q_root), dp)
q_root_next = normalize(q_root * Exp(drot))
root_v_next = root_v + rotate(q_root, dv)
root_w_next = root_w + rotate(q_root, dw)
joint_q_next = joint_q + dq
joint_dq_next = joint_dq + dqd
```



残差由输出归一化逆变换后积分，关节位置和速度各自使用预测残差。令 `LocalDelta(x,y)` 表示把 y 相对 x 的差转成上述 70 维局部坐标，实际监督为：



```Plain Text
scale_i = max(out_norm.std_i, error_floor_i)
e_t = mean_i [(LocalDelta(s_real,t+1, s_pred,t+1)_i / scale_i)^2]
e_contact,t = mean_foot_slip_and_collision SmoothL1(log1p(c_pred), log1p(c_real))
L_WM = sum_windows,t [mask_t * (e_t + e_contact,t)] / max(sum_windows,t mask_t, 1)
```



预测成本经 softplus 保证非负。模型学习下一状态与既定接触成本代理，想象中的任务、平滑及 AMP 奖励由各模块分别计算。每轮 10 次更新，每批 2,048 个窗口，K=7 个转移对应 8 个状态，共 0\.14 秒。窗口内从起点递推预测状态，动作与上下文使用记录值，mask 排除无效和跨事件转移。Actor 想象才会递推预测上下文、观测历史和策略新动作。模型监督与 Actor 目标分别更新各自参数。



对照官方参考源码，官方想象环境在预测下一状态后，使用该时刻的缓存参考位姿和速度计算跟踪奖励，并用参考高度、倾斜误差给出软终止。WMAC 使用速度指令任务、状态对 AMP、恢复计数和自己的局部残差积分。两个实现共享“真实数据监督动力学＋想象轨迹反传”的构造，模型输出含义和奖励函数各自由其任务定义。 [官方想象环境](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/envs/world_model_env.py#L163-L219)



原 MDP 的派生 body 姿态和速度缓存可能落后原始 qpos、qvel 一个 5 ms 物理子步，想象 FK 则直接分析预测状态。真实关节加速度取物理 qacc，想象使用 20 ms 速度差分。新增的四项平滑成本统一使用原始状态、动作历史、公式和 dt。复现时可按奖励项逐一核对两侧取样时刻。



### 7\.6 H32模型回报与BPTT推导



UTD0\.4表示每个真实向量采样tick平均触发0\.4次训练更新，当前累加器每轮采2或3个控制步，平均2\.5；不是把PPO rollout强改为2\.5帧。每个tick有4096个物理环境转移。Actor从Replay抽4096个真实起点，想象H32控制步（0\.64秒），整段反传后做一次Actor更新。



当前官方MLP Actor末层线性输出，没有内部动作tanh。真实探索在输出上加标准差0\.1的高斯噪声，再裁剪到±20；想象直接裁剪Actor输出到同一范围，不使用概率比或动作log\-prob。外层clamp在超出范围时仍会令动作方向局部导数为0；新AMP映射和请求力矩成本都不能绕过这个已有裁剪。



令d\_t为想象终止值，c\_t为“到这一奖励时仍存活”的权重：



```Plain Text
c_0=1
c_(t+1)=c_t*(1−d_t)
r_t=0.75*r_task_new,t+0.25*r_AMP,t

J(theta)=mean_batch [ sum_(t=0..H−1) gamma^t*c_t*r_t
                     + gamma^H*c_H*min(V_nu(x_H),Vmax) ]
L_actor = −J(theta)
gamma=0.99，H=32
```



为看清历史反馈，令y\_t包含状态、上下文、4帧历史、旧动作和生存权重，`a_t=pi_theta(y_t)`、`y_(t+1)=G_phi(y_t,a_t)`。把第t项写为`l_t=gamma^t*c_t*r_t`，末项写为B。采用列梯度，反向递推为：



```Plain Text
p_H = grad_y B(y_H)
q_t = grad_a l_t + G_a^T * p_(t+1)
p_t = grad_y l_t + G_y^T * p_(t+1) + pi_y^T*q_t
grad_theta J = sum_t pi_theta^T*q_t
```



其中`pi_theta`表示Actor对其参数的Jacobian，不是Actor函数值；`pi_y`表示对历史输入的Jacobian。遗漏最后的反馈项，就忽略了预测状态改变后Actor下一步动作也会变化。PyTorch对完整展开图自动完成这组递推。



Critic 参数冻结时，`V(x_H)` 对末端特征 x\_H 的导数仍参与 Actor。价值高于 Vmax 时，上界裁剪截掉这条局部价值导数。`survival_grad=False` 只截断 c\_t 的导数，前向值照常计算，task 中终止惩罚仍有其自身路径。当前保留生存权重梯度。



### 7\.7 Critic的lambda目标为何必须detach



Actor回报计算后、任何相关参数更新前，缓存旧价值与想象奖励，并在no\_grad下从后向前计算：



```Plain Text
R_lambda,H = V_old(x_H)
R_lambda,t = r_t + gamma*(1−d_t)*[
                 (1−lambda)*V_old(x_(t+1)) + lambda*R_lambda,t+1]
lambda=0.95

target_t = min(R_lambda,t, Vmax)       # 官方模式最后统一裁剪目标上界
L_V = mean [V_nu(stopgrad(x_t)) − stopgrad(target_t)]^2
```



等价地令`delta_t=r_t+gamma*(1−d_t)*V_old(x_(t+1))−V_old(x_t)`，则`A_t=delta_t+gamma*lambda*(1−d_t)*A_(t+1)`、`A_H=0`，有`R_lambda,t=V_old(x_t)+A_t`。这说明裁剪目标前的lambda\-return与相应GAE的关系，不意味着Actor改用优势概率比。当前代码先用未逐步上界裁剪的旧价值递推lambda\-return，再对最终各步目标统一上界裁剪；不能擅自把每个递推中间值也裁一次。



当前按5 epochs×4 minibatches拟合Critic，共20次；官方对齐模式在一次fit中复用同一随机分组。默认`real_critic_coef=0`，没有新增真实回报分支或Critic预热。奖励新增项会同时进入Actor与lambda目标，不能只给Actor扣分而让Critic继续学习旧目标。



### 7\.8 哪些图冻结、哪些图保留



|阶段|更新参数|保留的导数|必须断开的部分|
|---|---|---|---|
|真实采样|无网络优化|无需物理图|Replay状态/动作/历史与旧物理计算图断开|
|WM拟合|WM主干＋既有接触分支|多步预测对WM参数|Actor、D、真实标签不参与更新|
|D拟合|D|D参数、GP所需混合二阶导|真实/专家特征先detach；仅专家z重新requires\_grad|
|Actor BPTT|Actor|WM、D、Critic对输入；FK；历史；任务与平滑成本|WM/D/Critic参数；归一化统计；AMP中心；真实起点|
|Critic拟合|Critic|V对自身参数|想象输入特征、奖励、continuation、旧价值和lambda目标全detach|



核心上下文是`frozen_parameters(WM,D,Critic)`，不是把整个想象放入`torch.no_grad()`。前者临时将参数requires\_grad设False，仍允许梯度穿过输入；后者会直接拆掉Actor需要的路径。原代码还会检查Actor步骤是否向冻结网络泄漏参数梯度。



RunningNorm的forward从不更新统计；更新只发生在明确的真实数据步骤。想象观测噪声预先采样为无梯度张量；动作历史通道噪声为0，二阶动作成本可复用真实a\_\(t−2\)。reset后ActionManager将动作通道归零，观测环形缓冲第一次写入时用该有效帧填满4个历史时刻，而不是将前三个姿态/速度历史当全零；因此首步历史动作可以为零，但重力、位置等仍是有效观测。Replay窗口起点detach，但后续想象动作和状态不应逐步detach，否则会变成另一种截断训练法。



### 7\.9 恢复终止代理的直通梯度



`hard_forward_v2` 的恢复延迟终止使用有偏的直通代理。真实前向在根高度低于阈值或姿态角超过限制时判定为坏姿态。延迟环境连续坏姿态时计数加 1，恢复良好则归 0，达到 `max_delay_steps=250` 时终止并清零。计数和阈值判定在 `no_grad` 内执行。



```Plain Text
bad_soft = 1−(1−bad_height_soft)*(1−bad_tilt_soft)
hazard = bad_soft * sigmoid((stopgrad(candidate)−max_delay_steps+0.5)/0.25)
done_delayed = ready_hard + hazard − stopgrad(hazard)
```



前向 `hazard−stopgrad(hazard)=0`，因此延迟终止使用与真实规则一致的 0/1 触发。反向取当前软坏姿态的导数乘临近到期的门控。普通非延迟终止仍采用软坏姿态代理。真实数据的 timeout 作为独立边界记录。若 `max_delay_steps<=0`，代码保留计数、返回软 bad 路径。



直通导数用于补充离散计时器的优化信号，由软 hazard 定义。恢复高度奖励则用 `exp(−height_error²/std²)*clamp(up_z,0,1)²` 直接求导，正立度≤0 时处在该门控的平坦区。



### 7\.10 平滑成本和想象足滑的梯度



四项平滑成本先按原始量加权，作为负成本进入 task，并乘一次 dt。AMP 奖励直接乘外部的 0\.25 风格权重。令 `delta2_a=a_t−2*a_(t−1)+a_(t−2)`，原始平方成本对三个动作的梯度分别为 `2*delta2_a`、`−4*delta2_a`、`2*delta2_a`。首步两个历史动作来自真实观测，之后的历史动作连接此前 Actor 输出。关节速度平方和对预测下一速度求导。



力矩成本使用控制步开始时、裁剪前的 PD 请求范数，`tau_request=Kp*(offset+scale*a−encoder_bias−q)−Kd*dq`。对动作的 Jacobian 含 `diag(Kp*scale)`，所以请求力矩在电机限幅外仍有普通解析梯度。功率项使用按扭矩转速曲线限幅并扣除摩擦后的指令代理乘当前 dq，自动求导遵循该分段限幅，在部分饱和区对本步动作可以没有局部导数。两项都从已有的电机常数和状态解析计算，物理评估可另外记录 5 ms 子步的实际执行器力矩。



主运行的想象足滑使用 `learned_v1`。梯度由足滑成本预测分支传到当前预测状态和动作，真实仿真仍用足部接触传感器与奖励 site 的水平速度。代码另外提供 `geometry_instantamp_full_support_v1`，仅当平地的**转移前**处于延迟恢复环境，且骨盆高度低于 0\.55 m 或计数大于 0 时，替换该步想象足滑。用**预测下一状态**的 FK 计算每脚胶囊最低点 h 和原奖励 site 的水平速度 v：



```Plain Text
active = 1[||command_xy|| + |command_yaw| > 0.1]
w = sigmoid((h0 − h) / tau), h0=tau=0.001 m
c_slip = active * sum_feet [w * ||v_xy||^2]
d(c_slip) = active * sum_feet [
    2*w*v_xy · d(v_xy) − (w*(1−w)/tau)*||v_xy||^2 * d(h)]
```



完整模式同时保留水平速度和离地高度两条梯度，抬脚也可能降低此成本。历史 `geometry_detached_support_v1` 对接地权重 stopgrad，在使用同一高度与宽度时前向相同，只保留公式的第一项。几何模式沿用足滑权重 −0\.25，同时保留行走足滑和自碰撞的预测成本。当前只替换符合条件的想象足滑值，既有动力学和两路成本监督仍照常执行。实现见 `src/wmac/state.py::recovery_foot_slip_mask/task_reward` 与 `src/wmac/foot_slip.py::G1FootSlipGeometry`。 [InstantAMP 接地几何](https://github.com/RoboticsResearchHubVN/InstantAMP/blob/9fc8031055a18d58e84d9de90639660e7aa6b491/instantamp/sim.py#L248-L264)



### 7\.11 优化与精度属于复现协议的一部分



Actor、WM、Critic 各有独立优化器。每个 Sequential 的中间隐藏 Linear 权重（Linear 列表 `[1:-1]`）使用 Muon，其余参数用 Adam。Muon 固定 momentum 0\.95、Nesterov、5 步 Newton–Schulz、系数 \(3\.4445, −4\.7750, 2\.0315\)、weight decay 0\.1 及匹配 RMS 的学习率，正交化内部用 BF16。Actor 的 Adam 部分 betas=\(0\.9, 0\.95\)，WM 与 Critic 使用默认 Adam betas。三者恒定学习率均为 2e−4。D 使用独立 Adam，初始学习率 1e−3，前 250,000 轮按余弦衰减至初始值的 0\.03。



优化前先检验loss有限，再全参数范数裁剪到1；Actor额外逐元素裁剪到±0\.1，然后执行对应优化器。官方模式不因有限的验证误差或动作偏移超过兼容阈值而额外回滚，但仍拒绝非有限/失败更新。默认TF32的完整梯度定义与FP32相同，数值精度和跨设备结果不保证逐位一致。



模型导数的可靠性需要用真实物理轨迹检验。11 月 20 日的等预算恢复重置对照冻结了 Actor，在同一批留出初态与真实动作上比较两组模型。分层重置的恢复归一化状态 RMSE 在第 7 步下降 11\.4%、第 32 步下降 20\.4%，第 1 步综合差异的区间跨零。低位近直立子组第 7 步误差增大。分层组第 1、7 步误差也仍高于续训前断点。这个结果衡量了单种子、单恢复片段的短程续训和开环预测。该抽样试验仅用于诊断，主运行继续使用原重置规则。



### 7\.12 原始资料和固定源码



- 动作片段分布与对抗风格奖励的原始定义。 [AMP 原论文](https://arxiv.org/abs/2104.02180)

- 真实轨迹上的概率比裁剪策略目标。 [PPO 原论文](https://arxiv.org/abs/1707.06347)

- 想象训练的共享动力学、反向传播和 Critic 拟合。 [SuperMimic 论文 v2](https://arxiv.org/abs/2610.28480v2) · [固定官方代码](https://github.com/HybridRobotics/supermimic/tree/abc124c1bd5976341034668af3d12c087f2f1f55)

- 固定尺度的 style 函数与真实分数 EMA。 [InstantAMP 固定实现](https://github.com/RoboticsResearchHubVN/InstantAMP/blob/9fc8031055a18d58e84d9de90639660e7aa6b491/instantamp/amp.py#L106-L148)

- 重建源码中的核对入口为 `src/wmac/runner.py` 的 `update_discriminator`、`imagine`、`lambda_returns`、`update_actor_critic`，以及 `src/wmac/amp_rewards.py`、`src/wmac/state.py`、`src/wmac/networks.py`、`src/wmac/optimizers.py`。

    

## 第8章 全部奖励、单位与梯度



本章对应 `src/tasks/amp_loco/amp_env_cfg.py`、`src/tasks/amp_loco/mdp/rewards.py`、`src/recovery_rewards.py`、`src/wmac/state.py`、`src/wmac/reward_integration.py`、`src/wmac/smooth_rewards.py`。下文计算采用 4096 环境、V3 平滑系数、AMP 判别器批量 25600 的运行配置。



物理积分周期为 0\.005 秒，每 4 个物理子步执行一次策略，控制周期为 0\.02 秒。因此真实控制一步、Replay 相邻一行、世界模型预测一步和专家相邻两帧都按 20 ms 解释。专家数据是 50 Hz；D 更新次数与 UTD 不会把参考片段变成倍速播放。



先在一个控制步内定义“奖励率”，再乘一次 dt：



```Plain Text
dt = 0.02 s

R_task_original(t) = dt * sum_i [w_i * base_term_i(t)]
P_smooth(t) = 0.0175*C_second(t)
            + 0.000625*C_velocity(t)
            + 0.000625*C_torque(t)
            + 0.00003125*C_power(t)

R_task_new(t) = R_task_original(t) - dt*P_smooth(t)
R_total(t) = 0.75*R_task_new(t) + 0.25*R_AMP(t)
```



`R_AMP` 的幅值系数在映射内取 0\.1，按控制步计，直接参加混合。0\.75 / 0\.25 是任务与风格的混合系数，风格每步对总奖励的贡献在 0 到 0\.025 之间。任务贡献仍取决于各项实际值，Actor 接收的风格梯度也取决于 D 和世界模型的局部导数。



代码对应两个入口：真实 RewardManager 注册一个 `wmac_smooth_penalty`，其 weight 为 \-1，返回四项成本的加权和，由 RewardManager 按现有设置乘 dt；想象端 `_task_reward()` 调原任务函数后减去同一个 `dt*P_smooth`。两条入口不是在同一条轨迹上连续执行，因此没有“双扣”；各自得到同一定义的 task reward。



### 8\.1 基础任务的恢复门控



设 `b_t` 为是否分配到延迟终止组，`c_(t+1)` 为本步后的连续坏姿态计数。当前物理奖励的恢复激活条件以及想象端对应的前向量为：



```Plain Text
physical_active = delayed_env AND (counter_after_step > 0)
imagined_active = delay_flag * clamp(counter_after_step, 0, 1)
```



`hard_forward_v2` 前向计数为整数，所以该量的前向值为 0 或 1。不是所有被分到恢复组的机器人在整段 episode 中都领取恢复高度奖励；它必须仍处于连续坏姿态时钟的激活阶段。恢复后该时钟归零，正常行走奖励重新激活。



真实延迟终止组比例是 0\.4，最长连续坏姿态为 250 个控制步，即 5 秒。`recovery_fraction=0.5` 是 Replay 起点采样比例，`expert_recovery_fraction=0.5` 是 D 的专家片段混合比例；这三个数字用途不同，不应互相替代。



`MotionResetManager` 启动时随机选定延迟组。普通环境从汇总后的行走帧均匀抽样，延迟组当前从恢复片段的全部帧均匀抽样，能抽到倒地、过渡和站稳时刻。此前分层重置仅用于冻结 Actor 的诊断。相同预算下，严格倒地分层组的第 7、32 步归一化恢复预测 RMSE 分别由 2\.100 降为 1\.860、由 4\.066 降为 3\.236，第 1 步综合变化不明确。低位近直立起点第 7 步误差由 1\.486 增为 1\.732。这个单种子对照没有把分层采样合入训练入口，也没有训练新 Actor 测量动作相似度。生产抽样以 `src/tasks/amp_loco/mdp/events.py` 中的 `MotionResetManager._write_reset_state()` 为准。



### 8\.2 十项原任务奖励的精确定义



下表 w 是乘 dt 之前的系数。用 `u` 表示上面的恢复激活量，`q` 表示关节位置，`dq` 表示关节速度；真实量均在本步奖励计算时、reset 之前读取，想象中使用预测下一状态。原mjlab的部分body/sensor派生量在这个时刻滞后raw qpos/qvel一个物理子步；后文单独说明，不能把下表理解成两端全部逐位相同。



|项名|原始表达式与实际含义|w / 门控|
|---|---|---|
|`track_anchor_linear_velocity`|`exp(-sum((v_target_world-v_torso_world)²)/1.0²)`；目标是将 `(cmd_x,cmd_y,0)` 用 torso 的 yaw 旋转到世界坐标，比较三个线速度分量|\+1\.0，乘 `(1-u)`|
|`track_anchor_angular_velocity`|`exp(-[(cmd_yaw-omega_torso_world_z)²+sum(omega_torso_body_xy²)]/3.14²)`|\+1\.0，乘 `(1-u)`|
|`track_root_height`|`exp(-(pelvis_height-0.78)² / 0.3²) * clamp(pelvis_up_z,0,1)²`|\+1\.0，再乘 `3.5*u`|
|`body_ang_vel_xy_l2`|`exp(-sum(omega_pelvis_body_xy²)/3.14²)`；名称含 l2，但代码是正的指数奖励|\+0\.5，乘 `(1-u)`|
|`is_terminated`|本控制步实际终止标记；想象中为对应终止代理|\-200\.0|
|`joint_acc_l2`|真实端为末物理子步 `sum_j(qacc_j²)`；想象端为 `sum_j(((dq_next_j-dq_j)/0.02)²)`|\-2\.5e\-7|
|`joint_pos_limits`|`sum_j [max(q_low_j-q_next_j,0)+max(q_next_j-q_high_j,0)]`，使用机器人 soft limits|\-10\.0|
|`action_rate_l2`|`sum_j((a_t_j-a_(t-1)_j)²)`，是在动作缩放和 offset 之前的执行动作空间|\-0\.01|
|`foot_slip`|对接地脚的水平 site 速度平方求和；仅 `norm(cmd_xy)+abs(cmd_yaw)>0.1` 时激活|\-0\.25|
|`self_collisions`|当前配置有 4 个接触历史子步，统计每个子步中是否存在超过 10 N 的自身接触，再对历史求和|\-0\.1|



anchor 是 `torso_link`，高度门控与朝上量使用 `pelvis`。不要把 torso 的旋转误当 pelvis 的朝上方向。0\.78 米来自本机器人 `KNEES_BENT_KEYFRAME` 的默认根高度，不是从 XML 的默认 qpos 自动猜出的另一个高度。



默认soft limit由机械范围中点对称缩至原半宽的0\.9倍，公式为 `mid=(low+high)/2; half=(high-low)*0.9/2; soft=[mid-half,mid+half]`；机械硬限位仍不变。



终止条件包括根高度小于 0\.5 米，以及倾斜角严格大于 70 度；另有原来的 20 秒 time\-out。延迟组连续 bad 才积累时钟，恢复正常姿态即归零。time\-out 作为 Replay 边界单独处理，不应冒充坏姿态终止概率。



### 8\.3 恢复高度为什么有朝上约束



只有高度项时，倾斜身体抬高骨盆也可能拿到高分。`upright_v2` 保留原高度 Gaussian，并乘根部正向朝上的平方：



```Plain Text
up_z = 1 - 2*(normalized_root_quat_x² + normalized_root_quat_y²)
gate = clamp(up_z, 0, 1)²
height_reward = exp(-(height-0.78)²/0.3²) * gate
```



直立时 gate=1，水平或倒置时 gate=0，中间姿态连续变化。这是高度奖励的语义，不是给电机加了限幅，也不是对模型的改动。gate 在 clamp 的区间外可能没有导数，不能宣称它在所有姿态都提供连续的恢复方向。



### 8\.4 qacc 与接触成本中的代理边界



真实 `qacc` 每 5 ms 都会随接触求解变化，而 Actor 当前只能用控制步端点的预测速度作 20 ms 差分。即使预测的端点速度完全准确，平均变化与瞬时加速度仍不是同一量。这不是 `dt` 算错，也不能只把差异归咎于世界模型预测误差。



真实 RewardManager 以接触传感器计算脚滑和自身接触原始成本，作为现有 contact branch 的监督标签。当前 4096 运行使用 `learned_v1`，想象中的两项成本都由该分支输出非负预测。Actor 更新时分支参数冻结，梯度仍传回输入状态和动作。选择几何足滑时，监督标签与分支训练照常进行，只有平地恢复的想象足滑从几何计算取值。



#### 可选 InstantAMP 足滑与梯度



用步前状态和 context 选择想象分支，`delay_flag > 0.5` 且 `pelvis_z < 0.55 m 或 recovery_counter > 0` 时应用几何代理。支持权重与脚位点速度取自步后预测状态的同一份 FK。每只脚从 MuJoCo G1 的七个足底胶囊各取两个轴端点。若脚体位置为 p、朝向为 R、第 k 个端点在局部坐标为 e\_k、所在胶囊半径为 r\_k，则脚底净空 h、原 site 水平速度 v\_xy 与成本为



```Plain Text
h = p_z + min_k((R*e_k)_z - r_k)
v_xy = (v_body + omega_body × (R*site_offset))_xy
w = sigmoid((h0 - h) / tau)
C_slip = 1[||cmd_xy|| + |cmd_yaw| > 0.1] * sum_foot(w * ||v_xy||²)
R_slip = -0.25 * 0.02 * C_slip
```



两种几何模式都复用原脚位点、命令门控与任务系数。几何最低点用实模型的胶囊几何计算，支持权重是连续代理。真实仿真仍用 `feet_ground_contact`，非恢复想象状态仍由预测头提供足滑成本，自碰撞一直沿用预测头。



- `geometry_detached_support_v1` 用 `--recovery-foot-slip-contact-height-m 0.001 --recovery-foot-slip-softness-m 0.001` 可与完整梯度做同参数对照。接地权重作为常量，成本梯度只包含 `2*w*v_xy·dv_xy`。

- `geometry_instantamp_full_support_v1` 固定 `h0=tau=0.001 m`，保留同一速度梯度，以及 `-||v_xy||²*w*(1-w)/tau*dh`。后者让模型接触边缘的抬脚与压脚也影响损失。两模式参数相同时前向相同，完整模式的净空梯度会受胶囊最低点切换和 sigmoid 尾部影响。

    

本实现由 `src/wmac/foot_slip.py` 计算几何和梯度，`src/wmac/state.py` 选取恢复分支，`src/wmac/runner.py` 为 Actor 与 Critic 目标共用这一套想象奖励。训练入口没有自动打开几何模式。 [InstantAMP 几何参考](https://github.com/RoboticsResearchHubVN/InstantAMP/blob/9fc8031055a18d58e84d9de90639660e7aa6b491/instantamp/sim.py#L248-L264)



原mjlab的 `ManagerBasedRlEnv.step()` 有一个刻意保留的时序：奖励计算前 `xpos/xquat/cvel` 等派生量可能比raw `qpos/qvel`滞后一个5ms子步，随后 `sim.forward()` 才统一更新。真实奖励按原环境传感器/派生量读取，想象端按预测raw状态重新FK，因而基础项还存在这一时序代理差异。本次没有改物理引擎调用顺序或世界模型去“消除”它。新四成本不需要body FK，统一使用控制步起点raw状态、终点raw关节速度与真实动作历史，具备相同定义。



`soft_landing` 在原任务中是注释状态，没有启用。本次也没有把它偷偷加入；新增的四项不能被描述成“原版漏迁移的着陆奖励”。



### 8\.5 新增一：二阶动作差



```Plain Text
C_second = sum_j((a_t - 2*a_(t-1) + a_(t-2))_j²)
coefficient = 0.0175
```



一阶差限制“变化有多大”，二阶差限制“变化量改变得有多突然”。它是动作空间成本，不能直接称为物理关节 jerk。真实关节轨迹还受到 PD、力矩转速曲线、地面接触和惯量影响。



令 `e_t=a_t-2a_(t-1)+a_(t-2)`。在不计同一动作出现在其他时刻成本中的贡献时：



```Plain Text
dC_t/da_t       =  2*e_t
dC_t/da_(t-1)   = -4*e_t
dC_t/da_(t-2)   =  2*e_t
```



想象序列中同一动作还会出现在后两步的成本里，因此反向传播必须累积全部贡献。把 `a_(t-1)` 或 `a_(t-2)` 在每个想象步 detach，会改变这个目标的梯度。



复用现有观测历史即可获得起点前两动作。Actor 的 4×96 time\-major 帧中，末尾29维为上一执行动作；因此第一个想象步的 `a_(t-2)` 是 `history[:, -2, 67:96]`，第二步的更老动作是起点真实 `a_(t-1)`，后续来自之前想象出的动作。没有新增 Actor 输入、WM 输入或 Replay 字段。reset 时原动作缓存归零、观测历史缓存清空；下一次有效观测会由 CircularBuffer 将这一帧复制填满4个历史位置。因动作分量为零，起点前两动作仍可正确取零，但关节位置、重力等观测不是“三帧零加一帧新值”。不跨 episode 接历史。



### 8\.6 新增二：关节速度平方



```Plain Text
C_velocity = sum_j(dq_next_j²)
coefficient = 0.000625
dC_velocity/ddq_next = 2*dq_next
```



使用下一状态的关节速度，真实与想象端的时间点一致。它惩罚速度幅值，不惩罚每个脚接地时刻的冲量；低速度也不必然意味着成功起身或动作自然。原 qacc 成本仍保留，本次不是将其替换掉。



### 8\.7 新增三：裁剪前 PD 请求力矩的 L2 范数



现有世界模型没有输出真实电机力矩。本次不扩大输出，而是从同一状态、动作和 context，用原电机公式计算控制步起点的请求值：



```Plain Text
scale_j = 0.25 * nominal_effort_limit_j / Kp_j
target_j = default_joint_position_j + scale_j*a_t_j - encoder_bias_j
tau_request_j = Kp_j*(target_j - q_t_j) - Kd_j*dq_t_j
C_torque = sqrt(sum_j(tau_request_j²))
coefficient = 0.000625
```



encoder bias 取自现有 context 的第1到29号分量（Python切片 `context[...,1:30]`）；第0号分量是脚底摩擦，不能把这两个切片混淆。真实执行器目标也减 encoder bias，因此该符号不是经验调整。



在非零请求处，`d||tau||₂/dtau=tau/||tau||₂`。链到动作的直接导数为：



```Plain Text
dC_torque/da_j = (tau_request_j / ||tau_request||₂) * Kp_j*scale_j
```



还存在经过预测 q、dq 和前序动作的导数，全部由普通 autograd 计算。零范数处 `torch.linalg.vector_norm` 的当前反向实现返回零；测试覆盖了这一点。



本项有意罚裁剪前请求。若只对饱和后的力矩求普通导数，已经超出限幅的动作可能收到零力矩梯度。使用解析请求成本保留 PD 公式的真实导数，不使用 straight\-through，也不改变实际电机限幅。它是控制请求负荷代理，不能写成“世界模型准确预测了瞬时真实力矩”。



### 8\.8 新增四：限幅和摩擦后的指令功率代理



原电机按速度和功方向选择力矩曲线：



```Plain Text
same_direction = (dq * tau_request > 0)
y = Y1 if same_direction else Y2
y = min(y, nominal_effort_limit)

bound = y                                      if abs(dq) < X1
bound = max(0, y*(X2-abs(dq))/max(X2-X1,1e-6))   otherwise

tau_clip = clamp(tau_request, -bound, bound)
tau_command = tau_clip - Fs*tanh(dq/Va) - Fd*dq
C_power = ||tau_command * dq_current||₂
coefficient = 0.00003125
```



代码对乘积取绝对值后再求 L2 范数；数值上与乘积直接求 L2 相同。这里用当前速度，不是下一速度，也不是四个物理子步真实功率的累计。clamp、符号分支和速度转折仍有分段导数；没有给它们另加隐蔽直通梯度。力矩成本使用裁剪前请求，功率成本使用限幅/摩擦后的指令代理，两者的含义不同。



各常数由实际初始化的 `JointPositionAction` 和 `UnitreeActuator` 读取，检查29个关节映射完整、常数对环境一致，并写入checkpoint哈希。实际训练使用 `from_env()`；CPU校准使用 `from_config()`，后者应取 `KNEES_BENT_KEYFRAME` 的 default offset，不能取 XML 的零位代替。



|关节组|Kp约值|Kd约值|名义effort\_limit|
|---|---|---|---|
|hip pitch/roll、knee|99\.0984|6\.3088|139 Nm|
|hip yaw、waist yaw|40\.1792|2\.5579|88 Nm|
|waist/ankle pitch/roll|28\.5012|1\.8144|50 Nm|
|shoulder、elbow、wrist roll|14\.2506|0\.9072|25 Nm|
|wrist pitch/yaw|8\.6110|0\.5482|10 Nm|



名义effort\_limit不等于每个状态下的最终可用力矩；实际还按 Y1/Y2、X1/X2 和摩擦处理。默认动作尺度、上述 PD、曲线和实际控制器均未因新奖励改变。



### 8\.9 系数演进与验证范围



V1 为 `0.01 / 0.00025 / 0.00025 / 0.000025`，V2 为 `0.015 / 0.0005 / 0.0005 / 0.000025`，当前 V3 为 `0.0175 / 0.000625 / 0.000625 / 0.00003125`，顺序为二阶动作差、关节速度、PD 请求力矩和指令功率。HoST、HumanUP 的动作维数、范数、dt 和训练任务各异。力矩平方和与力矩 L2 范数连系数单位都不同，配置值只能连同公式和量级一起比较。



此前 V1 在固定旧 WMAC 真实轨迹上，两批恢复段的扣分约占原任务奖励的 6\.22% 和 4\.96%，行走约 0\.76%。这些比例是 V1 的固定轨迹校准。另有 12 份记录、96,000 个控制步用于核对实现，解析请求和限幅结果与首物理子步原执行器记录逐元素一致。当前公式所表达的物理量限于控制步起点代理，后续物理子步与碰撞峰值需从仿真记录读取。



历史奖励实现通过 CPU 检查、小网络物理烟测和 4096 环境的一轮旧权重迁移。验证范围为公式、梯度和恢复协议。长期起身柔和程度要在固定初态下测加速度、受力和参考动作相似度，原 `best` 选择依然使用既有成功率、续走和速度等指标。



### 8\.10 Critic 价值上界



官方细节对齐模式使用保守正奖励上界，不用 tracking 项目的上界原样套在 AMP 上：



```Plain Text
positive_rate_upper_bound = 1 + 1 + 3.5 + 0.5 = 6
R_upper = 0.75 * 0.02 * 6 + 0.25 * 0.1 = 0.115
V_upper = R_upper / (1-0.99) = 11.5
```



这是保守上界，几个恢复/行走奖励的门控实际上不能同时取各自最大值。新增四成本均非正，因此不提高上界。代码只对价值加上界裁剪，没有新增负下界。上界点的 clamp 会影响价值对输入的导数，不能在推导里把 bootstrap 当始终线性且无裁剪的量。





## 第9章 正式训练、保存、恢复和评估



### 9\.1 完整默认配置与必须写出的开关



完整启动命令选择：



```Plain Text
profile = supermimic_official
amp_reward_mapping = instantamp_centered_tanh_v1
smooth_reward_mode = host_humanup_v1
imagined_recovery_foot_slip = learned_v1
discriminator_updates = 2
discriminator_batch_size = 25600
```



直接运行 `python scripts/train_wmac.py --profile supermimic_official` 时，AMP 映射为 `clipped_lsgan_v1`，平滑模式为 `none`，D 批量为 512。`train_smooth_weights_v3.sh` 传入居中 AMP 与四项 V3 平滑系数。正式命令再显式确定 D 批量和足滑模式。



|参数|当前完整命令|
|---|---|
|task / num\_envs|`Unitree-G1-AMP-Flat-Aligned` / 4096|
|seed / iterations|42 / 400000|
|精度 / 编译|TF32 / `torch.compile=True`|
|Actor|384→2048→1024→512→256→29，GELU＋LayerNorm，原外层clip±20|
|WM|隐层2048×3，原71状态/35context/29动作，10次更新×2048样本|
|WM监督窗口|7个相邻transition，对应8个存储状态帧|
|imagination|H32、B4096、Actor每轮1次更新|
|Critic|输入136、隐层256/128，5 epochs×4 minibatches|
|D|隐层1024/512/256，每轮 2 次更新，每次各抽 25,600 个策略帧对与专家帧对|
|task/style|0\.75/0\.25，AMP幅值0\.1|
|平滑成本系数|0\.0175 / 0\.000625 / 0\.000625 / 0\.00003125|
|恢复语义与想象足滑|`hard_forward_v2`、`upright_v2`、`learned_v1`|
|γ / λ|0\.99 / 0\.95|
|UTD|0\.4，每轮真实向量控制步按2、3交替，平均2\.5|
|初始物理缓冲填充|8个控制步|
|模型预热 / Critic校准轮数|0 / 0|
|真实Critic监督系数|0，不加入新的真实回报分支|
|Replay容量|1000个向量控制时刻；默认checkpoint不保存Replay|
|save / log / evaluation|每400 / 4 / 2000轮|



GPU训练规模在48GB RTX 6000 Ada上验证过；仅奖励组件基准的几GB显存不代表完整训练的显存。其他GPU的峰值会受驱动、编译与缓存影响。减小 `--num-envs` 或网络只适合资源受限的验证，不再是表中同规模配置。首次编译耗时不应计入稳态每轮时间，再据此误判训练退化。



当前正式配置每 400 轮保存一个编号权重，单个约 113 MiB。若 40 万轮训练期间都保留，编号权重将占约 110 GiB，本地磁盘还需容纳环境、评估、缓存和日志。烟测的模型和磁盘开销小得多。



#### 按物理交互量对齐 AMP 批量



旧 PPO 配置每轮采集 `4096 × 24` 条物理转移，执行 `5 × 4` 次 D 更新，每次为每类取 `4096 × 24 / 4 = 24,576` 个帧对。因此按“每类抽取帧对数 / 物理转移数”计，PPO 的名义比为 5。WMAC 仍按 UTD 0\.4 在 2、3 个物理向量步之间交替，平均每轮采集 `4096 × 2.5 = 10,240` 条。原 D 批量 512 时每类每物理转移只抽 `2 × 512 / 10240 = 0.1` 个帧对。将批量调为 25,600 后，平均比为 `2 × 25600 / 10240 = 5`。



此处对齐的是采样次数。专家帧对有放回抽样，固定 50% 来自恢复组，在批量增大时同一参考帧对可以重复出现。WMAC 仍以 Replay 中的有效真实转移为策略样本，D 仍只更新 2 次，AMP 幅值和混合权重维持原值。批量放大了 D 每轮正反向处理的样本量和峰值显存，真实物理交互量和世界模型的更新预算保持原配置。依据为 `rsl_rl/algorithms/amp_ppo.py` 的 `AMPPPO.update()` 和 `src/wmac/runner.py` 的 `update_discriminator()`。



### 9\.2 无当前机器访问权限的从头训练



完整手册内嵌的迁移源码是必要输入：只 clone 原始公开 AMP\_mjlab 和官方 SuperMimic，不能得到本项目私下整合后的全部改动。无需公司盘或当前电脑，但必须把完整Markdown或自包含脚本保存到新电脑；任何删除了源码附录的节选都不能独自重建。完整Markdown和其Feishu导入内容必须保留复现数据，不能只保存讲解部分。



复现脚本的默认目录固定为：



```Plain Text
workspace: $HOME/amp_wmac_smooth_ws
project:   $HOME/amp_wmac_smooth_ws/AMP_mjlab_wmac_smooth_rewards
logs:      $HOME/amp_wmac_smooth_ws/results
Conda:     $HOME/amp_wmac_smooth_ws/tools/miniforge3
env:       .../envs/amp-wmac-smooth
```



默认重建、安装、CPU测试、镜像准备与小规模GPU烟测完成后，要正式从头训练：



```Bash
source "$HOME/amp_wmac_smooth_ws/activate_local.sh"
cd "$HOME/amp_wmac_smooth_ws/AMP_mjlab_wmac_smooth_rewards"
bash scripts/train_smooth_weights_v3.sh \
  --num-envs 4096 \
  --imagined-recovery-foot-slip learned_v1 \
  --discriminator-updates 2 \
  --discriminator-batch-size 25600 \
  --log-root "$AMP_LOG_ROOT" \
  --run-name local_4096_ppo_amp_budget
```



若使用 `--workspace`，上述各路径整体换成脚本打印的该目录；不是再去寻找作者电脑文件。不要在新电脑直接依赖激活脚本的默认 `yao-rl` 环境名，重建脚本创建的是独立的本地环境，并用 `activate_local.sh` 选择它。



此命令没有 `--resume` 或 `--init-policy`，Actor、Critic、WM、D全部随机初始化，不需要旧PPO60000或当前WMAC96800权重。新目录也不创建指向公司盘的日志符号链接。原17条行走和1条恢复 NPZ来自固定公开基线，镜像在本地生成；不引入复杂地形或感知数据。



另开同规模的 InstantAMP 几何梯度实验，在同一环境执行下列命令。开关会记录在 `config.json`，几何的接触中点和平滑宽度各固定在 0\.001 m。



```Bash
bash scripts/train_smooth_weights_v3.sh \
  --num-envs 4096 \
  --imagined-recovery-foot-slip geometry_instantamp_full_support_v1 \
  --discriminator-updates 2 \
  --discriminator-batch-size 25600 \
  --log-root "$AMP_LOG_ROOT" \
  --run-name local_4096_ppo_amp_budget_instantamp_slip
```



### 9\.3 保存、恢复与显式目标迁移



当前 learned 足滑与平滑奖励的实现版本为 12。打开几何足滑时为 13，并写入几何协议与参数。权重同时保存 AMP 映射、EMA、四项成本协议、解析电机常数及关节映射哈希。恢复时沿用权重保存的协议与批量，显式迁移授权每次单独传入。



从本机已有的同目标 checkpoint 续训时，先将 `WMAC_CHECKPOINT` 设为该文件的完整路径：



```Bash
python scripts/train_wmac.py --resume "$WMAC_CHECKPOINT" \
  --log-root "$AMP_LOG_ROOT" --run-name local_4096_resumed
```



这是从自己的新电脑运行产生的权重恢复，不要求作者电脑上的任何旧权重。`iterations` 是目标总轮数，不是额外再训练多少轮，恢复时必须大于保存迭代。默认未保存Replay，恢复也不是物理状态逐步无缝接续。源配置会恢复，不能另外重贴不一致的 profile；不要把相同 Actor 输入维数当作可任意互换网络架构的理由。



如果用户另外带来了旧目标的 WMAC权重，`none→host_humanup_v1` 才需要 `--allow-smooth-reward-migration`；AMP旧clip映射→居中tanh需要独立的 `--allow-amp-reward-migration`。新电脑从头训练不需要任何迁移开关。



已启用平滑奖励的续训若要调整这四项系数，用 `--allow-smooth-reward-retune`。已有 `learned_v1` 权重如果要开启几何足滑，用 `--allow-imagined-foot-slip-migration`，并显式给出模式。只改变该目标及实验的运行参数，D 批量和其他训练目标继承断点。例如在新电脑上已产生自己的 `model_*.pt` 后可执行



```Bash
python scripts/train_wmac.py \
  --resume "$WMAC_CHECKPOINT" \
  --imagined-recovery-foot-slip geometry_instantamp_full_support_v1 \
  --allow-imagined-foot-slip-migration \
  --log-root "$AMP_LOG_ROOT" \
  --run-name instantamp_slip_fork
```



`WMAC_CHECKPOINT` 设为该权重的完整路径。若为最终轮次权重，增加一个高于该轮次的 `--iterations` 目标。



目标迁移保留四网络、四优化器、归一化、随机状态、UTD小数累加器和训练计数。若旧断点包含Replay，新平滑目标迁移时丢弃其中缺少新成本的旧reward标签；正常的新目标恢复则可保留匹配的Replay并切断重建模拟器边界。没有新增模型/ Critic预热阶段；无Replay的冷启动仍需原有8控制步填充。目标变化会重置新分支best选择记录，并保存源权重哈希与迁移历史。



### 9\.4 评估、训练奖励与历史取证的解释



Actor播放与共同物理条件的评估继续可用，没有加动作滤波。公平物理对照会把评估 task reward 标注为共同基线奖励、未含训练新惩罚；不能把这个字段当成新训练的完整混合目标。历史未适配的新目标梯度探针明确拒绝，而不是漏算新成本后给出错误结论。



旧PPO60000权重的观测归一化统计强烈支持 time\-major 历史输入，但缺少当次训练日志，不能据此证明具体补丁安装文件的历史身份。可复现方案直接核对并安装固定补丁，明确设置Actor/Critic的 `history_ordering="time"`，不依赖历史猜测。Actor384维本身不能区分time/term，它们维数相同而排列不同。



### 9\.5 开源来源与方法差异



- HoST固定 `70bb580949a336a920833700e4b5dc3bf7fe87ce`：[二阶动作差源码](https://github.com/InternRobotics/HoST/blob/70bb580949a336a920833700e4b5dc3bf7fe87ce/legged_gym/legged_gym/envs/base/host_ground.py#L1064-L1074)。本次只借鉴奖励项，不移植相对于current\_q的控制器或动作幅度课程。

- HumanUP固定 `7516e0f27e6f4d1e7365cf64ea577a78247bd8cb`：[速度、力矩和功率成本](https://github.com/RunpeiDong/humanup/blob/7516e0f27e6f4d1e7365cf64ea577a78247bd8cb/simulation/legged_gym/legged_gym/envs/g1track/g1waist_track.py#L821-L865)。本项目力矩成本改为原PD的裁剪前请求范数，是为可微WMAC作的明确适配，不是原项目实际力矩监督的复刻。

- 原版AMP与本项目的真实奖励源头为公开基线 `6c7a2947fccc973e4af8e6d90e550400f1b6fcfc`。旧PPO的likelihood\-ratio梯度与本Actor的直接模型BPTT不同，不能以“奖励式子相同”推断梯度相同。

- 对应许可证已随源码附上：HoST MIT、HumanUP相关文件BSD\-3\-Clause；保留第三方声明，不从未固定的latest下载替换当前实现。

    

### 9\.6 在自己新电脑上选择权重、评估与播放



激活脚本会导出AMP\_LOG\_ROOT。下面自动选择最近一次正式 `local_4096_ppo_amp_budget` 运行；若尚未训练，会明确报错，不拿烟测权重冒充正式结果。



```Bash
source "$HOME/amp_wmac_smooth_ws/activate_local.sh"
AMP_RUN_DIR="$(python - <<'PY_PICK_RUN'
import os
from pathlib import Path
root=Path(os.environ['AMP_LOG_ROOT'])
runs=[p for p in root.glob('*_local_4096_ppo_amp_budget') if p.is_dir()]
assert runs, '尚无正式local_4096_ppo_amp_budget运行；先启动第9章正式训练'
print(max(runs,key=lambda p:p.stat().st_mtime))
PY_PICK_RUN
)"
AMP_CHECKPOINT="$(python - "$AMP_RUN_DIR" <<'PY_PICK_CKPT'
from pathlib import Path
import re,sys
run=Path(sys.argv[1])
models=[p for p in run.glob('model_*.pt') if re.fullmatch(r'model_[0-9]+\.pt',p.name)]
assert models, '此运行尚未保存编号权重'
print(max(models,key=lambda p:int(p.stem.split('_')[-1])))
PY_PICK_CKPT
)"
printf '%s\n' "$AMP_CHECKPOINT"
AMP_EVAL_DIR="$AMP_RUN_DIR/manual_eval_$(date +%Y%m%d_%H%M%S)"
python scripts/evaluate_wmac.py --checkpoint "$AMP_CHECKPOINT" \
  --mode both --num-envs 32 --steps 250 --seed 2026 \
  --recovery-height-reward upright_v2 \
  --output "$AMP_EVAL_DIR" --save-transitions --save-trace
```



评估显式指定 `upright_v2`，因为评估CLI兼容默认仍是 `height_only_v1`，不会自动由checkpoint切换；报告的基础任务奖励统一为v2且不含新增四成本。输出包含metrics、初态和可选真实轨迹。稳定起身默认要求高度\>0\.65、up\_z\>0\.8连续25步；续走要求额外50步及速度/yaw误差阈值，定义见脚本CLI。仅报告成功者耗时会隐藏失败，因此同时看成功数、存活、续走与力度指标。单一分数不等于柔和性已解决。



```Bash
python scripts/play_wmac.py --checkpoint "$AMP_CHECKPOINT" \
  --mode recovery --viewer viser --num-envs 4 --seed 2027
```



浏览器访问脚本打印的本机地址；有图形桌面可将viewer改为native。无窗口短播放可用 `--viewer none --steps 250`，与显示视频慢放不同，它仍按真实20ms控制周期执行。



### 9\.7 ONNX与部署输入输出



保存checkpoint时默认同时导出该run下的 `policy.onnx`；它是Actor推理图，不包含WM、D、Critic或奖励模块。它接收时间优先384维历史并输出29维动作，网络内含其归一化。官方MLP没有内部tanh，宿主仍须按元数据执行原外层±20动作裁剪、动作affine与PD接口；20不是关节角度限值。不要把ONNX输出直接当Nm或绝对关节角。



额外奖励只是训练目标，部署不需要在线计算这些成本。输入排列、归一化、关节顺序、控制周期和PD参数仍必须按导出元数据核对。原best选择仍以成功/续走和速度排序，不代表自动挑出了最柔和的checkpoint。



## 第10章 关键实现与不可混用的约定



|文件|当前职责|
|---|---|
|`src/wmac/config.py`|profile、旧默认和显式新mode，CLI参数合法性|
|`src/wmac/amp_rewards.py`|居中tanh、只读EMA、AMP统计断点校验|
|`src/wmac/smooth_rewards.py`|无参数四成本核、解析执行器代理、常数哈希|
|`src/wmac/reward_integration.py`|真实聚合成本、历史a\_\(t−2\)、一次权重缩放|
|`src/wmac/runner.py`|采样、WM/D更新、H32、Critic目标、迁移和导出|
|`src/wmac/env.py`|pre\-reset真实转移；复用pre\-step原状态供新奖励|
|`src/wmac/state.py`|raw状态/context、Torch FK、基础可微奖励与恢复代理|
|`src/amp_features.py`|专家/真实/想象共用195\+8特征布局|
|`src/wmac/networks.py`|Actor/WM/Critic、RunningNorm、冻结参数工具|
|`src/wmac/optimizers.py`|选择性Muon\+Adam、有限性和状态协议|
|`src/wmac/reference.py`|AMP、新成本、官方来源与版本化目标协议|
|`src/wmac/checkpoints.py`|旧格式显式解释与恢复配置，不继承迁移授权|
|`src/wmac/motions.py`、`mirroring.py`|原17行走镜像、缓存来源/哈希、恢复数据保留|
|`mjlab_patch/.../observation_manager.py`|真正按time展开历史，重复查询不推进缓存|
|`tests/`|CPU 数值、协议、梯度、镜像、Replay 与迁移验证|



### 10\.1 当前AMP普通自动求导，没有隐藏的代理backward



```Python
# Actor阶段：D参数冻结，但输入仍在计算图中。
z = (score - self.robot_mean.detach()) / self.scale
style = self.coefficient * 0.5 * (1 + torch.tanh(z))
```



EMA统计和D参数被冻结，不等于把score或style detach。旧clip模式只是旧checkpoint兼容路径；不要把它与当前tanh段同时叠加。仅恢复终止hazard使用明确标注的有偏ST代理，不能把它误说成AMP也有STE。



### 10\.2 新成本的精确代码形态



```Python
# 所有历史动作均处于同一执行动作坐标；a_(t−2)复用已有观测。
older = actor_history.reshape(-1, 4, 96)[:, -2, 67:96]
request, _, command = smooth_terms.motor_effort(state, action, context)
C_second = (action - 2 * previous + older).square().sum(-1)
C_velocity = next_state[..., 42:71].square().sum(-1)
C_torque = torch.linalg.vector_norm(request, dim=-1)
C_power = torch.linalg.vector_norm((command * state[..., 42:71]).abs(), dim=-1)
```



真实RewardManager注册一个weight=\-1的聚合项并乘dt，想象 `_task_reward()` 在旧任务外减同一dt×成本。Actor与Critic用相同的新前向奖励；真实Replay的reward字段是task，不是task\+AMP，D每次按当前分数重算style。



### 10\.3 WM损失与normalizer不能擅改成“等价”表达



#### WM残差监督的实际坐标



第7章的模型target函数记作`T(s0,s1)`，它在s0的heading/body坐标中表达从s0到s1的差。输出归一化统计用真实`T(s_t,s_real_next)`更新；但物理监督误差实际是下面的代码：



```Python
def loss(self, predicted_state, target_state):
    scale = torch.maximum(self.out_norm.std(), self.error_floor)
    return (self.targets(target_state, predicted_state) / scale).square().mean(-1)
```



精确误差定义为：



```Plain Text
T(A,B) = concat(
  inverse(R_heading(quat_A)) * (pos_B-pos_A),
  rotvec(inverse(quat_A) * quat_B),
  inverse(R(quat_A)) * (linvel_B-linvel_A),
  inverse(R(quat_A)) * (angvel_B-angvel_A),
  jointpos_B-jointpos_A,
  jointvel_B-jointvel_A)
输出统计用T(real_t,real_next)，损失用T(real_next,predicted_next)。
状态误差不再减output_mean；均值只在解码网络残差时加回。
```



误差坐标锚在真实目标后继`target_state`。不能改成“网络归一化输出减去当前到真值的归一化delta”再声称逐点等价，尤其姿态和不同body frame下并不相同。尺度下限按输出顺序为：位置0\.003、rotvec0\.005、线速度0\.02、角速度0\.05、29关节角0\.003、29关节速度0\.05。



自回归窗口内的预测state接上一预测state；每步context却取真实Replay的`contexts[:,step]`，不是由模型自主rollout上下文。想象Actor阶段才由`advance_context()`更新恢复计数。监督时遇到mask无效项，会把当前预测状态替换成对应真实state并将损失mask掉，避免无效数值污染后续有效段。mask分母为有效转移数，不是含padding的固定B×K。



#### 五个normalizer对象按四类职责分离



|统计器|输入|当前关键语义|
|---|---|---|
|Actor obs\_norm|384维历史|RunningNorm，默认eps=0\.01，超过1e8计数后停更|
|Critic obs\_norm|136维特权特征|独立统计，默认eps=0\.01，1e8停更|
|WM in\_norm/out\_norm|135维输入/70维真实残差|输入eps=0\.01；输出eps=1e\-4；默认2e8停更|
|AMP norm|单帧203维|sqrt\(var\+1e\-4\)，clip±10，初始count1e\-4，无最大count|



这里前三类的常规std采用`sqrt(var)+eps`；AMP通过variance\_epsilon切换为`sqrt(var+eps)`。新AMP设置在每次成功D更新之后用真实策略两帧及专家两帧更新统计，Actor想象只读，不用预测状态更新这些统计。checkpoint必须保存mean、var、count与语义配置，不能只存Linear权重。





新增加的robot/expert分数EMA不是又一套特征归一化，也不改变AMP norm的方差定义。解析电机常数是静态buffer，不对随机初始化的learned cost head做拟合。



### 10\.4 最容易破坏梯度或重现性的改动



1. 用no\_grad包住整个想象，或detach预测状态/score/style：Actor路径导数被截断。

2. 把WM/D/Critic参数放进Actor优化器：策略更新不再是本方法。

3. Critic拟合输入或lambda目标未detach：价值损失泄漏至Actor/WM。

4. 想象中用预测样本更新normalizer/EMA：目标和尺度随策略图漂移。

5. 二阶动作把历史detach或错取最新帧：跨动作梯度或初始a\_\(t−2\)错误。

6. 用四元数4维当6D旋转输入，或者把真实关节q当Actor相对q：维数虽可能拼对，语义已变。

7. 把自定义电机请求当真实接触力/全部子步force，或用硬件名义50Nm替代实际转速曲线：奖励与实际执行不一致。

8. 用概率累加代替v2整数坏姿态时钟：改变前向恢复定义，Actor还可能利用时钟导数。

9. 把D的GP从expert\.sum改mean，或改成插值WGAN penalty却不改协议：损失尺度和方法都变。

10. 漏掉模式flags、装错RSL fork、没装history补丁，或只加载同维度权重不核对归一化/顺序/协议。

    

## 第11章 复现验收与训练结果的解释



### 11\.1 验收范围



|验收|能确认什么|不能替代什么|
|---|---|---|
|载荷、补丁和 298 个文件的哈希|当前源码和固定资产完整重建|依赖已安装、GPU兼容|
|独立环境版本和导入路径|公开依赖、正确本地模块、history补丁|数值完全一致的长训练|
|CPU 测试|协议、损失、梯度、历史、迁移和数学自控|真实起身成功率|
|8环境小网络GPU烟测|仿真→真实新奖励→拟合→Actor/Critic→保存/ONNX全链路|默认大网络性能、40万轮效果|
|4096完整网络迁移短测|已有大权重可在新目标启动更新|长期柔和性改善|
|实际起身/行走、全物理子步qacc和力矩|指定权重在指定条件下的行为|全部训练种子和所有初态|



在作者验证中，解析请求力矩/限幅与96000个真实控制步首子步记录逐元素一致；现场reset前后动作历史匹配，奖励率重构误差低于2e−7。生产形状H32、B4096、TF32的奖励/FK/D/输入反传组件从31\.331ms到31\.578ms，约增加0\.247ms。它不是整轮训练增幅，也不能用它推算默认GPU显存或预计多久学会起身。烟测权重只标为实现检查，不能冒充已证明更柔和策略。



本版保持世界模型的状态表示、网络与目标，移植了采样、日志和 Critic 的等价执行快路径。正式配置的 D 样本预算是显式变更。世界模型内的可微梯度仍需要用同一真实状态和动作的物理对照检验。



### 11\.2 本次从公开源重建的实际验证



本版验证直接从 Markdown 提取驱动和附录，在新的本地目录用公开 GitHub 固定提交重建。298 个文件与 18 个原始动作文件通过校验，重复执行验证了补丁的幂等性，5 项交付测试和 369 项 CPU 测试通过。附录 A 的配置与正在运行的 4096 环境训练相比，除结果路径外的 122 个配置项一致。



另用公开源、全新 Miniforge 和 Python 3\.11\.17 环境验证了 136 项依赖、`pip check`、history 补丁、本地 checkpoint 写入与重新加载，以及 35 个动作片段生成。RTX 6000 Ada 上，默认 `learned_v1` 和可选完整几何模式分别完成 8 环境、3 轮的小网络 GPU 验证。两个模式的 Actor 和 Critic 更新均应用成功，分别保存 v12、v13 的 `model_3.pt`，导出的 `policy.onnx` 也通过校验。几何烟测额外使用了正式的 D 批量 2×25600。小规模验证用于验收链路，40 万轮的策略表现由正式训练评估。



### 11\.3 应读取哪些字段



`config.json` 核对 AMP 映射、足滑模式、4096、TF32、v3 四系数、H32、UTD 与 D 2×25600；`source.json`记录来源和实验源码快照；`motion_dataset.json`核对17原行走＋17镜像＋1原恢复；`metrics.csv`和TensorBoard观察阶段耗时、model/validation phase误差、actor\_active、更新拒绝、Critic、AMP robot/expert EMA和style零斜率。新四成本在RewardManager内聚合为wmac\_smooth\_penalty，固定轨迹的分项量级可用离线校准脚本分析。



相同固定初态的两个策略回放仍可能因GPU接触数值出现分岔，须各做重复回放。统计先按初态配对，再汇总；不要把百万个子步当百万次独立试验。恢复耗时、成功数、续走、qacc/torque尾部和动作变化共同报告。



### 11\.4 世界模型近似与诊断边界



- WM\+contact预测、末端Critic、生存hazard是学习/优化代理；默认没有新增真实回报Critic锚点。

- 真实末5ms qacc与20ms差分、原body缓存时序、预测接触代价仍不同；没有偷偷改模型以掩盖差异。

- tanh远端、AMP norm裁剪、动作裁剪、正向朝上gate和价值上界都可能局部无梯度；不能保证恢复每个状态都有有效修正方向。

- 当前style判别器不是独立的人类自然度真值。共同恢复阶段与包含后续行走的固定时间窗排序可能不同，不按单个D均分擅判某条策略更自然。

- 新四成本系数经量纲校准但未证明长期最优。未把HoST/HumanUP的额外控制课程、训练阶段或网络合并进来。

- 实验改变任何mode、系数、数据分布或动作控制，都应另开run，并核对checkpoint协议。修改源码后原清单失败是预期，不能继续称作本文原快照。

    

## 第12章 飞书导入、导出与故障处理



### 12\.1 在飞书阅读与保留复现脚本



在飞书云盘创建或打开文件夹，从该位置选择“上传 → 导入为在线文档”并选中 `.md`。也可以按章节复制正文。公式与流程图用等宽文本显示，附录按短代码块拆分，手册整份控制在 50 万字符以内。用于复现时优先保存原始 `.md`。如从飞书取回，先导出完整 Markdown，再用第 3\.1 节驱动读取该文件。 [飞书文件夹导入说明](https://www.feishu.cn/hc/zh-CN/articles/360033241654-%E4%B8%8A%E4%BC%A0%E6%88%96%E5%AF%BC%E5%85%A5%E6%9C%AC%E5%9C%B0%E6%96%87%E4%BB%B6%E5%92%8C%E6%96%87%E4%BB%B6%E5%A4%B9) · [文档字数与区块上限](https://www.feishu.cn/hc/en-US/articles/360049067974-docs-limits-faqs)



如果从飞书再导出Markdown带去另一台电脑，不依赖图片或附件链接来携带源码；附录全部是正文ASCII文本。导出后先执行 `--source-only`：载荷SHA和源码校验通过才算完整。飞书客户端或复制过程若截断/改写附录，脚本会明确失败，不会悄悄以部分代码继续。



不要删除附录后只保留讲解版再要求脚本重建，也不要把附录复制为带额外缩进/省略号的文本。若只想执行，可携带同一交付的独立自包含脚本，仍无需原电脑其他文件。[飞书Markdown说明](https://www.feishu.cn/hc/zh-CN/articles/118804574721-%E5%9C%A8%E6%96%87%E6%A1%A3%E4%B8%AD%E4%BD%BF%E7%94%A8-markdown)、[导出Markdown说明](https://www.feishu.cn/content/article/7644456827538820052)。



### 12\.2 常见错误与确定处理



|症状|检查与处理|
|---|---|
|No embedded payload / checksum mismatch|保存完整本文，保留附录；重新导出未截断文件，不绕过SHA|
|已有workspace源码不匹配|使用新的专用workspace；脚本不覆盖研究改动|
|GitHub/包源连接失败|先恢复公开网络；允许同一未损坏workspace重试，不换私人未固定源码|
|找不到Conda/激活到base|先source生成的activate\_local\.sh；不要依赖作者yao\-rl|
|history\_ordering参数报错|使用独立环境bootstrap安装固定patch，再做environment校验|
|导入了另一份rsl\_rl/src|核对PYTHONPATH最前为重建项目，使用activate\_local\.sh|
|未启用新AMP/新奖励|从第 9 章的 v3 启动器命令开始，在 config\.json 中核对 AMP 与平滑模式|
|CUDA/EGL/Warp失败|核对驱动和系统EGL/GL/glfw，再跑小规模GPU烟测|
|首轮很慢|第一次torch\.compile不代表稳态速度；保留首次编译与正常更新两类时间|
|显存不足|不自动篡改默认配置；先用脚本小测验证栈，再明确另开缩小规模实验|
|有分数但Actor不更新|查看WM/D更新是否有限、EMA是否已初始化；不将所有兼容阈值当默认门控|
|resume拒绝|目标/Actor/数据/电机常数不符，或漏显式迁移；从同一合法源分叉实验|
|历史诊断拒绝smooth模式|属于防止漏算新目标的保护；标准物理评估可用，勿把旧探针标量当新目标|



## 第13章 原始论文、开源代码和许可证



- [AMP论文](https://arxiv.org/abs/2104.02180)：对抗式运动先验，不是逐帧目标跟踪。

- [SuperMimic论文v2](https://arxiv.org/abs/2610.28480v2)、[固定官方源码](https://github.com/HybridRobotics/supermimic/tree/abc124c1bd5976341034668af3d12c087f2f1f55)：可微世界模型、短想象BPTT、价值bootstrap和训练效率；论文性能不是本项目测得的成绩。

- [原AMP项目固定基线](https://github.com/ccrpRepo/AMP_mjlab/tree/6c7a2947fccc973e4af8e6d90e550400f1b6fcfc)：真实G1任务和资产。

- [InstantAMP固定style分支](https://github.com/RoboticsResearchHubVN/InstantAMP/blob/9fc8031055a18d58e84d9de90639660e7aa6b491/instantamp/amp.py#L106-L119)：本项目只采用robot EMA居中正尺度映射，保留LSGAN。

- [HoST二阶动作项](https://github.com/InternRobotics/HoST/blob/70bb580949a336a920833700e4b5dc3bf7fe87ce/legged_gym/legged_gym/envs/base/host_ground.py#L1064-L1074)：不导入动作幅度课程。

- [HumanUP速度、力矩与功率](https://github.com/RunpeiDong/humanup/blob/7516e0f27e6f4d1e7365cf64ea577a78247bd8cb/simulation/legged_gym/legged_gym/envs/g1track/g1waist_track.py#L821-L865)：请求力矩/功率代理为当前BPTT所作适配，不能称作真实子步监督。

    

`THIRD_PARTY_NOTICES.md` 和 `licenses/` 随完整源码提供。本文没有要求把这些开源项目全部装进运行环境；必要公式、优化器或镜像实现已经随固定补丁携带。



## 附录A 当前完整目标配置



这是纯配置目标；一键脚本再用本机workspace覆盖run\_name/log\_root等输出参数。包含 v3 系数、生产足滑模式与判别器预算。



```JSON
{
  "profile": "supermimic_official",
  "reference_alignment": "official_details_v1",
  "optimizer_algorithm": "muon_adam",
  "mirror_walk_motions": true,
  "motion_cache_root": ".wmac_cache/motions/g1_mirrored",
  "preset": "full",
  "seed": 42,
  "task": "Unitree-G1-AMP-Flat-Aligned",
  "amp_feature_schema": "grounded_v2",
  "num_envs": 4096,
  "iterations": 400000,
  "collect_steps": 0,
  "utd": 0.4,
  "warmup_steps": 8,
  "model_warmup_iterations": 0,
  "critic_warmup_iterations": 0,
  "replay_steps": 1000,
  "wm_window": 7,
  "wm_updates": 10,
  "wm_batch_size": 2048,
  "imagination_horizon": 32,
  "actor_batch_size": 4096,
  "actor_updates": 1,
  "critic_epochs": 5,
  "critic_minibatches": 4,
  "discriminator_updates": 2,
  "discriminator_batch_size": 25600,
  "actor_lr": 0.0002,
  "actor_betas": [
    0.9,
    0.95
  ],
  "policy_output_init_scale": 1.0,
  "policy_output_zero_bias": false,
  "actor_grad_value_clip": 0.1,
  "critic_lr": 0.0002,
  "world_model_lr": 0.0002,
  "discriminator_lr": 0.001,
  "discriminator_optimizer": "adam",
  "discriminator_trunk_weight_decay": 0.001,
  "discriminator_head_weight_decay": 0.1,
  "observation_normalizer_until": 100000000.0,
  "amp_normalizer_epsilon": 0.0001,
  "amp_normalizer_variance_epsilon": true,
  "amp_normalizer_clip": 10.0,
  "amp_normalizer_initial_count": 0.0001,
  "amp_normalizer_until": null,
  "amp_normalizer_per_update": true,
  "lr_decay_iterations": 250000,
  "lr_final_fraction": 0.03,
  "gamma": 0.99,
  "gae_lambda": 0.95,
  "exploration_noise": 0.1,
  "action_limit": 20.0,
  "max_grad_norm": 1.0,
  "task_weight": 0.75,
  "style_weight": 0.25,
  "amp_reward_coef": 0.1,
  "amp_reward_mapping": "instantamp_centered_tanh_v1",
  "amp_score_ema_decay": 0.99,
  "amp_score_scale": 1.0,
  "allow_amp_reward_migration": false,
  "smooth_reward_mode": "host_humanup_v1",
  "smooth_action_second_coef": 0.0175,
  "smooth_joint_velocity_coef": 0.000625,
  "smooth_torque_coef": 0.000625,
  "smooth_power_coef": 3.125e-05,
  "allow_smooth_reward_migration": false,
  "allow_smooth_reward_retune": false,
  "imagined_recovery_foot_slip": "learned_v1",
  "recovery_foot_slip_contact_height_m": null,
  "recovery_foot_slip_softness_m": null,
  "allow_imagined_foot_slip_migration": false,
  "gradient_penalty": 10.0,
  "direct_style_grad": true,
  "critic_bootstrap": true,
  "survival_grad": true,
  "recovery_counter_semantics": "hard_forward_v2",
  "recovery_height_reward": "upright_v2",
  "allow_recovery_semantics_migration": false,
  "recovery_fraction": 0.5,
  "expert_recovery_fraction": 0.5,
  "max_model_loss_for_actor": 20.0,
  "actor_trust_coef": 0.0,
  "real_critic_coef": 0.0,
  "real_critic_batch_size": 2048,
  "max_action_shift": 0.25,
  "replay_device": "cuda:0",
  "device": "cuda:0",
  "precision": "tf32",
  "compile": true,
  "compile_mode": "default",
  "compile_fullgraph": false,
  "reward_chunk_size": 32768,
  "actor_microbatch_size": 0,
  "benchmark_sync": false,
  "validation_env_fraction": 0.1,
  "validation_interval": 10,
  "validation_batch_size": 256,
  "max_validation_model_error": 20.0,
  "max_validation_contact_error": 2.0,
  "save_interval": 400,
  "log_interval": 4,
  "evaluation_interval": 2000,
  "evaluation_steps": 250,
  "evaluation_envs": 32,
  "run_name": "local_4096_ppo_amp_budget",
  "log_root": "logs/wmac",
  "resume": null,
  "init_policy": null,
  "init_discriminator": true,
  "checkpoint_replay": false,
  "model_hidden": [
    2048,
    2048,
    2048
  ],
  "actor_type": "mlp",
  "policy_hidden": [
    2048,
    1024,
    512,
    256
  ],
  "moe_variant": "hiking",
  "moe_expert_hidden": [
    256,
    128,
    64
  ],
  "moe_num_experts": 4,
  "moe_top_k": 4,
  "moe_gate_hidden": [
    256
  ],
  "moe_temperature": 1.0,
  "moe_load_balance_coef": 0.0,
  "moe_router_z_coef": 0.0,
  "moe_init_noise_std": 0.0,
  "critic_hidden": [
    256,
    128
  ],
  "discriminator_hidden": [
    1024,
    512,
    256
  ]
}
```



## 附录B 自包含源码载荷



下列数据包含 XZ 压缩的迁移补丁、源码清单、环境锁、安装与核验脚本及目标配置。请保留所有区块、BEGIN/END 标记和 Base64 内容。驱动合并数据并核对整个载荷的 SHA256，再应用补丁。



载荷格式：`amp-wmac-flat-doc-payload-v3`；源码文件数：`298`；payload SHA256：`c02add523776b414ae4a98ee5ca2b6bc139ab48f5721e2d12940a739369e01f3`；patch SHA256：`b914633f2270891813c54c7844d147c64a3b603b34c8a227853ee348380c366f`。



### B\.1 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.2 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.3 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.4 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.5 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.6 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.7 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.8 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.9 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.10 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.11 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.12 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.13 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.14 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.15 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.16 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.17 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.18 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.19 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.20 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.21 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.22 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.23 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.24 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.25 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.26 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
zHqMeT5899BacoE65uMtOPA7K7ozd4HuFmLxy70M5WPCVwHoRkg8o2ovXP4u7JmyHt5Hk5yLpSpAzpGRf/k4yLK6f2ZZsQpsn4NCJn5dgpdo/p9ZAId8CsIbNEdXQ4ySaieNvv6Z19nIsb82A+YBO0bqkRiFMCzQ8ZExtoG8HQ/IMcxsLZDJ5KT1DrWy/Q4b
kkFE5m9tjeV96RYC8NoGva2XSpX8j/+d43qlWx1y4P2s6bbqIMjW0OpcIoN6eW4wYKHJLFFnaIscDWm8tQ4qovkApQYS+US+XY7h8CzxS8jP2XwD6qsQFOpborcL33rkT4kKo6bu/LYTD4EOCHk46XUTka8Nvz82+p3a7xn7vXSi+Squjv08RP+dMIZhpFi9
i6ouww+vmlie/YqFuewjPCp1my+D24zkWXWozEuoV9bIqNevLZ9atkLA62zuAQiFlFNL0+bxRrkD0WA7thLQC5yyju+RSZD7orqFENUg8BZS+kl2nH/o3sR9gQM0qSv3Pzcrai//i5X/gjZt4TDVfazc8j5XJp17u2qmIJKewtig9WeCdxD7FRseGZh5eDJ3
HlskQFULupETE+ets/XrNUNu8iP6Mc7mHv9Bwvxryk5CDP4MVtfQ+uOyw1Ep0VT4NpeOJNC0sBoPs+iAX3tXNDhh+MV0GtkoMelluLjOUwr3GazcLQxZm/aCZb8o+fbx0U+4AZmlJ5E9CRC4Lq53+FTvJ0Mbi/a6RnpCFwutmgjayF6nGzb4zcsY4TNjsoJu
IlSZqDwepv5NZtztyQFOVZ2gMHIepJ3gN3tvlLnaLRgOhYToON5wv+XF4bUaCntSWo2DYmtgxZwg7zNhvV9DErYLcyQq0qdiPpOqVAEI4NCSVmOCxdERFOF3YQ43U9WvIjR5Gq60p2r/yNRBfnSxC0KEDhn16UhUIZ3QIcWPis344qYv/itX5rfQ9RYuPKKB
7JcJPKOcKNS7ccU3JwUh630E//r8ip0ZPsiZ3W3vwxlWe5HCVRUuSyynSJ0qzLQRQ3lpafDTFJXKvHWzj8K71jNZgxlrbcY7npUv+LYU+kU8XGSvDoKvHWPA1LnVaHpTuGa84913ArdvVyD/RmXxE4IrP9Y0cDgTHLnDXkLQhqgFucF0K7nvJyyJwkPd37NH
PjZdKlPTnAHhSmbsQfSPMHY6e3RRTtcZIC/8cXUH9dcLHqPQBH9wpjerJgV2LRjSNyN9sNz+55KXMYqdmBGFL3SR0i1VCcvOoVZKcT05PhRfk8ktCtevpEt0rBvkAJIIwLj2FR7taf7L1QAyOQLnkG154MyRG1zI6e1XuT5avYZUpFctsPWfUUwxXtvuIluf
4Whieit3IJy+1umXyhe0Kv2KPi979rhq5emPaeSqj4MZxpHAkS3GvMZgrgSNCsBBwPYtezGFgjAaBk/os1IORjgwl2EHbW4o1JWKpbbR5Pjo5n2w8WzqouNa1nA2fqnMcnJVtXxz87DPzMii0mA1xYcgWAMtxgDOfjEXQ2njbQ1xFOdrkLW78QhWA8xvxrQL
cRE2crIscdRcaQNjgKFLt8HuXcY2F3SciFUtidpMwH0m3nXMUbY31n5qoRC9sBYr1AqPNOL6Wlj6SdmRUHyklkEXeSMKBTstL+6YQZfLXVaE3DpKcqiRUZWcy7cc7TRfShOYW1PLulJi2iarHLuAD8FoJMCzthvE2L3l8IEmWWUClIXywMlDUxz9LPHw8yDR
766CC9DtELmWeuD7+nM8hleO4KB9Frx00oG4NXkFOyYOE4GKknHdzNEMlpTlT1OADrywoumkyjn5eNsPGyGvSKfqqG4No1CVS/ZmzhAyrHWV7uK6RZqjijfAk2I7pDyHFfhvKzeezjSJ0Ytl/QZ19FA3qkO1eDVt00Po5NQnFJolE5L2Mjgptj7HaEeTkjqC
XMH5QnKRy0l9iBL3voJh1T2TpoOnRknQeEczQn/jlyGJrUjcvCD0DWdTTCHbFIlCyaF5NgnOZRVCsm7hr7SjJhwUcy2inkW5eIHKyxaxREN/IDhXdzzYMI1+lCLsR4j20dA2Ck5r5nRLlZXF+p2yVQiV/VHfm5tZ1sH5+BMd2VIyXoA+iWKs24vTWSgQhlGK
0Rpiq8/b7X89zy27x+cLgYkOqVTPwv1JpN3/Os0y1Auu3ag99y4oNB24FVi9C6U1Zl+6JnajjnkGxm4oNMPHfeeKpfj+fd3K6uXSTtqSCdt05HxBOtotPAjR0ZPu0O7dsasUZwFBxxYk7gzcbrDnkiZ0uggVifTD6hUXJsHFS4E1hLaanTvMH2UCxX1YMGyr
IrQH7NJ5JSyYlzzuF350vMXEW+ToyI56Y30A+aLASSrD+e/Aua23csClo+ukCdqcaA/hYgYXoy6DRRMXOai0WqUY7LIdd1t2Grz/PZ6NfQy9Y6d64Kge/9GkjQ4fXFc0HmyYD7dbSp6qPlPR6tB9VuEnI0itl9YCHv5Gwanr9iSKqiHopUch7hoaQ8gmsQr3
pjUL1GHlftmmRyQpFdzNa7vlKwYj/PmDpWUIbvcD9GVw7Wk+++FkB6pFYXDV6lxceJDoQc/1/nHZ7kssQAs7cpdiiQess5QzZ5hyeUhesouj3aOcbRSQo6mn2dZ064HylQz2YvEc1LTQVCJXFm4oGbiDFa1554GXuoicxiGDoZPOxZgFLU84p/+6uqeS3x0f
X6tGydTv490se6Cs8Q/kdGe0vpheyhbMAR6tEIazUjxEhA8+dYUknvN5bwQP3iZr4YVU/nzhmuvhBi8jbWALOBKkymlDFum374kDAYddFfdxTLXxeesrnqBVvYh1S2JsfmDujK5KBR1znDETEGpDEDVNdx3YkH7zPyfb3+bGbs8kkbweQY2tHYAkBLWo9vXk
2NL8eg0wF+h1B6W7NOib4Qmt1bgIleBDd0cIxPLUewwHZJXIBM5IgD1dZll7tO4FfgwZ6Ef+0vDpIhmg59w1WjjDPGVC3l4nyyBWQm3oOhbLch8WG0vGHIuZK1Zpl7ElZxA+zgCn5ZfQjPYXXZWN/9bEfHKZaFB4WaB6v4BRDGvY8W0G4XNk6Y8c3g0kagQh
SCCRlG518I+iOmHCbUWeyJAKgEolDmnZgoo1Ms0t6zq7xlwTdLDL+8+av9xtihKH/yIhhpXUsW/8qUjQ9eClhl5tBytz9mCXqx8ikDN3orIjVTnPPvpQwEFD/sCTh3Myvle0vTNymGMmd9DYAsKuOKHjcePPhX/z9a7yRMsE7qb93ICzV1Af7zUPbq2ztjh1
4u3RVrsBJbR4HB1AXcvDs9f+wpn78/OtEi92aeefUriDWU+YaWEYuqQHRZXU/744dMD2ap7bHigGBTHn32S9YGANJOFfQImNPoh1HJe7wH+krbN5FmOfZPY8ftED4F06t6ksS/WJ1AfbZGgW1MJS91CWFCTrI/YWWychDgdT1gVM/Ybm1JfRlIiZRQL+3Jj2
/wrQVts4gSnpvBsxScTlPAT8T35kyY54HBjQ+QbQ4Uu/HU1zOSO1dr3ekTHiS/7zELY0QL9mvUhT01PuqPkkS/KEM2O3L+Wxxl85ZlwJZZ0I7CTXfnoRqPrPgO/32hMEpJrCwWbcWcfpH8c4WOWDvYcXa2rU/uEfXgRBXFnFfgjhwZjWD1fHciP2yWruPbNd
687shldIAI3gkV+SniY8NESmfY7OCt+7+aufxxmv7neNMm53GM3xnLh1T35xzDEM2WvRAMm4Pup3YrQey1CU9b5IgEgGz/RegRVBZ9clXQT589Fkz/LnnXVorrzAh53ol/1hyMKngPf399+Yc/i2bwoxFg7O4H1+WLLB46HNbUzb/9Ub2BdrTthQPnAKtR8J
SFnJgryhYvgCmkDRivORYXZJM2mAg3NKspR+ESuiu9Nj/QNirttasm0hZjuL4PjD66TZRb8QfGpQCZtcYxjHgZscV1AmuxiiYT6RkbgbTI7m5gS6Pxx81i/6IbARDXcMQp0Lboitjoa8WHAy8EDBVoBB8reyer8w3wTziTcOrxdmPcP++hs4muysA6t85d06
R07iARt/5xcFpyjXQ7XIBNMwT7HlHPSggctwZK8t+HOU0q5Lsat7jrgmErn+uFF6iYWczwCLxjpOVdlmjJaA439TlvLROWdfa3Z6I8/4GbRiHe4FAJPb7V8m2Oy00HgNfUvU4ISjRDU8vD8W+aVngayhjTDBBDdniQwOVNb2shOe8S/Es6ZevcTpchhbxTQk
JBlUZIFBSUYBIooGG7Lj2HqTyd5Q/B5MY9vSrOK3Shln355Dg3vVvswivBP2PoXh5UY4RKr50UXT89yuWcIpl+6LcdWSkcZWdBWDOe7BxOShIRHaNh/WiRDXc6qSRK0aPR1YXLSWjsz1AIjXxix4DjHPuUn3jvz2Izux3FohmImL4LvjCKZlXEPeTS0XGq+t
Fzm2tfS7oGVPCHOsQcyzhODRz6YJ0ilCW9KV1gX5iVFGDW3PCzqN+1zEiCDTXap3sg/QcEM3pOnmdC5SrTnH5xi0RoLY6tUpORNXgzKnv52/bo+hgZqs9TO+52PbtrXecoaoQRXXWmUaW/GRNFfXWiWd/ijSMCPuSoWGLi5rW5dRFo2EPrUu8hOEIwRAuUra
OipK2oxWrlOAysmnJoBpOv07jevyFJRuYoyDILnHxrOjS71nfRJrPNrlYbLZvUM1i/iTC9q7zQ32fXi6/kXI3NQ6SUygh4U4uROgtqD5d5ORA7Pg3ISk3gX4Jmu0LRSqcM3Y2hikQ3fyMuWz6I42ZGpZMtlBNplV5F0QpXFLVXXrqyoKFYHNC5Q9129isQ8g
CvSrtis7CjK6ASZIZGPocuGPiUylUM+99ZcPnAHGDyi+k8o6+wuUJAwLksiF4gzQPAY2ysDg+w6twM3J3lZG9Dv9Pd9dhp8ies+yQfXdKyE/jk+ZcMiVNCqY9Cqk4BvXq6B6kaxe7TvaixQGGNASKwkV4Z+lauhXRXXSsZJy8cCVu9UU4C2EVLSVdNcld3Of
QsOUH8Rvc4WcthhN2Be9DBXlLuHLknOu3zJU3ILOB6/tJUOd0HMhprNWGGiBTFaI4Wl8u8NMM7CEMk+T6AFKi9IBmRbdgvIpQh07xOO81kWRcpexgXlaVQ5is7Y+ueCayNSXra5mBz0xyruyL732c1TnKZ8KSOm/1kjxqk73PF5hJa71eFBoeHtwYdLHK3tC
8XSOyycfNpGA11488BpiHRyy6wm0S0oGQuLOiUm3z4GXPqC0dyRSxdb73yzWIZY/hn9RjqS+suApkRuFjTdsLeQjK2vvlesHt35vvA3pN7VGAH281GLPxs9di+sTOcIynOF0kI6QD5xf4ViHqaytin0zGrEwkZuDH9iwscfyCRHZhdtDvrL3xpCDd3zwlhZ1
pQVS2CkrYptEoqtVg2xbqb2Ay+xSslen4QRduZfrchJWrErDGaipDvbI1TyO49Wfbvx5qLa0Yk8DoJaf/lniOLRsYDOgcycObeHxkNe1eGUJF7KSUKiFTuvB0GABC6OtQn8Eg5uj4YsEXalA3BlcP6ZWlU6yCyHMscG3FFvQFhj0F79S2TaU++tC8H4D8jRm
C5sD0zW8lkEg8lcdaZ0GDV6swGsJ6rtFbuu2PYNAVM9Jbog8Gw3LNI6cIFK4rDFQyx3t5OJmR4Xuf/5Vk9samnW3dFrnX2codO+IH5vrT/bgaBj4Bx9j8jgH+d0XVbL3nhxkdCkutRhnKxNWrSVzRQv/x3Ixr9ZSVBeWRC5tm/OE2slR8iRUzw7zh1io5AIE
Ms72i6kIQgMR3T2Yn3xGAjCmAVuzwfKFYwGaCUIh8MOSBYAfg7Mje2XPM3j5DIDFNTaAMuWo6QzcHS49xAaozl0xeAKAPN3/lCe7f9XHVPE2ujymKAWEdf9L7hpIa4w+VWD035Oad1utAwLBeh2QXcn0k3Eu/rrCVnXDs7XurF4kg6ZMohCkmPUCTDR2OEED
ZsvHQGxmA+DDD8blfMM9NYujXZfOzircKBbYn6VM/ZgjAJhy6hscXNS5F4R06DOGFNe52b6iUGjX2espc2Fj+xGCoz/jtW3LGIf3RkFPr2F0aeTtMT1Fu+1CH4cMG7l0VZ9qjRgpglJWL531V3E+Ioa5JPC7rgWxjQW0SY4dtZSDoxZhYlx9JWqVJX3372Ym
yatPaXxQ/Cd9ENWvjTAd7cxd1XZRV/MIYvPWEzKykW+4uz+jRUCLhz+Spv138GaK+lgg6VQHt0oQ3nqOnoUkFUkvkua+sSWjZiLVgjrd/Ma5nH2porL3BshQCCVbeBHUYhG+AcHYwlpi6fgcuDAocXkOnNoEZrlnjYIOozKsroEBLHhBjea3q9YHdzovwtkr
I4diD+ouU2XY8I1Sm1kBf+gD9g3ZKufDqaQjOScgZdTQQbZRQ3g5GyI3Mm+QpdsMkB6+eHYp2ioQuH8fkeJqkYVIgqZ4/KyFN8DljN7xDd7rDagM0KOK8eqLpSprlINKj8M9at9FkLynl5pWphdNUaEXa0JoVt3jOW0mhgExszkX98aRVzX90P95p5vsS5qm
R7o674izk8tmcLB4ylxj/usb0dQmKmVm9KtonrTSoGTN7pWNE5DHqT14ohhQj7mlZgMWaAzHMtSxFvW+yL8ZnHQcI1CyuqANGlrXDQIYzPNI0LUBnSiw7ofwZPJhO/sDIGpHQnnZcZOSlaB2rE2nW/7OJiyfibTOcoXLvDwSQg0FN6GkT4Q1AQoh9+MLbmAh
xB60aeb1QNn95X9s8UxDRwm4cp1Iim9ffl5q+OIlIHi6VDa6grMKlBkWsiLisi3vxyhNzwkDg7DZt+fxLoVmNUeGGz9R9jal40nfZU8Dn/BKlVRsPLbLf7dClVpW39y9NR4ReyfBvQqNCpOXfvsoxNb0ieV8kKpxaQtlXsPIH9uzXBLlq8iUSLrt3h3HvHse
4eJodByWi0HCxajtTYTsmE/MtR3VDtGTTH3BNmomIEyUSkKMte/j1E6Th7wkTWUUk3fLKwNxvIs52xGo5kPn5HDdkyPoZWmHvlRHb9LK/ALN4uEcpQ7lgiRO/rZ6nQ/SL/IjecliUNnwtj2qTVN6JEZ/1TXv58FdPj51uWufpiy0UpJbKR1JPvLIDLnV77TZ
3qRK2xJmFmR2tGVGXqgE0CQ8INp93waCZ62W7VicP4LF4FKkY/cyyLcvo5gD/VevhYYdAn+63iaILa44MXQlCfbJI6PPFxgp+rVIqFzgP/fWxMpyT7/qyNHWoQFczql1PRxvpqfSFKSXH2c5I28KajISsHebTkuY+iE7xNLzCGw5nakP9cegPLWibl05h/dH
8buNKwXzMNNm9ceAtW2IcER3vRxM5vMnXQC+PO/tAxQ/Y9xlES/lbX1bZZK28spKZ/Hj2fI6M57EP7DFThFxdQ6RTp6dfQ0FQHccznHev7oHB6wlBo69i4ifLz1ZrvGiPyDA5v/WhoU6m2qXJANPSeQxnF8F+l8fV2cNJJgmmPPuJTyAtDSqxQkcertThPhX
qfQMi/WraotvEU0YBQn5Dpu17JJp+Zhv61P+5YomuLjRF+tDa9reA3tPu1/bhDzqEV2E6g7whBnk7wsIrLaLeekXftbT2o2no5SyQKY+nb052+EGD1Npn/w5zLzYiKDIfq7ysQL5w6Eon/dujnyhO6NpOdSobM+At8jplA9cWE0LKs0LMZpepnNIboq5C/EF
skWA80BbjUUyf4HEBeA5qsNWdiln5EcJsTUWdVE1N1I7SCzoize3hTeMZqjwtHe4yQH4/ZWYIPwVuZn8yeNNw8lQ3kGHbS8bP4I03cirOvQ0QzuaCQNZufEZTrNOqMk9FNEhN6NBO4zpWRWfd5o+IR2zpaadvxxPRzfh9jahnCJZnTQEN5B3CzgpKYeZflCP
krU32xrR1ZZFuoc2uBrQZuwVu9gHnfnastCiAZ8WiM3Wxgm36cgDk+EwlzZXKC8PqqY5bRWnAIfFznBXIEvkafy3GSxYGyqHa6oT+NycFOlS4R95K6xehffI3yKMQuQtsLGtWwANaxd0goc3HKeO+VlsDQxnUaL5talsSrknWEieLtnoSsVT7bwHoqSsToXz
SYKu3xt4NzJ6Fg1DgafFnQ8IshQc8g2x7x07YdAHWxT14pf4CzYOKO3YmPS6WMvfbyfFXtThkCJ7geVOoGzSjlpHU0eKdJEGrEZoZEloaFapYuDHAa+VLr8KQaxo6P64JeCnbMONbT3txKmFDcFXK1+2m3h+nRcx5dF9ZxQnrIRC4JNXPx6lutmzXgGu/xxH
2buKM6TLsu4RbFe4QdWOYAUlKmmkG/X9CfPc37UnVPUk/jCpqLQUqWaIypgbcZWoeLyIkCOBVc9uymU2LajzpHYHLmWBBO+UoHkuoF9DA0CaS185+kCHIblSy1ZOPeqUBMRBQFd6JCx16mu+z+vNrm0C4yB3MpcMnKrPLZnsqug5oSNkodIkn/9npIqOg/PT
NS90GF20EkTMdYIDmxtG1UDd/0Vg+ZawsEvxHlWTokWEcC5tZetVTmsY9oT3TQ7vIlVWXdusbtQeBJ2481SAZi/VwBLLrOymhKIUMf4wEdbsmWuOwcB7EEtV2hDL0qy2CyY8OVmxqfzlp4+1pyeh9xB/m0BUix7+e9kKQCiWsiTtiCPO35AXRvwvrL8ZU8RB
NK5L97jALLr9eSbmz/qbnjtG8bp6rJXcK8VBBuxxPe2cfBwQLI891dZWOEdBdu2ErNRAZ8NQtDRnPSCb1Wy1Fa/BpEGNYq1W0N28RvU+ZuLKv+o9g3a2MFh2Hi7qyfEf847rszLGF4kq4TuvxFTkyQNqlv3Dx7Z1JstG5P4WL4MfRaU2oliQIdXkIj/m4ozk
pl9O5LkyocH3XksQz94uSy2G7tn4E3p/zHQK1FijHHDEAuOTLdGHqHzSNTXBqwGqVz5g5Ej9/wwR27fv335T8rFJ9XQMI5MK81SVKiUIk4loHYCkTOAVlonD4zwtup2/+xzB64r9ZZVJr6NoU0B5qeieJkN34Sd5Q1oRua5+OB7/3uxeeGHpa+LLUHN+bnxQ
7RxhKJI5cg3rOnaWqI4WOFInQ5dXdj3/fAOFvVfX39ylFMmCEDlUGL+1YcbXQBs4K09vPWZlWEbmRAJqf+ap32VrybEeH4tAxzcdg7aYEPo5Grl+et9vAf9SA/xRHKO9udIJnwcqiZ/kQfQN6bqkt6I5lMXgekDqtplpNdxhHNceL2fnOHsp6X0qBS4H/VPE
+yKTjUadEWdlQ70/90Ltz6SVaDSnOpRs0rQgUzlqIlzp01w7q1V+Gyqb0EcEuysHO0JUt4AtASLkeEj8DkKGZWMj7hBkMbQNNPLhv0tr/8kpqBQ70UAnU0m3f9vHT1Ryz1VYtuQet73Q0fdLWrTA6KknSXfxKVG7mRCnhOCVFZ/zhfWperjFZQLOSd3UaEEj
nvgkisdB0Lfzt/aZIONN7AP2r2CpOn9QDvYxroH9a+++mEfI7dcIHrJkPeoheRzvUeQWrYkMy9FeuJCob95u0yF8J9Tv6ow4SklmR186Jej0LeD6NSx740aYGLVa839Vy/3vLFXK+qjg6i40sTokkzUO04JWQ5ONpxEIU8Q9oHFrdyT1BiSA1pSajJfGGQ8w
cgTHs3WRXXfITPWv6x0/4hxsxjzh5pfsbRU6TNOGGV9VziJlj/xJh1rtkkj1tzyjbOVERovmhmOs5KaM+75xtXkR3eytLcrxQP1yJEWeqDQQ0eg8v9pZJBbf8AS0U+18XTCKdCFKGXbUKf1w5d67CdE7tVfd4hJT0CI+t9jYD1WvksCxdilezvJvGyJX6ZPO
Y6nnqLGCOMzyPOf/eD1qD0F6af2z1BRU8JqkAIbLeO+go1eUFokxJ8d+a7sxdNkhelnzBYrIarynkJFuaQZ1HhicuNPVli41GDPlfFEdtPtJzjwz6g83DFw3VuOOlvLkw5GS/ZHKRSgRDw/MVs9Gxqg4L+u0qNSFLMS5X5PWt1pqs99LraTqSIbOh+bSbD8W
9q3/C8gM/GkHwQvc4ZwmEs50kMejkluurdCJ7D7ZuZZCbFpR6dIVRBkxXMQ29+lWIXLGAYJN700YNA4HO78tHFMs0gWbdJ8lZEfhp+A2BJg7YEIGVlwFdTho5LWUSWA7OjnAv7F99WRfoB17jXHTVA2zE3LosgWWrTht2TJh9zCI98/vWevVdAjVOAvQ2NKy
FB4/GxV3m44zzY7chD/CE/8xCPruIIJbmGFAMpZKnEPixVcJhvpmCLDsGQAEeyfEGFc2QVJXL3eOO3DHpn9PPLSDGlwHijrysTBK730vdKcwPQ7jk7TRunQfongM+xL1DPqAhOIHeNPaxsal2f9bachsbryXtRgtyyZx8pOSVSUDpYpDLiu+ll2LL87YbBkG
eaTh0/8GVn/iY0Y8iq1oamnUFJdIKpT/ETfOWRVAIhL2yzSr74naclioi1xT0MPi6e4XUBV4zIqOtdhLu8Zd2ga3jCnrrjGo7kq8wi8sGV7aa73+TV7tbyY/rovt/J5oIF9NSitv/d30PUfBGO86HIA14eg7q62pSUgwcvxYYzG99SQdVe2nXRJmecp/ECNK
y4In3iHjQ/eg+wGufrPm2CwJ8zmCwQ8vDFrAu2tHvMoH/cbArY5Hn/dIHawUJTU51ukLc7wx/IqKtXKzVKDal/Hmm9nKKEyrRAzVux9Y5YA2206d+VQDGEL+wTxjWymykQD3jdJkqYUzC2Wt6bAz/yH1UIikqm+/O7BvZhIl571l5dR0IRykvfIpjDPd1Kfi
jGuWCNjdSM9Otqy2PiCzuXBXX7p603lN/DFAZeHT84ByUL/u8+Q0r5c698Hm/Ao4Gc6KnSXM9xaU5vhnYHZd4pzTSQxbKwT7WfWKFCgozL83BcjxHEWbBQ1ToU7p16LJMnip5bfMvOphFOUTUyVMUk6E/bY95l2dQndsYJTZ08sjV0RUOHnR+r8yv7XBQca4
43UCxdtOCuDSfYpyIRNtNQ1eJfUmDKrCMVXzBrv5xb5Yxpj7EoYJsqbzfjA1ywqT4uxWq08cBb845Tkcflf3ZuJJFi76jy5MOJbPDIMlpBh/m0BsXaDd9MMc/elBXnPJHcaFBrpRCsJOrruKev0c+dAxawpvhEZA3f8AhIhM0cUhEqHs05zm55vpQzkwCEUp
3l7XEMiObtGCQNhenPGn4iKLgLn6O3R+aWDyJbvSiLo1FU8/GHbTxVsIRmQdOy4akQgWCb0RLKz2VrjSXAQa/ADGnhdgIiDoSSfKngBx2EvpZp03xFJw1J5y/3VN1iNqUf/2kFabbBskSd5+swtNkbWtucxB6S/dtPoVGY5zqdhLxdCWn3ECHMzBL+qw+55F
lFfNoDNfH1JRbeBu8zwcyfiVSHwm7C6A42bG48xC+twCw3oCka7ZnQOZiwQJFGVKnD4CmD34tZXvWWK9zQp+XlrN1fL2C5QI5pjzAeD4O6BkSdEXDJqN48xN1Juxyca4U4Jp2qbLXEXWTwVA5AN0JDK66vQEVq47fNWI6ewpiDAPX/s/mCODvbUPiGYfNqHQ
NXHx1K1syRVtG12TnAy8Cw3XbyOJXVm2Ri6Djp2q3/V7PDfT/TL6ujx9nxDprz8TRecxvPUM24Uz4J7XlCX4eAuZkvdDTg1//LjTYOIsvB3qeBmp6uVE9MEd7j36zWVIzf0UFL0ZMW2f403bUwx3Uq5JqYIrpbtz4ss3X3jSxdDkazG0HaH/SlX4Xx0IopKQ
k1NSYr4Iu4jVGQ9wFUoMgV3fzEPDoKJ3cE7M2QRsLGaC5TtGG+sfSDYGUZujH8c3B0f4EtKUALGrW7ip7XFynpSvr86CzXM6x0GCwamdpFgcG+Ad4/k915hHRxr1/lfyrZa83mPl1rl1GP7IzO8Bhf2TI/wHv07j3YukL5KbW5EwbOVwhwU7Cb0aFB0M0ESR
x1eAuG9Yh3NyMMfSyKrLkMbNGKB4ifzW3VDJXWAFGskNejOU6rBvoLa6EKyQKdcan4TK5A0DxqmkKkTHoPYrxo53pX1iKM9vYx4uVAhE5ZGCqj2ucXlazStF+tPbbX4+bdNwCLUkenmLrpBzTlgqsGueYqsDqQyXmQeGz1HWH0GYaUsNKQzg+GmFxwR7byAy
bZPOA6lc6Bd8QWZU9CbKfz7h88xVAXvHFY0lDk5O3f1FwR4zFMhQT60orfTNdOBdip+BPulZhwlPMp+gk2tWEWeEGKF8vGSITlNhGmw/uBqq2am+wgL6sJl/0FglzsObeQVwtjXcO5HOskwkb6L935ClhnvO0lq8DCtjD7J3LAfBl062gFDBmImQoSqGQB3M
3bUOJM/Qd72FegJO45SbnIZOx/vB7q2MJrDnQ05tPsw19P2PdMy2Eg3GFeAcgkwXQAeu0Kue7WLoudEJYRWs3xvvYQFKaxHD99YZ+1+3P7/w+2BY89sdzLdAxrINAGZimjlXxh9BmqWT5e+3jpxZe7MCaogNdO50G6//gYV8BP3kMVNcI33SHYsqfN/G85OH
tT+En5Fj8r5J8dbzABCtFoZVwqxteBOcIcdA2oV0C79171Y75LUOLKmy92ngPyVeB0VzvHVJmg6zf1WkPnThJ/0PvDvYWlOh/Z1p+4biyHAQXjcO0wf/DqH3QKsleZ6R0CeYgK2ym+TQ9XyJriDqKrOnXTDQqLMECTnoOx9TE2desQWJkzzEzOU8BWfwTHvx
fXI2a0pad3uOZCrz6Yf4q/pACSxsKyua033oZXIMPLtBzSK1FVh8jesRlmJj15hrdr4t1Hkvla2NdBePihk8VERiqhQPI2spKh6qVYEVqZYitEhJKjSeEp0D4MijzlgZ/S4BgxF8OQH1HoiH5aAfDwti9fTJUKl04YtbA2RhQGZQlxmmA0+KWyQ2dWIF26rC
oV1SI4hd6UdA4xJsMB0WdU1eQyZoyVJPTH7IIvGT1gyjFWhrgRAT6Nz4aGAESX/stPpzGTrn9co1o2Ji4DvLAQwupH313q7oF083invfsVvXstgGLNvO28EMTLURjcGgI9g2qUip2+Wc68lQgQDVa0tJXjfAu/fJEq/y2orBqkxU7tLS4S9NBCRuMOoXn1aM
jPnG9YQOggF0lryXenuSAH5d/fY0SbtqNcyu1R5HW4CYj/zVWjFc/mDjHbUzr4hmRPdvZrV9iwd/uP4hhkawgL67XVlLAui+B+2Vnmup0wEgqVEkTYa7u0wbvZgBFHWrcTOzQTsu80KiEW+dztdALhddFAuCHD+bu7AduhZkE7ISteWivL9wIuFux3Y+B5qH
Y/eJC0BS+djnq1snJdTwkUkvkMTliJVu8MFwaqISqJfgaIB7b98jY0v9KlrBNMlqzU0ppZGfcKjOXBNBXfBJGzrbChTAxKYqK6nKXEPocmnAakEEyn3SoTWHBziaZZm5wser4uobXx6dauJxS110mzWfZ1zUByipkM1/VRqoUcHRnFlS796J1R01J1y0CjUs
ArUQWdYj9GdeRA5dJLnr3roFPIEMdJx4J2E7EtWMUypRTGepLe+vZLmGNmNu8TOlqItGKsluJsS3E2cqLdS978jG+c1Jshxd88MIc3GhbOpBy2KdXpHYFQZZ6BH69yTOM5/chX3LXQsuNRUNANiZ8pxiCGtwxSenXs/cBKSHDunQVzxlBd1IYnTnJrLvP+UW
Vmgn0vUCXbw1cOUD8Ynv1AztPHvEJcUjVtfZcbcLWiFMvcbslE0Y4aVX6egH/R6EPR/94BcWVqURaXRRP7pSi8HMr9RNTu+xSh9bDzAdfWpmQ0nZkKS5n9N4IREPJ330DGYiGEiwWviPcMvDAA/jhopGdGLCNaNz/8phJM/1D3PsoYeRpjneHbxRaQELWmqd
4Vg9+r7eQ4siEdI9q7DNDm4x/HLFl6MIEVXcuabQxsIJf3PyFxccpIgaqkRexq3XQ7XMzzyeaBIGhcffxU01MRlCA65t1ElB/QmUdua05jpk5Y1Z6FwVSVU4s20IWT6tZ+VKUOQqDrlHOo+ayxKmyYLyY68rqUUb/DjvKj1PCX//QScizV9m7YV6dsoUCF/D
WbLDlVlwzQP18AGLdZi5uK5OedCk6Px/V1gw5GRSi+l0++lGM0RszeB8C/Z/YrVNjVRMISO1ZYFzBLnjldkpm0f2zLzii62zUQsZBhJoG+gP1fRiHAFPr5JfPhlOcEZRadm7JPzBBIpoC6d4WcqZQOcNy4uvWjdzQa+UvrJzoJ+fMmnq9/Y0EpK0hRBXS6dC
XYrVlVjvWYBwbvOFAspLBYc9F4Nl7np1W8sNz1arLPAc79U83YcoOfPAjWdFI+4Ow2PetuJqxOrcgJRGBfTuk4Kjujs72JDDp7Iz88RptHSk6PB1tJjWKTNAGAVsdj5EUAsvOW6quJ91tNTUaMBZtJqNA9DkDi6dIdhwpGFFurw6ZO0+uVhoHpwdhGAugEfz
AMP_WMAC_B64_END
```



### B\.27 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.28 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
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
AMP_WMAC_B64_END
```



### B\.29 源码数据块



```Plain Text
AMP_WMAC_B64_BEGIN
2uzF8SnW5b0MsEV48eU+H3Lc61x7oUM/oF/bfMRzsw0fH7RMpy5/LG5qm/kdGhN1pRpjvxDoEWGgzC66mDKlYq7qlC4TlIAQyy+p9V1bzbmKQAWMNuLB7sTKIA5Yh+oBgRdjJ0EBtCrljSkDK4lr+f2dHmxWYDUN/0XLsc0EqUCvZoxpgDJRX5R/kL+nD39c
CLXcAwgr4/laNWs+ynWvTE2DTHoV8hIHXdnHPXwQ6t6o1BXBlSUbyLblWxBi68oXexyE1pP4UbTL6ZgU/HydD8WFZ2MPuAQznJzJ7g8ocRx9sG+vh0sU5p1OWi2ydbiY5SoRSZOUErv7yO+21f+DFKthAnT08sA6QtxLGGD7yNTVuUilliU1SWdBmGZ7hUFI
SsvTXtPsvlGkeaGlRdl2b5jzMHb0DfG7xxcwQWuw7+JWvgY9ERYPQHKSPTSmto/ZqkGtxLX1BeeasFu0TFbbnrZ3gqjyFmrbQ0DxyaSp3OH8VyxEa/hjEv/U3GXqHNp1A/e+zMZjVsYNaSpqGQuPWhzNuslp6rOdZ5uac5JxERshW74z6opthenUPseOpwdW
OsFmp7SDhWlQzSQ7V8YURu/yV+d88DeTxKHkXVafXFd8rnl0iYw8WnJvmc7mkKSW9ieOnfL8nzvNb1Znwkt7EyfUsQHMVXxMPwWu+WJVPBWwDY7pqVQ2WCNLZXoey6/V519BT5dppo8Zk71v3XrlYa9XdA9p0CApCCF1xQLLm1koarRPjH/6l6gnHlZgHAxm
zXVe/t/gffIJSZhZT4frVT+552Vp4WhF4l5WK+jNpq+gxYWretCjiq5LMDD5VvJ/x1G59qkcyDVxlLIQd3Zzan0aKqUuIdRJpaOYR3c1qh4BjIJTg7RtV4Pt7RsGpkX5VJS9wg7O4keu8NUCKBzQJbTQXLcT5Ctz8baVBpXsAZR6ZtmpkA8jtynYBfXZ618r
+QIMbx6bC//YjeqolFkDf3MOi6jGmLJI9Wuix8ZwfdBGTgMo/9M1hbTFcWl46SVSN9ZSGZ45QRXOT0nZyiUTpUhvCZF/vRdyKs/a5qv5qwvvgwROgr0bezWQZXeAonvErV7dkznYzHzSmjKCStbOM4OqAS7/aj3gFccBpLGArC81iqufQLnlMiurVGHGzKQZ
FiteT6V5upI7jV2PpskThoKdZHB8P42OWTri7u97txB++qrLKtJk/KS4thj6OeHWCcvmfXYZ2rC+ZASJ20PE+6Zy5pVROAvNoTmp1PO6SNM3Aw6d//aAm0B/iPy/5OlKeYLSzgvBF4pS5hmu/3Z6zUrfyTPPVL671Q+YJpgCWrdd+LAazGExJRTQH1VN1nET
U3RxABtW4W1E/sS6CRDzSHB/pH9BUWJq1bPkJ5dVXHyAODftyZgS7zqtXK4GRJmmpJ6T/CbrukK6FYP7ADsQY20jIkF/rh1aFDFJ5GJJEEDl1qDLBL0jHy28UsHSlknpQT5W6MIpZ+gwaZ0dWQqVDfqfEMYdHEPNTL+26WOK57ZGS28YEUu7GJtNaLKnAT0J
LoTnF9vgmes05Xecg34Fn2zeW8slNtPI5DYrXysL8+WWy2U2VmaHuxa6dtfZpPyAuzG6Prao9N8+9+XKTEPJT4jbRXNCgtKVRKTgvM+wjx/Pg9WpEtXSqQAlkOj3a3TFtDQDKHe1SUPYNeGKGLUsTMTkfD8sqgBiQmWRRROqb9mYNscVWiZhh9hwF56VIvkx
+EaRPL6UgClgZjumCTOL7ciR3cEpz0pyYBLKYVlnvJ5/RHGRzYhNAYu0VCK1dHPtXlsVy4nz1RXn6/qGpV3HqXgkFdoLEJlTuT2iAGI/fI9Y+aQ2vn+Ix7OFzrvUWek/zX+te6zF/P2jqHq8nfVD42UdjVZMRY6loTLm0ao+02u4EoLbX0vz9umJVFpmRy6t
HKfmguZP1pMEaSk7cWS+c/5NG6ya8eIXSv4YkF3V5C5d9NX8QjNvfb4tPR0sgq/B5sBp6HRGq2+yNSc4gk+9d8+qcu8ucBEWoMB4JdyEIuYhIh1Ktsd5Be56Tf/Pw3t9jwUzcAIvJxIH5OJDJyGpQNbfMIWUcdYg5p/pceF/gfW2iNGJWHf0+JZ1Bv+I6+2T
5ThToGwto1D0coLaeiwbiUAlP7xWUaLmayramfAbEFrhspGWMsCOma03u2vdomxCDbV2MSAYrcmXqP+qPsaQYIvLkfO3fzHTynX5yXP1UDB75iLyMvLKCNORwsQDKwFmpU9Y8CmXYAbq1nF8IZ5kkxee20dcVVp6ReyPPN39+FfbXkad4XU/XGg1R1yHfiTs
BJqSUucVc1tBNFWcJgQ6nt4BmikaH/7VjWZOR41Ps5gM0NcUTH+RWshI6mNnVSETE5tfOWuGsgLmEy8yDSB04nOdhstT5e7diiXQD2Ze7LS1fwcHDtYsdTv4BxvGTGE3jLBPkszr2l7parUhCo63YlGrRud3kGtNaamu6gHe7Q3lYTctQDXw4aoHxwhwJ+eO
jmli7CncxX62VOV+428Jn6K01eNQIOKxSVz6JCk72olh8EYZYpHrRPbeF94D3/4HwuSCTcSwZbX64J6sxUygohwMW5ExiBSAMQVKmd6/zRkbbqrP5qZfjgxmj9ZtKPB9s90Wuj5I0R0m0/HVLWjzEfIWN6Tr3r7y4guNo3Ip/qiDx4GYzVKkPEIVm+vUXwfi
nl43gjIfWx0JBAlwkqR7mnAKsrZ8oePz92QHJGZQ1v4m4eM6C2DktUw+2KaCYwCVTlZaCPCVLDBVxzOixZbnHS3sl5pfsWwHvMMCkUkTmxhY76Ksfh1BZA/hQai5CEozfBjH3e7ajX/qfopAtaCVl6uOyHnKrO8no6Pgic4Sv87L0cktvPzZNl5bSMx4rrT1
cyUEKBcwakQW8V+HzbVXYc/ZLyHJxRtcxTr7YQp4cYUYLmcE1gDbOnBPDkWOk3SzDRtKmQ4K+t1Q8gfwEy7hrw4MV/h8Fm3GqZoi1y+BKL3O+n/9OJ+0MGnZAOa0Pfvj9zGulWaiJjAK2ubaopU0qVdyJ/sswjRJyUE9ZNNoP3iwYwHTM0Gp1V5kkGYZjWJX
pPNOusEpikmkVFHEtvXsdCqu/Lw5JCseobZlAkRUsunrtsiRb9jRiuMHjB6XucEnH4GKJDoXisMhdej6p/d/2B/HDpGfjo7GL88zy529ZSx39OsDqZ7QgeiXgvk28xMeC4zYkeYH5l4fvjQ2zIPi0rkn0DNPesd8rjpoCniRUDhD0UFWXcoE8BdO8J+82Eil
JBv2dIdImypncNLcvcY6LuosJ16lysxcV8mOHvNVZrjL46m0FZ8TDAAs/4Em4uUG+AjrcrEa+FQmnqhBayLJxV0+fVS3Sq3cr9IKXqjO4m8Jrf5fQFGQ2GaZ6JoEcJNQuTKck6712H9OIMpNDenknnGXma4Lf9cZg65fbR84gORgBr647AjfXer+S/UlpStI
mzX4q2fv+iVzwjy7Q2o8338mFx6WqySsNXhBCzEs4VTofbAbvTnAyZeAPnVPHtPEGGnQe6hGkMpokigYVG0ApP89aoSm/yZ+/oHskbEQhySwVXU5lqH0xxt7BC8erGR3arkjYmtLC84mnLF+o7y4ZgBu9JJTgJg+b3s2hEGEtUdM9oPWH90uEgJ+ohz8LRFN
1gbnJVPCgrdbdWChyqtdROINaeY5ZJILZSgytSADHhawqYb7K/WgmqnCxWwVdxfJyGF4+rnuUMk4WhMv5u0iHxA0BN7HpjsqE/SqAxXIgmNFhOKMCInIC02GWDfbF6COiJjvrYgeT89q+imRUeCEnK9kW6s8y5Ed4kPLJcJOgwqCKoUQi5Hwo2tSDuRtd6eC
HvzuzqZb/ueA7BRAfYDu1EdFBIkCSa6xNDi+P9FfIMZyvJza0NXKzjeUromHryuW5gHifPUMwl7E15Jb03+JQbizD8mkKd1RmBY1UjdPLhJfTJ0k04YeiMP12nS9tRBWyECloF+UVM2Sa9fsdbW8SbvKP6iN92O5he/xedXZ1mfcvep46FPaXWKkl+1VibRC
+xHH3BVqvOq6pLidChOI0uWpi+39kcxB2zYvDZgUz/f+b52RfoHvM9Izn+EQ16jnNO75ShlLQVAb5nW97Qum4h9zWvyoK0z5lcBWw80lq6E24ykSFKAf7ybZuAQopkwUQpyJ4q9hKxkqq7u2UShY+qhbazdYE1AFTuOU4b678DE0L3OnhcDoVpwSLcp8iPs2
A2snfkbLQN3Tn6ga6zv3YD+GierSP6R1eVhtFdcwLRPAVXYW4LdWJJXJGdrxMW6WccGhAkMFG0Dedy5f+5AkTl87i9eedEtZPI+kRFhAlIYiWB+ayVTxgovMQi60WLNSj/ZY/OVQEj/TcgesC+vwDj7RsNNKeOlwjq7RdI21FGuK9h9RuljM28dmghS6BBxF
1gSR9GVfyEmQ854twxjczIsj958R0XDsa/U238enoFenOhnCVNJG6ZBqJq2PURO8OGqmYsbQ1QxXhBHqYoIXC4QnYvf6Cl4uwLUtLb8mXeyfsTgT0p9kGlDVBJpCy03wBXGqrbwtUwSP7tlFvrh3Lvq3rcBi7kOCEB3xGD6+iuJqTIb6n1A/DnVyPToNGCjM
e2jZUI8bHf+eZFKYtcYjbDQuq0a5fTchogX2ehUi/4vvMynvmBDwxiFO4o4jy+aWdM3aQQQzY1VfEgNU75LR2Q2v8pqOWlIhdaLbk9EOQlU6fc4u+b5yp90YuC2MQY4tv7luUwKC3pMDntshANAaqahQuERDKfJgx5p2PbI71UCOU5Fu5N58TzMGBiu89grp
Abwbcgg+zWsz4bI/PjQNYz8xfXQOEJdgowuMcbEDfJ5ra3F9gcnCfQJjc/7oHPA0zxhEess1cI0rQ3ldbzzI5BlBZWgicsRq5MTid5s0OZ2n1AD84vZDSpDWcpevDSZpFwfdH5poYtbZ1YyX+FxvHl4VN3fTuNylfLC1I6GvXpxIh9O64O9Ij/NBQpi7i2i3
igtcnQDylxs/EZ6cu4VTZ3djegukXdvYZ79ukZBWZtv3Q6VfhLOJ9fKjMpXlPJOFK8/lh1RQhCNVfxFJ2olCe6ZnFFJKKKZ6U1H84rngi3+3YjV9qi+vUH2k3Kbm4E3W5RnoyFNuz9VDD1kRloL6693LxUXBuTLQa6zby2lL1p83qmyRc0VlKP5AW8TG/dXf
6jXGALG+wkhUPmmKwiewNa3dWOEUbyxxW6W2tfacqnIdIEVQkFZoUWh76U3simzTaXGHrnQdRFpwKISQ9yyrkRopisn2IQ2aQPtsfYQNaj3kQs6wQ2+ts4nU8zXFbHVCTA3nVkrjEFVWTPTbeAq+3COhozqFOcgse7paBSRR2itOTS9YkHoKdO37Pn9J0dMg
MM4AlsEcr5TT+5CLTJaRxMrV9FQmYYXr+7x6o1VfrOpBwd7PCSUZWC6rVQXpD7jvAhn2fVcTvA79PPEKHMH3g+33BPwO8qnmOFZ/GQNSB8/idXhq0Aq555jH1MD/UoZd5YcVJNC6N2TaClOXzfslup5G8qAuOnEspoheWK2tO0pGyNoR9nCzp6FSeBMQ7Hcc
Wh08w5SlZBYOG04kGsYTC7VYY1ICEXKvCYszi3klKDnvA/D0BkUXgWPdHcFSlXYIZEH1/BHU9yl3dsXzVYl1Msz+SwdmADdKf99hmfYPXiD7WYKvgf0VkaBv+hTwl6C0QLy46iuIakvJqSo0+goFZ8n1gzj5lB0gkp9Gpkxtjuq8GUqkDdXd2HpnD6Inoyl6
OuYUfdIaEygLxGAgNmR0CTHvZer9G8+EDKKnGgRMYqG3BLqEfiBEmbL9uLTQKUMs6OB1mVoJAII+7f92HsC7aDhh6XSTtB+fH6cJFikovRfG+Fr6sa2mWFh4j/92N9B5xsa19Nsv35W4aQOjyCYw4j5lpUveRbuN1VGiR9h65RYup9AEfSjw6l5ZmyHPsUeo
LQPCEwFsxDiPfd8IOdliFZi2s+JKvA1X2uE5ZfFcFjFgTfVw6mvi7g+IPoZ4V6M+MRBb1In76py4lf27ukuWtMxYCsuzVlmn/nHksYbsoj8l5GpIPVJEXd4N9CG1D7YEBpogxX21jbuvEDi28mbewEJZTavWYAQAfkxENmsVnyzzOZN2wtqxp3N4UfF1T2vw
6AWMxrtDk0RN/vflDr2nde7dJFuxuYtx2M6Z/NuX1P8KxX0Pr8O9cWPmrBSCEC/EVieRGOV+SjxeEb4u79jh813OO2PMyFWxeDCILvpMBOo0ECSxN3sRFUHkDL4fQC/DsFaDpyam3ogODurne+RaZUW++1XCFTA6ku0Xoqz0hjfmxH0PJ369N7t1L8SzK92h
d7cG5Nazv9Jn+J7XGBvT+a8KdKOmzhijZ/JjwfjJXVGcn7OzgCuJcR+oZYdxaKRlD1ZH5HlaR6ce13wLtarXW9K4Rn2jJbksBGNq6+oNOiz7Ude3TiN3QoSP2x8bzfyPeGMEgEukEDYA+hjO/8L3kqucSBI2qcYX2R10iIbu8iGPyUHFfwEiF+KDmBAJUZNB
MlzCVXAtKITFLbZdJdgE8s/5ydvpztX7CVTusb9BOi+DA+Ai3Pb6c6P9Iqbg4Jd1YPdZBQxR+7gQWUIcrc3/IwN6xCt9POjxXQGSv+x8BigC5idr93Hg34gbIzW/fmQxQEC9wSByYb0HRT1+A89Uke/L64e6vPVBy6+ZHfZfDL4wBcvYuK+HZmi87KZmbkar
ogsKTvR+sEy2/u/ekAVOtg9v/bDTqeKirao6qD+J9H0RTNEWpu61DGv6fwjwJ/CFD04+Tdwi7G6YijqIGLBjmI/IzuqPsDxAF6yxZ4oQgyhxsWmu34lyAihVUpRBbgELfSltKHASGoiPFWkrh3hoA0c1Qj/sqw+qO8Emw6YM+qJT96iNToyFeF69SnJsFmf9
Pet2TZa95kSzGGNkyNlZcMFvMccujjIYsITB4LzFw9ls1SKaYldroNaIICgvKnq9arPAXY0m0Z8BuQkSU2wqlwzC5ZKfjxSVLoFBZS3ZWhWy9rhHY12LfQ7J4OB43JVR1+av3ox6/iIDOR+1EIv/J7BQwD2tGGu+cQJlzb3zZ2SQblxSgaqC7nQfmNewZTpl
ViQN1Y9PBO+n6Fponn34jyiiw1NTItfc4TYt5A7XJBjX8btfIJU8Do7nncQr6iRbF75X4veQbJJeA0IsQh2VrLxEPnh6t0v6wqdYDBbY+b7HkVisRGwar1ibEqe20gtQ+zb1YewQQ/StNZsOoLT9cr7MDuCFHZ1DY4AmZRMkU6vKRrDerUCG4C7Bc9Ringwy
+meV7KK+LsqYpMHmEqXATEKgk73usMYI6vvlQoWC9kI1OzSdTmnt4/7uZmjyU3orOb6+318AQYE8hh+irsADh1T9JPKtGvU71hA6OuxufrzSyBMbtvoqAC+IFrw4WRmdmY+hzwPOmcNKzsFZudvmhyEwud85VbCTxugpMevvF618S3z1GKezP3O/SaWKHRuB
BilluSbU8bVJ5f0yEtk8wlxuWqjuDtA2aa3SrAvcDEOThllg3W/2xVbgcFIQL4J5hKaQm/2SRkH/a+JGtuHH9ZH/NgsIfwwbydt14dOFeWmortPJUBiQmbwUzjI+357i82qTMpLAr40fdXvNr0C0ecRXUAtcQkNgXxD75i7iAcA9rOODqSeg/DSl8XzMmpTd
/pjd0moqMw04G/d+YRNAz44kcKMKqM+/oJyQbMUgXTrIE1pgC3mBT1/OEmefi879of9ZPGU/03tBiQHTUGofbRBoQ5yFeM6A/xwWlRnu6pQ20Ghvkhhpdn11jk95oZHiVqvXjBVns/hdN0lhFRkF+5+7RGW8EeR79T0dA1Q98+eNZiCgCz10JRNpYhTZqeQo
Ap4qRxAeerNGpE4eV71jN/2QFCeJrB94k0BJkVauV/syAUcE+yluw824+QOslfLfjE3f408GIiMVFJ5povJMWZpiEb5VEJdGa2IdogjY37AZIbqSJ2FJkCvFagrpKuk8sXH17vNCTIIl8EJsjrVNvky9tnAfyeOwRENh6BwbfPblZmMsWmmDzHoilKQd1jAU
0Ba6Ekf0L3lIbSWB3ogDQAYSSr50TrKSbJfQV0Cy7E6IJ1DP0bIyyhiagh2f91yc8KtMCeWQEVfwIKWVRZ5NGakGXPLUELY97maq6ec9vXQl8eQ2drkwPB1PMUndJ+MJoUGU3LZv1TftFQ1eTGck8hHRH3tB0aChTL/WMPSLCCzrpI/0L6Oju6B8oLSOd8Rd
e2sp9TE6ynzy5ZXXV7wnOCImSHn1FSdDqzJvSvsEh6oxOPMSxfeNScBXPMzvr+xQGGhjeG38lzgSVne7elydZAPdDHW8s1MyTXA+cFjbEisvTouPeXnmys4kFdD3o7crwHtwJ6oSnwxhXZHKN3SPt9w2hQJWncIryXYuxgF+/VSZs8U59W8jZOgDuK681u7R
zV1jN87A339PA3dk+A4aLbqaXqlfJ+HqhWMs5foQzCJEIcQAarwgUXEpstmvlsxHt90EybYe2mmGEXSqXNz+v6TmX297v6cxukUaPMXkdy5getoq9e/+/V11it43ZhCje2uRKCl8hkKv0Phym0qweLV95vQllqA4AHA8OLZXqzYU3zfYcWO7/N9JA4jAK17C
pmnPctJoze5UzNI/TztgW068vvnxVjZRRUPewiJULQ3o1z4/RdyAdu+UpusrZIZCXwji3bQtMcb+SUb7IwfjZxunKkK9vVeAYKTd/i1qJpcCaogPMgVg+uys1plvYv0soVhxoWs2ndYPEUtwpEgyVrMWnTs+yGAK9RZEtuEWp89ktrUN1qctyj3CjP6zASVP
eDAikdEAYsYsWpj9CRflH7hKHveKh74ZWTe7t8N2B18WW3HJMGKLGE2TE6d+vgm8FuvDhz/NrAJXnWxDT8ZKAVQ3fBmdAnqvG7aw0lS7llkJE3/hT3xNpGtMPNs2243Pt7++NF9NOeo1vesLvk6HKMuHpl2HKnRzIGvj2I5XcVw2WhBQnI19KN4X8ev4JkbK
0WgGuahNqiOkne9q5qln4q9qU0CUBzohlotoxSh4cju7lnKEjBlPAhsYvGQj7IrQriQBXJ8qx3O5LN14u33EGmqxfDHUVhiVsPK0ugYwN/GQGLL85cLcgoQFP/5zRi7mMDMyYH3i+Y6/LqLYAA3Dgn97mySrxMUUyG5NkJvuJvvQPttz6+qE2lElvmAdSTs7
TmgQP4hz+3kYz4xZqqd0w0gkRgKjpZRVZD6Zu0BPtVVbBINbehcJoKTE6RZBOt2c6ZDRWkazkGzHOXPIo9PGvTd6xQ8hiNt/mYjdyPgDpirYfi24EX3Vod1tYzjQxCv0CMYGzeueiN+Se4WYvz9+eF5ERWLGaxinCJzFNwKXyJ/mw0EOO1uPLAngay6GDM9/
bS6cVfWQ6XiarFp8HtuDNQ/9AcX7Kuo8sbR+l++QbrPN+uck2fIqQ32/zqLF0/ufhNj7YthBgHGCwx0Ogk4UNQGu9KF/M++Q8AlojEwY81gjTQ0rh0jFPVHBgqoB0BNDK0VZhRDQ3iv9gWpqu/UXQuYw/gwF9xYNNZ1Ou8qWu2p+B7fjmSIJ9JONr7InaWl1
+FvXB/eCWyYcjCStEHsYrIjC2T/F7FU0EBaC351MwEgDNOuKqPM587A3ghm0TMOTWSa0xPi5pLWp/0a+hh1dHHXTRwXwi88HmZmkgY+vlLfODB3Rv6cVo6srNPNfYMaRN6NPZ9Xd+wK3Om5+KEtTENLg9PbnjSHEJQa9Pz8H3THvrf29548j4NT3AAn21eQx
MMRikONSE7kd0R25bfacPilfbPFw5utk0rmD8DXfc41uNbnDTSBzRVnBY4dLFZBq7q+yIJUhppVwlADuGttEIwaC/3mq5g4zi6FUV2YQlW6JhsiflJHW+WcEwLMr4QM/dZXbQ4kJB6weeEnGQ9V0CgWw4HZ2IkFAsHuPmIvs/Z+VcYoC3sSZEjXvilzJa2eY
DrogcNQVKFk/hK5ogekPjsFT0Y3fCf2cxI8tF5wvAjGWhJcuQ5kYvCm6kn4BCv3WsanE0He+ISa32NSKe69+/Er9aAQ/vrhsN0NxotzAyabVnRWf/TeeVpkJZNAsaHlDBWgH1SbswZyq30EAeIi57oq9ErYPjibHmTkAYc3p2ObG+VyXsnJzT62Hoxs299cA
Drx4OEQ/u+Lu+Zy2lzksC1WUpQJ3wz+m9lI3GnJkKxilAiHR65/hZcCXSr0OJKBD+6LECPDVnu09zdz3Yy8bmG8eUGXc/IcmM9LcYn4GG7LoUCNL9SkDmAvPwxwdDoT7LS9dEASoiPo43usPUEXXERrZPEM50+SwzsTLCiy5CQ9aWr54X5xK/OjRU+unHQ1C
pJMO0aizx1sWMCiEwWZtj9UM2Zvn6JcFDEVUfJalp01FWicVismwuGKPwHts4uFib3Edr+NCw2p5ZzSJDqjTplPsf6UQiEcyvHhgBY5ZfbNip49NRt57/IoGkuHwsnrij+5aYVQwxCBIGCP7Hf95FnOx8fQkQqyzq39l1y1alKJWynZMEixKYAsnoOGzCf2j
WIKktH4ooxffH//A/tPuLDc8efOg0SCeSK8A45umdh8KWuPcWrZxS1kl8gqnkiGTESwGE9xvpewxHWPzZI2yYa+LW+D++GNqO4n0Y7e/NtOr5xTu2JZuOtoU6afLJHsvdUgf5fplYiEuNkpgPWq20fpwVVjoVhabNvI+b/egu1nZ3G3Z31+qMgsT/oMWvZcV
JEXHbqEey0pZsZJpnXDB1NekQBqmb/o7PNSHvNV7iBsF4i1nGUlC4T4onoI72HKjZs1A1o4PYNZbWU+BdYcw80O+1QP0YnA8TRZU931q/Wl8diehvP6C84eSimPwh/OqiFjH/nUxmgvZ7ykMNgfu4c+JDszc6s7c+PqaFZ385vo3Q6tPNcmkzgzDPpIgxUYe
86OvP0S4nSdW3DqEMpgrokETOWGWNSqhv3Ku4kWyHjRk1TNwo7E6SMIzeNZZN2znV2wa3TCcNL/AtiadeYO7iYoxR5WEjXs/4zkrwng9E+KJxuH4ZfiQO8pPCIRqFacWtRqRwyUx+VUKzHKVxBdqqVTqWfHwDuTbNWRNcV7aJknObEYWuJMQsoonycRB8ol7
f7tq22qE+UUF0Fq6aqAeDpAmnyqza9YDRaDmJKqKke+VhgITFg00FUmGuI0sgLgzZBkYYcayFpNDcwurvCzndS4ta5c/59238c1f2qBs8tfc0TC5SL48oJtHvmZtS2me7p/Sv3jL67RdoEkCB7hnXmVz7sYm0M173cdlRpZsj1xF3Zj7qmyIPS53atyDBbJo
9JVXSgTjrNCcN3Cm9G+CJy1PXG4pZWad8lHlWNDDlT4GeB2mbXxqB3sbkYV/F1Oe5jMeM2YCTZRs7Wj4PqFHVyQEGmj1nJpBLzpMujgBU6vDxxl1Atk94NT+KZgC/ieISK1+ndBWD61jJA6JeCxg/f/jV7F9q2e5xOAOmVZrrUET05Sqzhoz88Y8Uyr1waSj
sSXr/pNy2YhhdWQ/r/L8pMIctPItSf2GlifWbcmCteFsOONCsmC8zXScVCkqpmI507vavPKeSqeZrrplmcDteUridt2XZzxqhQgU/HkL9EMyTTFhMRgnxcTMTzJKOSGKK9YePneIdMPQUsvkTZ4vOAWPQ7Cm+QAy5WHtDNLoDaSMLsvQzYuqwWYrV3g7aetW
Vt/NLwmjOiUyBVJ2Y7I9d0/cE1QPFTyknWLo51TknA3zNaDqSbPdyK60xZMRuBwLoj2+I13Yd+sFlK3Z4o8/Mij1eUjjmm7pr1euCH8KTH+vNKhq1ipX7EVcP+bSgur1DmmfWFTxpk3MCQ26JLR0aAxKwJF6lV3cAxiMWE08yxWf+ShCrzD1MQtd/ShoBqYa
asKnQa31mW1bKI6oj0nUpdxRamCe/0a5rE5bo/aimKEAUTxWrtRy4uZPJQhZH9sfZdMLVpRgMcdWLrOABI4HKzYmP/SrtsNFiXGnhAodiAB0T3bUDy8xXwAB7IAT8oNpDfIZw7HEZ/sCAAAAAARZWg==
AMP_WMAC_B64_END
```



附录结束。完整复制本文件即可按第3章重建；无需访问作者电脑或共享盘。

