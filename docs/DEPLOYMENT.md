# 部署、环境选择与数据恢复

最短流程：**初始化源码 → doctor → 选择profile → 安装/构建 → 恢复数据 → 最小验收 → 使用**。核心服务不要求常驻GPU服务器；Web是可选的本地只读查看器。

## 一、准备稳定源码和Python

```bash
git clone --branch main https://github.com/MingASA/KernelTuningLab.git
cd KernelTuningLab
git checkout v1.0.0
python3 scripts/bootstrap.py --profile core --env .venv
source .venv/activate-ktlab.sh
ktlab doctor
```

核心需要Python≥3.10。GPU兼容profile按记录的Python minor选择；缺少解释器时使用集群已有模块或自己的可信Python发行包，不升级系统Python。离线时使用校验过的source bundle和core wheel：

```bash
python3 scripts/bootstrap.py --profile core --env .venv \
  --wheel "$KTL_CORE_WHEEL" --wheel-sha256 "$KTL_CORE_SHA256"
```

`bootstrap` 新建私有venv，拒绝覆盖旧环境，安装后记录 `runtime.json`，生成 `activate-ktlab.sh`。它不执行GPU实验或模型/API调用。

## 二、选择执行环境

| Profile | Python | 核心依赖 | 用途 |
|---|---|---|---|
| core | ≥3.10 | 无运行依赖 | CLI、SDK、CPU、SQLite、报告、Web、L3 |
| optimizers | ≥3.10 | SMAC2.4.1、KT1.5.0、NumPy2.2.2等 | 成熟原生优化器 |
| cuda-cu124 | 3.11 | Torch2.6.0+cu124、Triton3.2.0 | H800/4090记录栈，Flex/SGLang兼容路径 |
| gemlite-cu124 | 3.11 | Torch2.6、Triton3.6.0、GemLite0.6.0.post2 | raw-A16独立overlay |
| flute-cu124 | 3.11 | Torch2.6、Triton3.2、FLUTE High组件 | 24个P32模板的独立profile |
| moe-cu124 | 3.11 | Torch2.6、Triton3.2、官方vLLM0.8.5组件 | 完整MoE专家路径 |
| 5060ti | 3.12 | Torch2.13.0+cu132、Triton3.7.1 | 原5060Ti核心GPU验证栈 |
| local-model | 3.11 | Torch2.6、Transformers5.8.1 | 原本地推理基础环境，权重另行恢复 |

这些是记录过的环境profile，不是对未来驱动/架构的统一兼容承诺。完整源机freeze在私有资产库 `deployment/locks/`；公共关键组件约束在 `docs/data/`。新的GPU、dtype、执行路径须单独验收。

兼容profile可加 `--with-optimizers`；冲突的NumPy等锁不能强行合并。不同生产后端使用独立环境，通用依赖可以共用经过SHA校验的wheel缓存。

## 三、离线安装与原生资产

```bash
python3.11 scripts/bootstrap.py --profile moe-cu124 --with-optimizers \
  --env "$KTL_ENV" --wheel "$KTL_CORE_WHEEL" \
  --wheelhouse "$KTL_WHEELHOUSE" --wheelhouse-manifest "$KTL_WHEELHOUSE_MANIFEST" \
  --constraints "$KTL_LOCK" --components "$KTL_COMPONENTS"
source "$KTL_ENV/activate-ktlab.sh"
```

`--wheelhouse`关闭网络解析；manifest为 `files:[{file,sha256}]`。完整约束排除核心包自身，以免旧freeze将核心锁回0.1.0。安装、CPU加载、设备probe、数值通过分别记录。

### MoE

从固定官方wheel提取原始组件，不安装整个服务引擎：

```bash
python - <<'PYCODE'
import os
from kernel_tuning_lab.production.vllm_moe_compat import prepare_components
prepare_components(os.environ['KTL_VLLM_WHEEL'], os.environ['KTL_COMPONENTS'])
PYCODE
```

提取器核对固定wheel与原始二进制/源码SHA。该组件不是Marlin-only子集。实际执行还检查Torch2.6/CUDA12/ABI0及原始注册接口。

### GemLite

```bash
python3.11 scripts/bootstrap.py --profile gemlite-cu124 --raw-gemlite --with-optimizers \
  --env "$KTL_ENV" --wheel "$KTL_CORE_WHEEL"
source "$KTL_ENV/activate-ktlab.sh"
```

生成私有raw-A16 overlay，保留原生数值函数；不修改原包或全局Torch。范围为W2/W4 A16，不是任意GPTQ/AWQ checkpoint导入。

### FLUTE

恢复的runtime目录必须含相同Python模块、扩展和包元数据；manifest绑定扩展SHA、模板、架构和精确资源facts：

```bash
python3.11 scripts/bootstrap.py --profile flute-cu124 --env "$KTL_ENV" \
  --components "$KTL_FLUTE_RUNTIME" --flute-manifest "$KTL_FLUTE_MANIFEST" \
  --wheel "$KTL_CORE_WHEEL"
```

原High扩展为cp311/Torch2.6/sm89-sm90产物。Python ABI、Torch ABI或GPU架构不匹配时重新构建，不能复制旧venv后宣称兼容。构建流程复用 `scripts/probe_flute_resources.py` 和 `scripts/build_flute_subset.py --accumulation high`，详见[生产支持说明](PRODUCTION_STABILITY.md)。原Mixed与High的身份和历史记录分开。

### Flex、SGLang与模型接口

Flex/SGLang兼容路径使用 `cuda-cu124` 与仓库内固定源码。其他Torch版本为不同profile，不能把源码可表达当成实机通过。模型先恢复名称、来源、revision、配置/权重文件SHA和许可证要求；共享权重不随归档自动复制。API凭据通过用户自己的正常授权配置，不进入Git或归档清单。

## 四、E盘资产库

Windows位置为 `E:\GPT\KernelTuningLab`，当前WSL路径为 `/mnt/e/GPT/KernelTuningLab`。换机器先核实实际挂载和空间，不能仅创建同名普通目录冒充E盘。

| 目录 | 内容 |
|---|---|
| `archives/` | 原始封存与按内容去重的资产包；未完成传输以partial保留 |
| `catalogs/` | 源路径、SHA、对象归属、元数据保留项 |
| `sources/` | Git bundle、固定源码快照与发布源包 |
| `deployment/` | 依赖锁、原生资产、可信来源、可复用wheel/source |
| `checks/` | 传输证明、逐成员SHA、SQLite恢复检查 |
| `records/user/` | 用户自己同步的可直接查看实验 |
| `private/` | 机器说明、恢复receipt、私有操作指南 |

资产清单区分实际内容与metadata-only缓存。后者保存SHA、版本和来源，不能算作已复制的内容。模型权重需另有复制权限和容量；不把原始私有数据、账户、UUID或凭据上传GitHub。

## 五、从归档恢复

```bash
export KTL_VAULT=/mnt/e/GPT/KernelTuningLab
ktlab archive verify --vault "$KTL_VAULT"
ktlab archive restore --vault "$KTL_VAULT" --source H800 \
  --prefix "$EXPERIMENT_PREFIX" --output restored-data
```

前缀从源catalog选取。恢复保留源目录结构，核对成员SHA并检查SQLite；拒绝路径穿越和覆盖不同内容。它只恢复数据，不重新测量、编译或运行旧队列。数据已打开使用时，建议恢复到另一个新目录，保留两份状态。

源码用Git bundle恢复；原生runtime按前述profile恢复。PTX/CUBIN和扩展保留为原实验的身份与证据，不跨架构自动复用Triton编译缓存。

## 六、最小验收与使用

```bash
ktlab run --spec examples/quickstart/cpu.json --output artifacts/new-install-cpu
ktlab doctor --devices
# 仅在获准空卡下执行：
python scripts/with_idle_gpu.py --gpu auto --expected-model H800 \
  --receipt artifacts/new-install-resource.json --timeout 360 -- \
  python -m kernel_tuning_lab run --spec examples/quickstart/moe-h800.json --output artifacts/new-install-moe
ktlab web --root artifacts/new-install-cpu --root artifacts/new-install-moe
```

每次资源准入用新receipt；恢复实验可保持原输出目录。资源冲突、编译失败、数值失败和缺数据分别保存，不放宽阈值来验收。

本轮验收分为源码/核心CPU恢复、私有依赖环境恢复、目标机器最小GPU运行和人工上手。现有机器的新目录验证只证明所记录机器与环境；没有新架构服务器时不标记跨架构通过。最终人工操作由用户执行，自动验收不等同于用户已验收。
