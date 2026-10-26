# 安装、第一次实验与数据复用

Python ≥3.10。CPU流程无需Torch、Triton、Matplotlib或GPU，SVG报告使用标准库生成。安装来源是本仓库源码或自己构建的wheel。

## 源码安装与CPU实验

```bash
git clone --branch research/5060ti-readiness https://github.com/MingASA/KernelTuningLab.git
cd KernelTuningLab
git checkout --detach 35ac36ed66becf19271c9ed3e97950fb67a03d87
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
.venv/bin/ktlab probe
.venv/bin/ktlab research --spec examples/research/cpu.json --output artifacts/cpu-plan --write-only
.venv/bin/ktlab research --spec examples/research/cpu.json --output artifacts/cpu-study
```

若系统Python缺少ensurepip，使用本机完整Python或已有 `uv venv --python 3.12 --seed .venv`，再执行pip安装；无需修改其他CUDA环境。本轮实际复现过这个WSL问题。

这是固定的5060 Ti交付基线；main尚未合并相关成果。需修改源码时先创建自己的开发分支。4090/H800的安装与实验必须使用[跨GPU报告](CROSS_GPU_PHASE1.md)声明的准确实验commit，并先确认服务器GPU、调度、时长及workspace授权；不要把5060 Ti的cu132锁文件直接套到旧驱动服务器。

`probe`默认只检查CPU环境；`--gpu`才加载CUDA。CPU测试跳过需GPU项目，设置 `KT_RUN_GPU_TESTS=1` 才启用物理GPU回归。

研究命令在stderr显示计划和逐job进度，stdout保留JSON；`--quiet`关闭进度。`--write-only`列出job、trial、观测和独立验收上限，不加载插件，不证明GPU可运行。正式执行前可用 `ktlab check --spec <v2.json> --output <目录>` 做实际预检/验收。

`validation.max_configs`是整个v2 job的独立验收尝试额度，不是每个Shape的额度。要覆盖全部可用Shape/context，至少给每个context一次尝试；例如6个Shape而额度4会如实返回partial，即使搜索观测本身均已数值通过。提高验收额度后用同一输出恢复，旧attempt保留，兼容搜索观测可缓存复用。

## 阅读输出与恢复

先打开 `artifacts/cpu-study/report.md`：

| 文件 | 先看什么 |
|---|---|
| `summary.json` | 状态、观测、新物理测量、缓存、验收和job墙钟 |
| `trials.csv` / `observations.csv` | 同一上下文下的候选序列、best-so-far、延迟与成本 |
| `failures.jsonl` | 失败job、具体case/config和分类原因 |
| `research_manifest.json` | 请求维度、job状态、源码commit与原v2输出 |
| `runs/<group>/measurements.sqlite` | 完整历史、原始样本、协议、正确性、源码/硬件和失败 |

CPU测量证明CPU调用，不是GPU性能。GPU Graph样本是捕获的完整GPU计划摊销时间，编译、输入/参考、Python服务调用在该latency之外。编译成本是测量墙钟的组成部分，不可重复相加。

```bash
# 中断后继续；旧attempts保留，验收会重跑
.venv/bin/ktlab research --spec examples/research/cpu.json --output artifacts/cpu-study
# 只读已有证据重建报告，不执行新测量
.venv/bin/ktlab research --report-only --output artifacts/cpu-study
```

兼容缓存命中仍消耗观测预算，不算新物理测量。被终止候选保留诊断，恢复时重试。不同输出目录之间没有GPU调度器，同一张卡应串行执行。

## wheel与源码目录外使用

```bash
.venv/bin/python -m pip wheel --no-deps . -w dist
python3 -m venv /tmp/ktlab-wheel-env
/tmp/ktlab-wheel-env/bin/python -m pip install /absolute/path/to/dist/kernel_tuning_lab-0.1.0-py3-none-any.whl
cd /tmp
/tmp/ktlab-wheel-env/bin/ktlab assets list
/tmp/ktlab-wheel-env/bin/ktlab scaffold algorithm --output my_search
/tmp/ktlab-wheel-env/bin/ktlab run --spec my_search/experiment.json --output my_search/results
```

wheel包含库、CLI、内置资产、公共数据小样本和第三方许可。仓库examples和完整研究脚本由源码仓库提供；只装wheel时用scaffold生成独立示例。文件插件相对实验JSON解析，安装包插件通过 `kernel_tuning_lab.assets` entry points 发现。

## RTX 5060 Ti最小GPU路径

以下固定栈已在本机独立重新安装验证；不修改系统驱动、系统CUDA或其他项目环境：

```bash
uv venv --python 3.12 --seed .venv-gpu
.venv-gpu/bin/python -m pip install -r docs/data/5060ti-requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu132
.venv-gpu/bin/python -m pip install -e .
.venv-gpu/bin/python -m pip check
.venv-gpu/bin/ktlab probe --gpu --hardware 5060ti16g
.venv-gpu/bin/ktlab run --spec examples/research_phase2/gpu-minimal.json --output artifacts/first-gpu
.venv-gpu/bin/ktlab research --spec examples/research_phase2/gpu-smoke-research.json --output artifacts/first-gpu-search
```

最小示例为两种小GEMM配置，保留独立参考、最终实际输出与输入完整性检查。原四类十阶段gate和GPU回归另行执行：

```bash
KT_PYTHON_BIN=.venv-gpu/bin/python bash scripts/run_gpu_smoke.sh 5060ti16g artifacts/core-smoke
KT_RUN_GPU_TESTS=1 .venv-gpu/bin/python -m pytest -q tests/test_cuda_graph_runtime.py
```

其他GPU先看[迁移检查单](GPU_MIGRATION.md)。完整proposal-suite有许多Kernel、Shape和额外验收，`max_evaluations=4`并非整套只测四次。

## 导出、Replay与L3

从manifest找到具体的 `runs/<group>/measurements.sqlite`，替换到下面命令；导入保留原证据等级：

```bash
ktlab export --database measurements.sqlite --output measurements.jsonl
ktlab import-jsonl --database restored.sqlite --input measurements.jsonl
ktlab report --database restored.sqlite --output artifacts/restored-report
ktlab replay --database restored.sqlite --split test --budgets 4 8 --seeds 0 1 --output artifacts/replay
ktlab l3 --database restored.sqlite --output artifacts/l3
```

示例只有train标签时，可用 `--split train` 做软件回放检查；正式评价先划分整个Shape/GPU再训练。CLI从已观测域重建有限候选：匹配SDK资产提供合法性，资产不可用时明确限制为已观测配置。coverage不代表完整开放空间。精确自定义空间可直接传给已有 `run_replay(records, searcher, explicit_space, budget)` API。见[科研协议](RESEARCH_PROTOCOL.md)。
