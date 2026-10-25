# Kernel Tuning Research Lab v0.1

面向硬件特性的算子参数寻优研究平台，覆盖 L2（有限测量预算下单 Shape 搜索）和 L3（有限变体集合与分派）。原始立项书和 PPTX 保留在仓库根目录；其中历史实验图表没有原始数据，**不会作为本平台实测数据导入**。

Python >= 3.10，核心包无第三方运行依赖；Mock、搜索、Policy、SQLite、L3 和 SVG 报告均可在 CPU 上运行。

## 新 Kernel 与算法：配置驱动接入

新贡献者从公开 SDK 和可运行模板开始，无需阅读或修改平台底层：

```bash
ktlab scaffold kernel --operator gemm@1.0.0 --runtime python --output my_kernel
ktlab check --spec my_kernel/experiment.json --output artifacts/my-kernel-check
ktlab run --spec my_kernel/experiment.json --output artifacts/my-kernel-run
```

`scaffold algorithm` 生成独立算法模板。算子、Kernel、搜索算法是各自版本化的资产，通过文件、模块或 entry points 注册；v2 实验只选择资产、Shape、精度、空间范围和预算。平台自动做接口/参数映射检查、空间交集、算法合成验收、有限预算数值核对、完整流程计时、断点缓存与结果归档。多步骤实现声明 `ExecutionPlan` 和工作区，初始化也计入每次完整调用。

正确性声明可按版本、内容和范围跳过重复数值核对；声明来源与平台实测核对分开保存，默认 verified Replay/L3/报告不使用声明性能。父进程执行阶段硬超时。缺 CUDA 为 `deferred`，不会冒充 GPU 验收成功。

底层可靠性检查覆盖输入完整性、重复调用及计时后输出；失败样本不会作为成功性能。执行配置先冻结为快照，输出目录互斥，重跑保留 `attempts/` 历史；被终止的候选可恢复重试。证据等级、预算和生产使用边界见 [HARDENING.md](docs/HARDENING.md)。

```bash
ktlab assets list
ktlab run --spec examples/extensions/cpu_gemm.json --output artifacts/extensions/cpu
```

这条 CPU 示例自动组合两个实现、两个算法和两个 Shape，并复用跨算法的测量缓存。新增算子、条件空间、正确性声明以及错误输出/非法建议/不兼容能力示例均已提供。接入步骤、契约和配置说明见 **[EXTENDING.md](docs/EXTENDING.md)**；实施计划见 [AUTO_INTEGRATION_PLAN.md](docs/AUTO_INTEGRATION_PLAN.md)。原 v1 `benchmark` 和完整 Adapter 插件继续可用。

## 基础研究工作流

项目书中的 W4/W3/W2A16 反量化 GEMM、独立反量化、Attention 和 MoE Grouped GEMM 已接入同一资产流程，并提供生产形状与开放搜索空间：

```bash
ktlab proposal-suite --runtime python --output artifacts/proposal-cpu
ktlab proposal-suite --runtime cuda --hardware 5060ti16g --output artifacts/proposal-gpu
ktlab proposal-suite --runtime cuda --scale production --hardware H200 --write-only --output artifacts/production-specs
```

融合量化、分块 Attention、persistent/grouped MoE、Split-K 和实际 TMA 路径保留可调 tile、流水线等参数；接口支持硬件/输入/工作区预算裁剪及参数化验证案例。RTX 5060 Ti 16GB 已通过十阶段严格 smoke 和八类新增算子的默认 GPU 套件；实测边界、首次失败及修复见 [5060TI_RESULTS.md](docs/5060TI_RESULTS.md)。实现与格式见 [PROPOSAL_OPERATORS.md](docs/PROPOSAL_OPERATORS.md)。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m kernel_tuning_lab demo --output artifacts/demo
```

Demo 覆盖三个虚构 GPU、四类算子、每类八组 Shape 和 24 个候选配置，执行六种搜索算法、独立 Policy 评分、K=1/2/4/8/16 的 L3 选择和精确对照，再输出 SQLite、JSONL、数据划分清单、策略版本日志和 Markdown/SVG 报告。已完成的一轮运行包含 2304 条记录、2880 次搜索比较、64 次策略评分和 240 个 L3 结果。所有模拟输出均标为 `mock`，**只能验证基础设施，不能证明真实 GPU 性能**。

较大规模的 CPU 路径也提供一条命令：

```bash
ktlab scale-demo --output artifacts/scale
```

这会在 **139,968 个配置**上执行六类有界候选池搜索，并求解 **128 Shape × 64 Config** 的合成 L3 问题，同时给小规模投影做精确对照。大空间未全量测量，因此全局 regret 保持未知。核心测试另覆盖百万级空间下的采样和 Replay，禁止隐式全量展开。

可选安装绘图与开发依赖：

```bash
.venv/bin/python -m pip install -e '.[plots,dev]'
.venv/bin/python -m pytest -q
```

真实 NVIDIA GPU 上安装对应 CUDA 版本的 PyTorch 和 Triton 后运行：

```bash
.venv/bin/python -m pip install -e '.[gpu]'
.venv/bin/python -m kernel_tuning_lab benchmark --backend triton --operator gemm --hardware H200 --max-evaluations 8 --require-success --output artifacts/gpu-smoke
```

支持 **H200 SXM、H800 SXM5、RTX 4090、RTX 5060 Ti 16GB** 的规格配置及实际设备兼容性检查，`--hardware` 接受 `H200`、`H800`、`4090`、`5060ti16g`。5060 Ti 用作上传 Hopper 前的轻量单卡验证机；会拒绝将 8GB 版本识别为 16GB。未核实的字段保留未知，真实执行读取物理设备属性。5060 Ti 已有实测，H200/H800/4090 仍须各自实机验证。

| 算子 | Shape | Dtype | 实现边界 |
| --- | --- | --- | --- |
| GEMM | M/N/K | FP16/BF16 | Contiguous、FP32 累加、掩码处理尾块 |
| Batched GEMM | Batch/M/N/K | FP16/BF16 | 独立等形矩阵，三维 grid，无广播 |
| Softmax | M/N | FP16/BF16/FP32 | 最后一维、稳定减最大值、任意行长需由 BLOCK_SIZE 覆盖 |
| LayerNorm | M/N | FP16/BF16/FP32 | 最后一维、非仿射、显式 eps、FP32 统计 |

上述四个 v1 适配器中，GEMM 搜索 BLOCK_M/N/K、num_warps、num_stages；归约算子使用独立的 BLOCK_SIZE 空间。它们提供独立 JIT 计时、Torch 正确性验证和 CUDA Event 热态采样。新增资产中的 Split-K、TMA 和完整流程计时见 [PROPOSAL_OPERATORS.md](docs/PROPOSAL_OPERATORS.md)。

后续在目标 CUDA 机器上一次检查全部算子与支持的精度：

```bash
KT_PYTHON_BIN=.venv/bin/python bash scripts/run_gpu_smoke.sh 5060ti16g artifacts/5060ti
# 本地通过后，在服务器将 profile 替换为 H200 或 H800 再运行。
```

5060 Ti 属于 `sm_120` Blackwell。原生后端要求 Torch >= 2.7、其 CUDA runtime >= 12.8、Triton >= 3.3；可选依赖为 `.[gpu-blackwell]`。脚本执行十组严格检查，任何被测试配置失败即退出，成功后生成 `validation_summary.json`。安装、预检及迁移步骤见 [RTX5060TI.md](docs/RTX5060TI.md)。

CLI 更多命令：

```bash
ktlab --help
ktlab hardware
ktlab write-spec --backend mock --output examples/mock_experiment.json
ktlab benchmark --spec examples/mock_experiment.json --output artifacts/spec-run
ktlab replay --database artifacts/demo/measurements.sqlite --output artifacts/replay
ktlab report --database artifacts/demo/measurements.sqlite --output artifacts/report
ktlab export --database artifacts/demo/measurements.sqlite --output artifacts/export.jsonl
ktlab import-jsonl --database artifacts/import.sqlite --input artifacts/export.jsonl
ktlab import-public --database artifacts/public.sqlite --input kernel_tuning_lab/adapters/data/benchmark_hub_convolution_A100.sample.json
ktlab demo --holdout shape --output artifacts/unseen-shape
ktlab benchmark --backend mock --operator layernorm --searcher surrogate --max-evaluations 8 --output artifacts/online-layernorm
ktlab demo --hardware H200 H800 4090 --operators gemm softmax --budgets 8 16 --seeds 0 --output artifacts/profile-mock
```

AutoTuning Benchmark Hub 已支持固定 commit 的完整目录发现、校验下载和整文件导入：

```bash
ktlab hub list
ktlab hub import --kernel convolution_milo --gpu A100 --database artifacts/hub.sqlite
ktlab hub import --kernel dedispersion_milo --gpu A4000 --database artifacts/hub.sqlite
ktlab report --database artifacts/hub.sqlite --output artifacts/hub-report
```

本轮实际导入上述两份完整缓存，共 **15,492 条真实上游记录**；带运行时间的记录缺少可独立确认的正确性证据，因此保持 `correctness_unknown`，不进入已验证性能 oracle。目录核实、文件哈希、许可证及已导入范围见 [PUBLIC_DATA.md](docs/PUBLIC_DATA.md)。已核对的活动目录没有 H200/H800/4090 数据。

本地插件也能从 CLI 使用，外部实现无需修改核心包：

```bash
ktlab --plugin examples/plugins/deterministic_mock.py benchmark --spec examples/experiments/plugin_mock.json --require-success --output artifacts/plugin
```

`examples/experiments/` 提供四类算子的 Mock/CUDA 实验规范；[自动生成的报告示例](examples/report/report.md) 包含比较曲线与精确解对照。核心实现状态、研究边界与后续工作见 [STATUS.md](docs/STATUS.md)，架构见 [ARCHITECTURE.md](ARCHITECTURE.md)，实验协议见 [PROTOCOL.md](docs/PROTOCOL.md)，硬件来源见 [HARDWARE.md](docs/HARDWARE.md)，扩展接口见 [INTERFACES.md](docs/INTERFACES.md)。

新资产接入见 [EXTENDING.md](docs/EXTENDING.md)，完整 Adapter 见 [PLUGINS.md](docs/PLUGINS.md)，结果入口见 [examples/README.md](examples/README.md)，完整验证记录及规模/数据证据见 [VALIDATION.md](docs/VALIDATION.md)。初版日志在 `artifacts/final_validation/`，5060 Ti 回归在 `artifacts/rtx5060ti-validation/`，自动接入的合同/端到端测试、成功失败示例与模板执行日志在 `artifacts/auto-integration-validation/`。
