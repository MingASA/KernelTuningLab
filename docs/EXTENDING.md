# 自动接入：提交实现，按配置做实验

项目书算子与更大工作负载已通过本接口接入，见 [PROPOSAL_OPERATORS.md](PROPOSAL_OPERATORS.md)。`SpaceContext` 还提供环境及可配置工作区预算；空间支持 `resolver(context)` 和 `constraint(config,context)`，兼容原三/四参数形式。`ParameterSpec.description/unit` 描述不同类型的旋钮。算子可用 `validation_case_factory` 按语义参数生成验证输入，编译资源可用 `ctx.record_compiled_kernel` 统一记录/检查。

本入口使用 **v2 实验配置和公开 SDK**。算子、Kernel、搜索算法独立注册，实验文件引用已有资产；新增资产无需修改平台、内置注册名单或 CLI 参数列表。v1 `benchmark`、已有 Adapter 和历史数据继续可用。本轮不含 Web UI。

## 1. 从可运行模板开始

```bash
ktlab scaffold kernel --operator gemm@1.0.0 --runtime python --output my_kernel
ktlab check --spec my_kernel/experiment.json --output artifacts/my-kernel-check
ktlab run --spec my_kernel/experiment.json --output artifacts/my-kernel-run

ktlab scaffold algorithm --output my_algorithm
ktlab run --spec my_algorithm/experiment.json --output artifacts/my-algorithm-run
```

模板生成 `assets.py`、`experiment.json` 和简短说明，初始实现可直接执行。Kernel 模板复用已工作的实现、输入服务和默认空间；替换 `build` 后继续运行同一套检查。算法模板提供满足预算、历史和合法性约束的有限遍历起点。目标文件已存在时拒绝覆盖。

研究多个算子/实现、Shape、dtype、seed、预算并与基线比较时，直接使用
`ktlab research --spec study.json --output artifacts/study`。搜索插件仍使用本 SDK，
无需接入新的 Runner 或存储；通用矩阵配置、自动图表、成本/失败分析和断点恢复见
[RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md)。

CUDA 模板使用 `--runtime cuda`；在有对应依赖和 GPU 的机器上执行。无 CUDA 时返回待验证状态。5060 Ti 安装步骤见 [RTX5060TI.md](RTX5060TI.md)。

不使用模板也可以。独立文件只需 `from kernel_tuning_lab.sdk import ...`，并导出：

```python
ASSETS = (operator_definition, kernel_definition, algorithm_definition)
```

仅新增 Kernel 时只导出 Kernel；仅新增算法时只导出算法。也支持 `register_assets(catalog)`。安装包可声明 `[project.entry-points."kernel_tuning_lab.assets"]`，值指向资产、资产序列、模块或接收 catalog 的注册函数。文件插件路径相对于实验配置文件解析，模块名按 Python 导入规则解析。插件是显式加载的可信 Python 代码，执行隔离用于限制故障和时间，不是安全沙箱。

## 2. 资产引用与发现

```bash
ktlab assets list
ktlab assets list --plugin my_kernel/assets.py --kind kernel
ktlab assets show --kind operator --ref gemm@1.0.0
```

三类资产均有 `id`、`version`、`api_version=1`、`metadata`、`source_files`。规范引用为 `id@version`。省略版本仅在同名资产只有一个版本时可用；存在多个版本时报歧义。相同版本和内容重复加载幂等，同名同版本不同内容在同一目录中拒绝覆盖。

内容指纹包含声明、入口代码及其源文件。额外导入的实现助手文件须放进 `source_files`；平台不会猜测任意 Python 导入图。运行生成 `assets.lock.json`，保存精确版本、内容指纹、声明及解析后的空间。修改实现、参考、容差或约束会改变相应测量缓存身份。算法版本和 options 绑定 trial，兼容的 Kernel 测量可跨算法复用。

平台自动记录 Python、Torch/CUDA、Triton 和驱动等已支持运行时的版本。使用其他编译器、外部模型或数据时，应将其版本及身份作为实现特有信息放入资产 `metadata`，将本地依赖文件列入 `source_files`，这些信息参与指纹。平台不自动解析或安装任意外部工具链的传递依赖。

## 3. Kernel 提交内容

`KernelDefinition` 的必要字段：

| 字段 | 内容 |
| --- | --- |
| `id`, `version` | 独立实现身份 |
| `operator` | 算子资产引用，例如 `gemm@1.0.0` |
| `build(ctx, **parameters)` | 准备/编译实现，返回无参数 callable 或 `ExecutionPlan` |
| `space` | `SpaceDefinition`，声明 Kernel 的完整可调参数与有限域 |
| `runtime` | `python` 或 `cuda` |

可选字段包括 `parameter_map`、`dtypes`、`architectures`、`supports(hardware, operator, shape)`、`source_files`、`claims` 和 `metadata`。纯 CUDA/Torch 实现不强制依赖 Triton；需要 Triton 的实现声明 `metadata={"requires_triton": True}`。

`build` 只接收已声明参数。参数默认按同名关键字传入，`parameter_map={"tile": "tile_size"}` 可将空间的 `tile` 映射到入口参数 `tile_size`。缺映射、重名映射、额外参数和入口签名不一致会在预检中失败。

`ctx` 提供：

| 属性/方法 | 用法 |
| --- | --- |
| `inputs`, `outputs` | 命名输入和输出，算子负责约定；GEMM 为 `a/b → out`，归约为 `x → out` |
| `shape`, `operator`, `hardware` | 输入尺寸、dtype/语义参数、实际硬件 |
| `config`, `seed`, `runtime` | 当前完整配置、输入随机种子、执行运行时 |
| `torch`, `device` | CUDA 时惰性加载的 Torch 与设备编号 |
| `allocate_workspace(name, shape, dtype, fill=0)` | 在 build/setup 中声明一次性工作区，默认总限额 256 MiB |
| `workspace` | 已声明工作区字典 |

完整同类 Kernel 示例见 [cpu_kernels.py](../examples/extensions/cpu_kernels.py)，它没有实现 Adapter、输入生成、参考或计时。CPU 结果是真实 CPU 调用测量，不是 GPU 基线或 Mock 性能。

### 多步骤与初始化

```python
return ExecutionPlan(
    steps=(ExecutionStep("partials", compute_partials),
           ExecutionStep("reduce", reduce_partials)),
    initialize=clear_workspace,
    setup=one_time_setup,
)
```

`setup` 和工作区分配仅执行一次，单列成本；`initialize` 在每次完整调用前执行。正确性检查、warmup 和计时调用同一条 `initialize → 有序 steps` 路径，必需的清零、重置、拷贝应放在这里。编译完成后拒绝新的平台工作区声明。每个 step 的返回值可为输出字典，也可写入 `ctx.outputs` 后返回 None。

CUDA 计时在完整流程前后同步，以 `perf_counter` 记录端到端延迟，包含声明的主机步骤和 GPU 调用。该模式与旧 Adapter 的单 Kernel CUDA Event 模式分开记录。示例中的 `demo/gemm_pipeline@1.0.0` 实际执行两阶段 split-K 与汇总，重复核对会发现缺少 reset 的状态污染。

默认仍为上述完整调用计时。可捕获的 CUDA 实验可显式设置
`"timing_mode": "cuda_graph", "cuda_graph_batch_size": 512`，测量包含
initialize 与所有 GPU steps 的批量 replay，并记录不同 measurement mode。
Python/host callbacks 不随 replay 重复；capture/准备成本另计，最终实际计时输出仍核对。
适用边界、缓存身份和极短 kernel 的批次误差见 [PROTOCOL.md](PROTOCOL.md)。

## 4. 算子公共能力

同类新 Kernel 选择已有算子即可。当前内置 GEMM、Batched GEMM、Softmax 和非仿射 LayerNorm，提供 seed 输入、边界 Shape、常量/offset 验证输入、独立参考和 dtype 容差。

新算子提交 `OperatorDefinition`：

- `shape_dims`、`dtypes`、默认语义 `parameters`、每种 dtype 的 `(atol, rtol)`。
- `generate_inputs(ctx, seed)`：返回命名输入并使用 `ctx.allocate_output` 分配命名输出。
- `reference(ctx)`：返回独立的命名参考输出；不得调用待验收 Kernel。
- `validation_shapes`：可复现的小尺寸、尾维或退化边界案例。
- 可选 `validation_case_factory(parameters, requested_shapes)`：根据合并后的算子默认参数和实验参数，以及请求形状的 tuple，返回运行时边界案例 tuple；每项为 `ShapeSpec` 或维度字典。平台检查正维度、完整维度名和重复项，错误会使预检失败。声明中的静态 `validation_shapes` 继续用作 scaffold 默认形状。
- 可选 `validate(ctx)`：检查形状、dtype、参数及运行时适用性；该回调可能在输入生成前调用。

参考实现和容差也参与版本与内容绑定。自定义 AXPY 的完整算子与 Kernel 见 [cpu_kernels.py](../examples/extensions/cpu_kernels.py)，配置见 [custom_operator.json](../examples/extensions/custom_operator.json)。本轮用它证明新算子接入，不代表已补齐立项书中的所有生产算子。

## 5. 有效配置空间

```python
SpaceDefinition({
    "mode": ParameterSpec(("forward", "reverse")),
    "tile_size": ParameterSpec((1, 2, 4)),
    "reverse_chunk": ParameterSpec((1, 2),
        active_when={"mode": ("reverse",)}, default=1),
}, constraints=(hardware_or_input_constraint,))
```

参数支持 integer、有限 real、boolean、categorical；类型可自动推断。所有取值必须是显式、非空、不重复的有限序列。没有可调参数的固定实现可声明 `SpaceDefinition({})`，对应唯一配置 `{}`，无需伪造调参维度。连续无界域、无限生成器和条件依赖环会被拒绝。

条件参数在不激活时仍保留完整参数键，值必须为声明的 default。条件依赖向上继承，不会静默删参。约束签名是 `(config, hardware, operator, shape)`，返回 None 表示合法，返回字符串说明拒绝原因，也可返回 `ValidationResult`。

显式结构关系可用 `rules=(Relation("x", "==", "y"), Forbidden({"mode":"a", "split":4}))`，从 SDK 导入这两个类型。它们支持部分赋值检查和依赖传播；任意 Python 回调不自动解析，未赋齐参数时保留 unknown。`space.normalize` 只统一可证明等价的不激活默认值。原生后端的有限候选用 `SpaceDefinition.candidates` 或 `ConfigSpace.from_configs` 声明，候选集合不等于参数域的笛卡尔积。

算法可使用 `build_feasible(..., budget=SpaceBuildBudget(...))`、`complete(partial, ..., count=...)`、`sample_feasible`、`validate_partial` 和联合 `neighbors(min_changed=2,max_changed=2)`。只读视图也支持这些方法。完整构建结果为 complete/proven_empty；节点、时间或估算存储预算耗尽返回 budget_exhausted，排除已揭示点后的 exhausted 不证明原空间为空。条件/结构化空间的验收探测也使用有界依赖构造。完整集合抽样可均匀；构造采样是非均匀的，分布与拒绝原因写入 diagnostics。存储字节预算是估算值，不是 OS 级 RSS 硬限制，外层仍有进程期限。

内置搜索默认在 50,000 点以下尝试有界完整静态构建；超过预算则明确使用算法自己的有界活动池，余量低于池大小的四分之一时补充。Surrogate 每轮最多评分 4,096 点的新子集；日志披露近似。外部插件直接向权威空间提出配置，不受该池过滤。静态合法不等于实际编译或数值通过。共享内存估算是带校准范围的 advisory，实际编译资源和精确 workspace 上限仍独立核验。审计、原生优化器和生产实例见 [SPACE_PRODUCTION.md](SPACE_PRODUCTION.md)。

实验的 `space` 仅缩小 Kernel 域，不得加新参数或扩大值域。硬件和 Shape 约束惰性组合，配置最终执行前再次校验。空间的 `cardinality` 是约束前的笛卡尔积上界。平台只探测有界数量的点，稀疏大空间中找不到合法点会明确报告“预算内未发现”，不会冒充数学上的全空间为空。

研究限制只用于用户请求的形状，记录为 `requested_space_scope="experiment_restrictions"`。自动生成的 `validation_only` 边界案例使用该 Kernel 的完整声明域，记录为 `extra_space_scope="kernel_defaults"`，并继续受自身形状、硬件、上下文 resolver 和合法性约束限制。例如，生产形状可限制只研究 TMA 或 Split-K，tiny 边界案例仍可选用普通路径；请求形状无效或边界案例找不到合法配置时，预检仍明确失败。lock 和 preflight 记录这些范围，每个 case manifest 的 `space.resolution.space_scope` 标明实际来源。

## 6. 搜索算法

提交 `SearchAlgorithmDefinition(id, version, factory, capabilities)`。`factory(seed=..., **options)` 返回对象，其方法为：

```python
suggest(hardware, operator, shape, space, history, budget, batch_size=1)
```

返回至多 `batch_size` 个新的合法 `KernelConfig`。零预算返回空，不得重复已尝试配置；失败历史也属于已尝试。`history` 仅有已揭示观测和 attempted IDs，没有未测量配置的性能。`space` 是稳定身份的只读视图，提供 `domains`、`cardinality`、`fingerprint`、`config_at`、`sample_configs`、`validate`、`neighbors` 和遍历。大空间应使用采样或有界探测，不能隐式展开全部候选。

`SearchCapabilities` 声明参数类型、条件/约束支持、必要参数、最大维数和可选最大 cardinality。平台先做能力匹配，再用真实解析空间、零预算、单点和支持的混合空间进行多轮合成交互，检查返回类型、批量大小、合法性、重复及预算行为。正式搜索仍逐轮执行相同保护；不兼容时拒绝，不替换成默认算法。

完整独立示例见 [search_algorithm.py](../examples/extensions/search_algorithm.py)。其每轮探测量有上限，极稀疏空间可能提前耗尽建议；日志会保留停止原因。

工厂的 `options` 可定义自己的初始化参数。内置 transfer 可用 `{"asset":"transfer@1.0.0","options":{"training_database":"/absolute/path/source.sqlite"}}` 读取仅 train 且正确性已核对的历史记录；目标 case 仍受训练隔离检查。SDK 也导出 `SearchHistory`、`MeasurementRecord` 与 `ExperimentStore`，供需要历史数据的算法使用。

有状态原生优化器可选实现 `bind_trial(output_dir=...,identity=...)`、`observe(record,cache_hit=...)`、`runtime_identity` 和 `diagnostics`。最后一条观测也会反馈；重复原生提议由适配器按已揭示记录内部处理，不能替换成无关配置。版本、模型/代码 SHA 与运行时身份参与 trial 身份；工厂、提议、反馈和外部推理成本计入真实 wall，并单列在报告。失败罚值只能留在优化器内部，不能写成测得 latency。可信 Python 插件和生成程序均不构成恶意代码沙箱。

## 7. 配置与预算

最完整的 CPU 组合实验见 [cpu_gemm.json](../examples/extensions/cpu_gemm.json)：两个实现 × 两个算法 × 两个 Shape，共八个 trial。插件路径相对于 JSON 文件；同一资产可反复组合。

| 配置项 | 作用域与默认值 |
| --- | --- |
| `preflight_seconds` | 整个加载、静态和算法合成阶段的硬时间上限，30 秒 |
| `validation.max_configs` | 全实验验收执行次数上限，8；失败和按声明接受的实际调用都计入 |
| `validation.max_cases` | 每个算子最多追加的边界 Shape 数量，2 |
| `validation.max_wall_time_s` | 独立验收阶段的硬时间上限，30 秒 |
| `validation.seed` | 可复现配置抽样和输入 seed 的起点，17 |
| `budget.max_evaluations` | 每个 Kernel × 算法 × 请求 Shape 的候选观测上限，8，包含失败和缓存观测 |
| `budget.max_wall_time_s` | 每个 trial 的完整搜索硬时间上限，30 秒 |

验收与研究预算分别记录。验收先按 case 轮询再增加样本，保留探测拒绝统计。覆盖不足为 `partial`，缺设备为 `deferred`。抽样通过只证明列出的配置、输入和 seed；范围外的正式搜索配置仍逐个核对。研究预算为零不调用研究算法或 Kernel。v2 的 `observations` 统计已揭示候选，缓存命中消耗同一观测预算；`evaluations` 单列新物理尝试，`cache_hits` 单列复用。这样多算法组合的可见候选预算一致，缓存仍能节省实际测量成本。旧 v1 直接 Runner 的物理入场预算语义保持原样。

父进程监督子进程，涵盖导入、生成器、参考、编译、Kernel 和算法调用。超时终止工作进程并保留阶段、正在执行的候选和日志。监督检查存在毫秒级调度粒度；杀掉进程不承诺复位失去响应的 GPU 驱动。

## 8. 正确性声明与证据等级

`declare_correctness(kernel, operator_definition, id=..., source=..., evidence=..., scope=...)` 生成绑定 Kernel/算子版本及内容的声明。`scope` 必须显式包含 runtimes、dtypes、architectures、shapes、configs、parameters；支持显式 `"*"`、枚举、数值 min/max，也可限制环境精确版本、硬件名称、输入 seed 和有效期。

默认 `trust_claims=true` 允许匹配范围跳过数值参考比较；仍实际调用并检查输出结构和有限性。范围不匹配或旧内容声明会记录原因并回到参考检查；`trust_claims=false` 强制独立核对。完整示例见 [claimed_kernel.py](../examples/extensions/claimed_kernel.py) 与 [scoped_claim.json](../examples/extensions/scoped_claim.json)，其中 M=5 越出声明范围。

| 证据 | 结果记录 |
| --- | --- |
| 平台参考比较通过 | `status=verified`，CPU/GPU 独立标识，实际误差、容差、参考指纹和重复执行 |
| 接受提交者声明 | `status=declared`、`correctness_origin=submitter_claim`、声明来源/范围/指纹，两个 physical verified 标志均 false |
| 失败 | 独立 config/compile/correctness/runtime 状态与原因，不能成为成功观测 |

声明本身不生成测量记录。只有实际调用后的性能才为 `measured`。声明性能可作为本次显式信任实验的建议，并明确标记；默认 Replay、L3 和已验证性能报告排除它，trial 同时给出 `best_verified_*`。数字验收失败或输出结构失败覆盖声明，不能由声明把失败改成通过。

## 9. 运行与失败示例

```bash
ktlab run --spec examples/extensions/cpu_gemm.json --output artifacts/extensions/cpu
ktlab run --spec examples/extensions/custom_operator.json --output artifacts/extensions/axpy
ktlab run --spec examples/extensions/conditional_space.json --output artifacts/extensions/conditional
ktlab run --spec examples/extensions/scoped_claim.json --output artifacts/extensions/claims

ktlab check --spec examples/extensions/failure_algorithm.json --output artifacts/extensions/bad-algorithm
ktlab run --spec examples/extensions/failure_wrong_output.json --output artifacts/extensions/bad-output
ktlab check --spec examples/extensions/failure_space.json --output artifacts/extensions/bad-space
ktlab check --spec examples/extensions/failure_capabilities.json --output artifacts/extensions/bad-capability

ktlab run --spec examples/extensions/gpu_gemm.json --output artifacts/extensions/5060ti
```

退出码：0 通过；2 拒绝或执行失败；3 待设备验证或验收覆盖不足；124 硬超时；130/143 为 SIGINT/SIGTERM 中断。CPU 示例在没有第三方 SDK 的 Python 环境也可执行。GPU 示例选择 5060 Ti 16GB；在 H200/H800/4090 上修改硬件选择后重新验证和搜索。

## 10. 结果入口

生产扩展可设置`KernelDefinition.execution_identity`，一个CPU安全的无参回调，返回JSON兼容的实际安装源码/二进制/选择表身份。v2在缓存查询前调用；构建后才记录binary SHA不能防止旧缓存跨二进制复用。不要在回调中初始化GPU或运行优化器，声明其辅助源码依赖。详情及真实生产插件见[PRODUCTION_SUITE.md](PRODUCTION_SUITE.md)。

外部AI的config/partial/region/query与受审完整Searcher工作流见[生产套件示例](../examples/production_suite/README.md)。接口fixture、真实本地模型、真实付费API是不同证据；核心包不要求模型SDK。生成代码必须受审，环境转发不等于恶意代码沙箱。

| 文件 | 内容 |
| --- | --- |
| `preflight.json` | 资产/入口/空间/算法实际预检结果，含失败与 deferred 原因 |
| `assets.lock.json` | 解析后的版本、源码和参考指纹、声明及有效空间 |
| `experiment.json`, `source_experiment.json` | 可复跑的已解析插件路径配置与原输入配置；源位置见 `experiment_origin.json` |
| `acceptance.json`, `validation_report.json` | 硬预算、抽样覆盖与证据等级 |
| `validation_records.jsonl` | 本轮验收的实际执行记录 |
| `trials.json`, `trials/` | 各组合的推荐、已验证推荐、预算、缓存、停止原因和失败事件 |
| `measurements.sqlite`, `measurements.jsonl` | 包含历史的完整测量日志，断点缓存继续保留 |
| `run_summary.json`, `worker.log` | 本轮状态、阶段统计及工作进程输出 |
| `run_metadata.json`, `attempts/` | 本轮运行 ID、上一轮位置及保留的历史验收/日志/配置 |
| `report.md` | 当前组合、延迟、证据等级和归档链接的简明报告 |
| `counterexamples.json` | 从历史日志恢复并使声明失效的实际正确性反例 |
| `interrupted_attempt.json` | 超时、中断或 worker 崩溃阶段及最后一个正在执行的候选 |

重跑先将旧的验收快照和日志保留到 `attempts/`，撤销本轮完成标记，再执行新检查；累计测量日志保留。输出目录由 OS 锁保护，同目录并发调用被拒绝。配置在父进程冻结，重复 JSON 键及非有限值被拒绝；被终止候选不作为完成缓存，重跑可重新测试。执行时默认输入只读、输出无输入别名；计时结束后还会核对实际输出。详细行为见 [HARDENING.md](HARDENING.md)。源文件发生变更后须使用对应版本重新验收；将 5060 Ti 的 CPU/GPU 结果上传到 Hopper 不会替代目标服务器复验。
