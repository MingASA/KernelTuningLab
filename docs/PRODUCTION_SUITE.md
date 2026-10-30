# 生产算子与统一AI接入验收

**最新生产稳定性与交付状态见 [PRODUCTION_STABILITY.md](PRODUCTION_STABILITY.md)。** 本页保留此前验收过程；原Mixed数值限制、Split-K修复、CPU构建性能及当前运行profile以新报告为准。

开发分支`platform/production-suite-readiness`，基于旧研究分支检查点`dc6fda0`。旧研究原始数据、程序和失败保留；旧自动调度仍禁止恢复。本轮不做模型大小/算法排名矩阵，不训练模型，不因发现key而调用API。已读取的旧留出导出一律作为暴露的历史证据，不能给修改后的方法重新命名为盲测。

## 已核实的工程基线

原会话停止后的只读恢复已核对原始 summary、资源 receipt、逐候选记录和算法诊断；下文较早的构建过程保留为历史。五类后端已有以下 **FP16 定向实机证据**，不代表全部空间或 BF16 已实测，也不用于算法排名。

| 路径 | 执行源码 | 搜索观测 / 独立验收 | 已验证范围 |
|---|---|---:|---|
| SGLang / H800 | `f6b13c9` | 8 / 8 | 原生与显式、等长与不等长分页输入；关闭 PDL 的兼容路径 |
| FlexAttention / H800 | `d9eef4c` | 8 / 8 | Torch 2.6 兼容路径、dense/causal；实际生成的 W4S1/W4S3/W8S1 |
| vLLM MoE / H800 | `4ff5e2c` | 9 / 9 | 原生 0.8.5 兼容路径、均衡/不均衡路由、两个 M、两次真实 GEMM launch |
| GemLite W4/W2 / 4090 | `d765453` | 8 / 8 | 两个位宽分别覆盖显式 GEMM 和 Split-K；FP16/BF16 A16 源码兼容 profile 不提供旧 Torch 的 Dynamo 集成 |
| FLUTE W3 / H800 | `d765453` | 6 / 6 | 原生默认与多个实际结构模板；24 个 P32 模板构建，不能称 36 个模板全部通过 |
| GemLite W4/W2 默认 / 4090 | `cf8160e` | 2 / 2 | 两个位宽的真实原生默认选择/调优路径；默认内置调优成本独立于观测数 |
| FLUTE W3 / 4090 | `cf8160e` | 3 / 3 | 同一 sm89/sm90 构建的首次 sm89 验收：默认与多个显式结构模板 |

这些批次的资源门禁均 passed/eligible。MoE 首批发生外部任务冲突，2 次搜索和 2 次验收整批隔离，未复用缓存。SGLang 首批 PDL 编译失败、GemLite 两次导入失败、FLUTE P64 packing 不兼容及所有模型输出失败均保留；修复后的证据不覆盖首份失败。

成熟优化器在 `cf8160e` 的 4090 完整权威空间上各运行 16 次尝试：SMAC 为 13 次数值通过、3 次编译失败，14 次提议时模型已拟合；KT DE 为 8 次通过、8 次编译失败，完整静态可行集 1500 点、1 次原生调用、完成 1 代。两次独立严格验收通过。整体 `physical_gpu_validated=false` 保守反映存在失败候选，不能解释为没有 GPU 执行，也不能把 32 次尝试全部算作通过。

MoE 的两 Shape、两配置原始矩阵已直接导出 K=1/K=2 表。随后通过已有 Runner 在独立输入 seed 下完成 4 次真实分派，门禁通过；原矩阵没有重采。K 仍只表示配置数，不代表代码、packing、Graph 或物理二进制数量。

本地 Qwen3.8-27B 已真实生成完整 SDK 程序并收到开发反馈。C1 原文误用了不存在的返回状态 `success`；原文与一行 SDK 兼容修正独立保留，修正版契约和 4 条开发观测通过并冻结。C2 模型自己返回了与该修正版完全相同的代码，未人工修改 C2；它复用父版本的历史验收，不重测，也不宣称算法改进或新环境通过。

B 的自由输出失败与部分有效诊断独立保留。显式有限 grammar 变体已用真实 Qwen3-1.7B 完成两步模型选择和反馈：第一步采用前一尝试保留的真实模型响应，第二步新增 1 次真实调用；两步反馈均复用原常驻资源上下文中的有效记录，新增 Kernel 测量为 0。完整两点集合由模型 logits 决策，已揭示点排除；这证明有限任务上的反馈接续，不证明大空间 grammar 可扩展性、搜索胜负或新增计时结果。首次接续因 experiment_id 不同未命中缓存，在准备阶段前拒绝执行；失败、原响应和更正证据保留，未重复模型第一步或 Kernel 测量。

恢复基线 CPU 为 512 passed / 12 skipped；加入历史证据复用后完整回归为 **515 passed / 12 skipped**。可选原生优化器相关检查 17 passed（6 条上游 warning）。`cf8160e` wheel 在 `python -S` 下发现全部生产插件而不导入 Torch/Triton，并在独立无依赖 venv 中实际执行 CPU Runner，通过 2 次搜索观测和 1 次严格验收。最初直接解压 wheel 的试验仅向父进程添加 sys.path，子 worker 无法导入；错误保留，随后使用真正独立安装的 venv 验证。真实 API 调用仍为 0。

六条固定资源诊断已从原记录恢复并核对：无模型、常驻静默、推理后标准预热三种阶段各有两个配置；对应输入完整 hash 相同，全部 GPU 参考验证通过，常驻阶段具有生产者完成与阻止新提交的合作证明。它们来自模型格式错误但无外部进程冲突的两个 partial，按记录审计，不把原 command_failed 批次改为 eligible。没有重新测量或用它们推断共置收益。

当前远端原始证据已只读封存并回传本地恢复校验：H800 为 886 文件 / 29 SQLite，4090 为 605 文件 / 18 SQLite。全部 inventory SHA 和数据库 quick_check 通过；SQLite 采用只读源备份，原文件 SHA 前后相同。归档分别为 SHA256 `ea4ecf0cc426e35a2bf179ae26b2ac5121aaf849623f8436b9d0976b90de2118`、`1b9fef9cf0d7930e3a14ff66631d6bcb89018bc651a4dd1a14baedd8e5fcc319`。私有原文、主机信息和数据库不进入 Git；旧归档未覆盖，异机备份仍等待授权目标。

## 实施决策与准确版本

| 后端 | 固定来源 | 实际发现与边界 |
|---|---|---|
| GemLite W4/W2 | v.0.6.0.post2 = `f649e80e3014bc7ca8195573683c2c440f3bd94e`，Apache2 | 要求Triton≥3.6，import本身会读CUDA属性；CPU审计只执行原始host规则。声明GEMM/GEMM Split-K/GEMV/RevSplit-K分开。显式路径绕过Autotuner但保留原生derived heuristics和必要清零，原生默认允许其实际cache/pruner/tuner |
| FLUTE W3 | 0.4.2 = `9eb83a12d56949bbe7fe9c836ba97a67bd1e3761`，Apache2 | 原生成器有36个W3/36个W2/144个W4模板，结构参数映射真实ID。packing依赖TileP，不能跨布局复用。原StreamK workspace在128SM时为268437504bytes，已超过默认256MiB，验收规格必须显式给足预算。CUTLASS基线v3.4.1，私有include路径，不执行上游脚本中的sudo/chmod |
| FlexAttention | PyTorch v2.14.0 = `2b3ec34829036a65cd9d1398ea72a0167dc37470`，BSD3 | 主契约前向FP16/BF16、dense/causal；PRESCALE_QK固定。安全承诺由语义得出，FLASH和TRITON参数不混用。原Inductor单文件路径在该版本已移动到kernel/flex，不回退浮动main |
| SGLang paged decode | `1ab008f5b090e6595479ba57934e9211ab083565`，Apache2 | 原始分页索引/GQA/MQA和两阶段Split-KV保留；tile/warp/stage只暴露真实launch参数，分片数和workspace/归约一致 |
| vLLM MoE | v0.30.0 = `ced6857afa0ea7b2e3f0846a62e1394e90f15607`，Apache2 | requirements/cuda.txt固定Torch2.13.0；不假定需要Torch2.14。使用原始fused_experts_impl、override_config、try_get_optimal_moe_config。双专家GEMM共享配置，与受控Grouped GEMM不是同一范围 |

这些是来源与接口事实，不等于全部后端已构建/加载/GPU通过。支持状态在后续验收中分别更新。Marlin0.8.5原证据保留为专家基线，不再作为“大配置空间”主成果。

## 已实现的CPU基础

`SpaceView/ConfigSpace`沿用原接口。按需构造在每个可行叶子后重启，并对无产出的DFS遍历设局部节点上限，避免短批只消费首个高层分支；全局预算仍有效，分布明确为非均匀。完整静态构建及原生有限集合保持原语义。新增分支覆盖、参数顺序、稀疏联合约束、预算耗尽和恢复检查。

配置相关的packed只读缓冲现在可通过`ctx.register_prepared_input`登记，和逻辑输入一起做完整hash/alias检查；构建不能覆写原逻辑输入的基准hash。这样FLUTE不同TileP的packing及其他准备产物不会游离于正确性检查之外。可变reset/reduction workspace不能冒充只读输入。

API客户端支持标准Chat Completions与Responses wire格式，显式provider/model/endpoint/capabilities、有限重试、refusal/truncation处理、request ID/usage和未知费用。默认拒绝真实调用，在授权前不读取key；注入transport的测试明确是协议fixture。真实API验收仍待费用与出站授权，不阻塞CPU工作。

## 当前生产入口

- `kernel_tuning_lab.production.gemlite`：原生默认及四类显式算法家族，W4/W2是任务bits，绝不把bits改成性能参数。
- `kernel_tuning_lab.production.flute`：W3原生调优/缓存与显式结构模板，ID由结构参数反解，不作连续距离。编译集合在可靠build manifest缺失时保持unknown。
- `kernel_tuning_lab.production.vllm_moe`：生产原生配置表/默认回退与显式共享配置，完整专家路径含对齐、激活、加权归约及已标明的输出copy。原生nearest-M的插入顺序tie-break在导入导出中保留。

注册不导入生产依赖或启动GPU。执行时加载固定原生包/源码与二进制身份，不提供Torch计算替身。`native_sources.py`提取原始host函数或原封的JIT定义，数值计算始终使用原生实现。

- `production.flex_attention`：原生AUTO/TRITON/TRITON_DECODE/FLASH分别建资产，仅TRITON前向暴露tile/warp/stage；显式路径核对实际Inductor生成参数。增加固定Torch2.13.0兼容路径`cf30153c4c131c8164ee7798e5022d810682e2cb`，四个相关文件已与上游和本机安装逐字节核对，执行时继续核对，不将其报告为2.14运行结果。
- `production.sglang_decode`：原始paged索引、标准CUDA GQA/MQA wrapper、原生GPU Split-KV scheduler和两阶段merge；profile cap8明确记录，不冒充SGLang服务的全局默认。

## 有界CPU审计与实际构建

以下为FP16、group128及**规划128SM**下的源码静态审计，不是物理支持声明。依赖可用性独立报告；实际执行未绕过依赖或准入。每空间10秒/100000节点，报告Python tracemalloc峰值，不冒充RSS或显存。首个3秒Split-K构建未完成的记录保留，没有把前缀冒充完整集合。

| 显式空间 | 声明/原生候选 | 静态硬约束通过 | 编译/正确性 |
|---|---:|---:|---|
| GemLite GEMM | 2500（原max列表2000，补原专家使用的stage2） | 1500 | W4/W2 定向实机通过；不等于全空间通过 |
| GemLite Split-K | 7500 | 4500 | W4/W2 定向实机通过；不等于全空间通过 |
| GemLite GEMV / RevSplit-K | 150 / 120 | 150 / 120 | GPU待验 |
| FLUTE W3结构模板 | 36 | 24 | 当前逻辑 uint3 packer 仅 P32；24 个已构建，H800/4090 默认与结构候选通过 |
| Flex TRITON forward | 54 | 54 | H800 2.6 兼容路径定向实机通过 |
| SGLang paged decode | 135 | 135 | H800 原生/显式定向实机通过 |
| vLLM共享两阶段MoE | 2400 | 2400 | 0.30 官方扩展 CPU 加载；0.8.5 兼容路径 H800 实机通过 |

复现：`PYTHONPATH=. python scripts/audit_production_spaces.py --wall-seconds 10 --planning-sm-count 128 --output artifacts/production/audit.json`。按需构造是有界、非均匀的跨分支重启；native模板保留类别和结构属性，不制造连续ID距离。静态通过不代表编译或数值通过。

早期 FLUTE 审计的 32 个静态点尚未纳入逻辑 packer 的 P32 限制，原审计保留；补入已证实的准备约束后，在相同规划 Shape/128SM 上为 24 点。其余原始 CUDA/prepacked P64 模板仍是上游可表达能力，未宣称由当前逻辑 uint3 准备路径支持。

GemLite已从固定源码构建wheel。FLUTE在私有CUDA12.4/GCC11.2.1/Torch2.6环境用单CPU编译worker构建int3 ID `0,1,2,8,9,10`，目标sm89/sm90，420.38秒；6551376bytes，SHA256 `c8d43e3073cf8a104396dcf6d1fc463c0492ce7b4809073bddca89a244861464`。扩展在CUDA不可见时加载通过。原数值headers未改，只限制生成dispatch和私有CUTLASS include路径；6模板不等于6个物理编译Kernel。首个GCC8.5编译失败保留；复现入口`scripts/build_flute_subset.py`，不安装系统编译器。

vLLM0.30官方wheel在私有overlay安装，Torch2.13+cu132下两个stable-libtorch扩展CPU加载通过且CUDA未初始化。完整Python路径仍须完整可选依赖，导入失败逐次保留。安装、二进制加载、Python加载和GPU执行不能相互替代。

首轮目标机验收：FLUTE补齐私有jaxtyping0.3.2/wadler-lindig0.1.7后完整Python路径CPU加载通过，原缺依赖失败保留。H800首次SGLang四次独立验收均在编译准备时失败：Triton3.2缺少GDC/PDL符号，未得到可排名延迟。不是H800数值错误或硬件被判不支持。兼容层仅给私有language投影添加“被调用即static_assert失败”的JIT符号，绕过constexpr关闭分支仍被依赖hasher提前解析的问题；原数值函数、全局Triton和USE_PDL=False契约不改。CPU缺符号/cache-key回归通过，随后修复路径在独立 attempt 通过实机验收，旧失败没有改写。

## 身份、成本与优化器

SDK新增可选无参`KernelDefinition.execution_identity`：在**缓存查询之前**绑定实际安装源码、选定扩展SHA、软件版本、配置表及相关运行选项。当前≤1MiB文件直接取内容hash，大型不可变二进制按device/inode/size/mtime/ctime缓存，开销进入preflight/cache墙钟。回归验证“插件声明不变，原生文件变化→不命中旧测量”，包含同时间戳的表改写。不要在worker执行期间修改其依赖。

逻辑输入和config相关packing均做完整hash。FLUTE按TileP准备并单列准备成本；Split-K/StreamK初始化与归约均在完整GPU计划。MoE由原wrapper根据BM重建对齐元数据，两个GEMM共享配置；路由网络不计入。小用例增加“全零输出不能通过”的输入强度负对照，容差未放宽。

SMAC2.4.1/ConfigSpace1.2.2、Kernel Tuner1.5.0仍为可选依赖。KT DE默认持久callback执行原生算法，缺反馈时挂起；进程恢复时重建已揭示前缀。旧模式须显式`execution_mode=replay`。原生轨迹、恢复、重复反馈、host RNG和SDK契约测试通过；类别变量不作连续坐标，单点或不完整空间明确拒绝，不回退Random。1200点CPU合成问题B64：旧65次计算重放/2523次内部重复访问，新1次原生调用/33次内部缓存访问；约0.843秒与0.307秒。这是适配CPU成本，不能当作GPU算法胜利。提议、反馈、shutdown分别记账。

## API/本地AI与资源协议

A为已有专用模型，本轮不训练；B为直接决策；C为生成/迭代完整Searcher或规则。API/本地是部署方式。`ai/standard_provider@1.0.0`支持public_api、local_model、cpu_process，共用config/partial/region/有界space-query协议；不绑定GEMM六个字段、不要求大空间完整trie。平台补全、规范化和回退单独记录，不能归功于模型。旧完整小空间trie仍可显式选择。

API实现官方[Responses及结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)和Chat Completions薄协议，默认不读key、不联网；endpoint/model、付费和数据出站需显式授权。请求数上限为**每客户端**，多job需先计算总调用上限。原文、usage、request ID、重试及网络/总耗时保留，未知费用保持unknown。超时不伪装成模型格式错误重试。fixture不是API实测，本轮没有真实API授权。

LocalModelClient只用完整指纹的已有checkpoint，B/C共用任务。默认每trial/client只加载一次、权重常驻；同进程合作lease阻止新推理提交，由实际生产者同步，再覆盖候选准备、验证、warmup和测量。allocator/empty_cache不是静默证明。stage_offload为显式回退；C优先按整批生成/开发评测切换。另卡模式先独立获准和准入，再用ProcessCoordinator登记实际UUID及owned synchronize；不会自动占第二张卡。外部同卡HTTP服务缺可信合作协议时拒绝，不能凭HTTP、同UID或一次利用率当作隔离。

资源上下文进入测量和失败缓存；常驻OOM仅属于该上下文，不永久禁用架构。合作门禁CPU线程测试已通过；本轮“无模型/常驻静默/推理后warmup”的六条 GPU 诊断已按记录恢复，范围见上文，不用旧逐候选卸载实验冒充新协议。

驻留的B在同一v2 worker内推理和测量。监督worker是进程边界：父进程有本地模型时不能把其Python锁当成子worker的静默证明。当前显式拒绝这种跨worker交接；C的生成客户端在整个生成阶段结束后close，再开始整批开发评测。没有每候选卸载要求，也未实现远程/独立GPU服务的认证worker桥接。旧自定义HTTP入口即使声明separate_device，也不能用于CUDA测量。源码身份同时显式绑定已知input mixer与严格FP32参考辅助文件，避免只改辅助文件时命中旧缓存。

C完整程序入口：`ktlab ai-program submit/review/check/development/feedback/freeze/test/show`。原文、原程序、人工/静态编辑、源码SHA和父版本分别保存；必须明确可信审查后才执行。契约与开发评测用原v2 Runner/Store，冻结后不能继续同版本开发；读取test前持久化exposure。开发反馈不输出test路径/标签。受审进程不继承任意凭据环境，但不是恶意Python文件系统沙箱。CPU工作流fixture不算新LLM实测。

若迭代产物与冻结父版本的**实际执行程序 SHA、算法 ref、完整评测规格**完全一致，可用 `ktlab ai-program reuse --artifact <child> --source <frozen-parent> --spec <spec.json> --stage check`，再以 `--stage development` 关联已有历史证据。子版本仍需先 submit/review，并显式指定 parent；测试或已曝光任务不能复用为开发。复用绑定父 manifest、原规格、summary 和已有数据库的文件 SHA，后续验证拒绝变化。该入口不执行模型、Runner 或 GPU，明确报告新增测量为 0，不证明当前设备/环境已验证。之后可导出反馈并冻结子版本。

迭代子版本继承父版本的测试曝光任务ID，预算/seed变化不能把已暴露Shape重新标为盲测；与检查/开发已见任务重叠的test明确标为retest。仍需外部划分/来源审计，不能防御故意隐瞒父版本的可信调用者。截断、refusal等失败响应也保留原文、request ID和可用usage；本地加载/生成失败有独立日志，不产生候选性能。

## 原生规则、配置表与验收入口

`production.dispatch`把原ShapeDispatcher接到普通执行入口。export_table复用同实现/同协议PerformanceMatrix与SelectionResult，表绑定hardware/operator/kernel指纹、Config Budget和shape；未知输入拒绝。`production/config_table@1.0.0`通过原Runner实际调用显式候选。配置数、物理编译数、packing副本、代码字节、Graph分别保存，未知保留null。vLLM原生表读写保留nearest-M插入顺序tie-break；GemLite cache/pruner、FLUTE tuner仍独立保留，不能称为同K预算规则。

`python scripts/prepare_production_suite.py --backend gemlite --bits 2 --output artifacts/w2.json`生成有界规格；其余backend为flute/flex_attention/sglang_decode/vllm_moe。也可直接用`examples/production_suite/*.json`。先`ktlab research --spec ... --write-only --output ...`核对预算，在准入空卡及独立兼容环境执行同命令去掉write-only。每实现至多3研究观测，独立验收另计；原生默认内部调优可能远多于3，受worker期限约束并单列成本。相同命令恢复，失败原日志保留。FP16/BF16分别生成，不扩展性能排名矩阵。

私有证据入口`artifacts/PRODUCTION_SUITE_CURRENT.json`。旧PAUSE_REQUESTED仍有效，不自动恢复旧研究；异机备份无授权目标，单列待办。大型数据库、wheel、编译目录和服务器信息不入公开Git。
