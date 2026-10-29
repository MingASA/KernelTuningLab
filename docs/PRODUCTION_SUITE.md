# 生产算子与统一AI接入验收

开发分支`platform/production-suite-readiness`，基于旧研究分支检查点`dc6fda0`。旧研究原始数据、程序和失败保留；旧自动调度仍禁止恢复。本轮不做模型大小/算法排名矩阵，不训练模型，不因发现key而调用API。已读取的旧留出导出一律作为暴露的历史证据，不能给修改后的方法重新命名为盲测。

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
| GemLite GEMM | 2500（原max列表2000，补原专家使用的stage2） | 1500 | GPU待验 |
| GemLite Split-K | 7500 | 4500 | GPU待验 |
| GemLite GEMV / RevSplit-K | 150 / 120 | 150 / 120 | GPU待验 |
| FLUTE W3结构模板 | 36 | 32 | 编译6/36，GPU待验 |
| Flex TRITON forward | 54 | 54 | 生成配置解析CPU测试通过 |
| SGLang paged decode | 135 | 135 | 原始JIT定义CPU加载通过 |
| vLLM共享两阶段MoE | 2400 | 2400 | 官方两个原生扩展CPU加载通过 |

复现：`PYTHONPATH=. python scripts/audit_production_spaces.py --wall-seconds 10 --planning-sm-count 128 --output artifacts/production/audit.json`。按需构造是有界、非均匀的跨分支重启；native模板保留类别和结构属性，不制造连续ID距离。静态通过不代表编译或数值通过。

GemLite已从固定源码构建wheel。FLUTE在私有CUDA12.4/GCC11.2.1/Torch2.6环境用单CPU编译worker构建int3 ID `0,1,2,8,9,10`，目标sm89/sm90，420.38秒；6551376bytes，SHA256 `c8d43e3073cf8a104396dcf6d1fc463c0492ce7b4809073bddca89a244861464`。扩展在CUDA不可见时加载通过。原数值headers未改，只限制生成dispatch和私有CUTLASS include路径；6模板不等于6个物理编译Kernel。首个GCC8.5编译失败保留；复现入口`scripts/build_flute_subset.py`，不安装系统编译器。

vLLM0.30官方wheel在私有overlay安装，Torch2.13+cu132下两个stable-libtorch扩展CPU加载通过且CUDA未初始化。完整Python路径仍须完整可选依赖，导入失败逐次保留。安装、二进制加载、Python加载和GPU执行不能相互替代。

首轮目标机验收：FLUTE补齐私有jaxtyping0.3.2/wadler-lindig0.1.7后完整Python路径CPU加载通过，原缺依赖失败保留。H800首次SGLang四次独立验收均在编译准备时失败：Triton3.2缺少GDC/PDL符号，未得到可排名延迟。不是H800数值错误或硬件被判不支持。兼容层仅给私有language投影添加“被调用即static_assert失败”的JIT符号，绕过constexpr关闭分支仍被依赖hasher提前解析的问题；原数值函数、全局Triton和USE_PDL=False契约不改。CPU缺符号/cache-key回归通过，修复后的GPU路径仍须独立验收，不能把旧失败改成成功。

## 身份、成本与优化器

SDK新增可选无参`KernelDefinition.execution_identity`：在**缓存查询之前**绑定实际安装源码、选定扩展SHA、软件版本、配置表及相关运行选项。文件hash按size/mtime/ctime缓存，开销进入preflight/cache墙钟。新增回归验证“插件声明不变，原生文件变化→不命中旧测量”。不要在worker执行期间修改其依赖。

逻辑输入和config相关packing均做完整hash。FLUTE按TileP准备并单列准备成本；Split-K/StreamK初始化与归约均在完整GPU计划。MoE由原wrapper根据BM重建对齐元数据，两个GEMM共享配置；路由网络不计入。小用例增加“全零输出不能通过”的输入强度负对照，容差未放宽。

SMAC2.4.1/ConfigSpace1.2.2、Kernel Tuner1.5.0仍为可选依赖。KT DE默认持久callback执行原生算法，缺反馈时挂起；进程恢复时重建已揭示前缀。旧模式须显式`execution_mode=replay`。原生轨迹、恢复、重复反馈、host RNG和SDK契约测试通过；类别变量不作连续坐标，单点或不完整空间明确拒绝，不回退Random。1200点CPU合成问题B64：旧65次计算重放/2523次内部重复访问，新1次原生调用/33次内部缓存访问；约0.843秒与0.307秒。这是适配CPU成本，不能当作GPU算法胜利。提议、反馈、shutdown分别记账。

## API/本地AI与资源协议

A为已有专用模型，本轮不训练；B为直接决策；C为生成/迭代完整Searcher或规则。API/本地是部署方式。`ai/standard_provider@1.0.0`支持public_api、local_model、cpu_process，共用config/partial/region/有界space-query协议；不绑定GEMM六个字段、不要求大空间完整trie。平台补全、规范化和回退单独记录，不能归功于模型。旧完整小空间trie仍可显式选择。

API实现官方[Responses及结构化输出](https://developers.openai.com/api/docs/guides/structured-outputs)和Chat Completions薄协议，默认不读key、不联网；endpoint/model、付费和数据出站需显式授权。请求数上限为**每客户端**，多job需先计算总调用上限。原文、usage、request ID、重试及网络/总耗时保留，未知费用保持unknown。超时不伪装成模型格式错误重试。fixture不是API实测，本轮没有真实API授权。

LocalModelClient只用完整指纹的已有checkpoint，B/C共用任务。默认每trial/client只加载一次、权重常驻；同进程合作lease阻止新推理提交，由实际生产者同步，再覆盖候选准备、验证、warmup和测量。allocator/empty_cache不是静默证明。stage_offload为显式回退；C优先按整批生成/开发评测切换。另卡模式先独立获准和准入，再用ProcessCoordinator登记实际UUID及owned synchronize；不会自动占第二张卡。外部同卡HTTP服务缺可信合作协议时拒绝，不能凭HTTP、同UID或一次利用率当作隔离。

资源上下文进入测量和失败缓存；常驻OOM仅属于该上下文，不永久禁用架构。合作门禁CPU线程测试已通过；“无模型/常驻静默/推理后warmup”的新GPU验收待空卡，旧LLM证据不能冒充新驻留协议已测。

驻留的B在同一v2 worker内推理和测量。监督worker是进程边界：父进程有本地模型时不能把其Python锁当成子worker的静默证明。当前显式拒绝这种跨worker交接；C的生成客户端在整个生成阶段结束后close，再开始整批开发评测。没有每候选卸载要求，也未实现远程/独立GPU服务的认证worker桥接。旧自定义HTTP入口即使声明separate_device，也不能用于CUDA测量。源码身份同时显式绑定已知input mixer与严格FP32参考辅助文件，避免只改辅助文件时命中旧缓存。

C完整程序入口：`ktlab ai-program submit/review/check/development/feedback/freeze/test/show`。原文、原程序、人工/静态编辑、源码SHA和父版本分别保存；必须明确可信审查后才执行。契约与开发评测用原v2 Runner/Store，冻结后不能继续同版本开发；读取test前持久化exposure。开发反馈不输出test路径/标签。受审进程不继承任意凭据环境，但不是恶意Python文件系统沙箱。CPU工作流fixture不算新LLM实测。

迭代子版本继承父版本的测试曝光任务ID，预算/seed变化不能把已暴露Shape重新标为盲测；与检查/开发已见任务重叠的test明确标为retest。仍需外部划分/来源审计，不能防御故意隐瞒父版本的可信调用者。截断、refusal等失败响应也保留原文、request ID和可用usage；本地加载/生成失败有独立日志，不产生候选性能。

## 原生规则、配置表与验收入口

`production.dispatch`把原ShapeDispatcher接到普通执行入口。export_table复用同实现/同协议PerformanceMatrix与SelectionResult，表绑定hardware/operator/kernel指纹、Config Budget和shape；未知输入拒绝。`production/config_table@1.0.0`通过原Runner实际调用显式候选。配置数、物理编译数、packing副本、代码字节、Graph分别保存，未知保留null。vLLM原生表读写保留nearest-M插入顺序tie-break；GemLite cache/pruner、FLUTE tuner仍独立保留，不能称为同K预算规则。

`python scripts/prepare_production_suite.py --backend gemlite --bits 2 --output artifacts/w2.json`生成有界规格；其余backend为flute/flex_attention/sglang_decode/vllm_moe。也可直接用`examples/production_suite/*.json`。先`ktlab research --spec ... --write-only --output ...`核对预算，在准入空卡及独立兼容环境执行同命令去掉write-only。每实现至多3研究观测，独立验收另计；原生默认内部调优可能远多于3，受worker期限约束并单列成本。相同命令恢复，失败原日志保留。FP16/BF16分别生成，不扩展性能排名矩阵。

私有证据入口`artifacts/PRODUCTION_SUITE_CURRENT.json`。旧PAUSE_REQUESTED仍有效，不自动恢复旧研究；异机备份无授权目标，单列待办。大型数据库、wheel、编译目录和服务器信息不入公开Git。
