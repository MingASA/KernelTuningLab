# 可行空间、成熟优化器、AI接口与原生W4A16

本轮已打通完整声明空间→原生优化器→现有v2 Runner，以及真实vLLM Marlin→原生规则→有限工作负载查表分派。核心CPU **415 passed / 12 skipped**；可选集成 **33 passed**，其中8项核对原生优化器轨迹、反馈及恢复。独立无Torch/Triton的wheel环境完成CPU研究与资源发现。[公开数据](data/space-production/)与本页的有限覆盖边界共同解释结果。

工作在`research/space-production-integration`，起点`d7eaca5`；没有合并main。功能复现快照为 **`46a33442f307fee1451ff4ef6cfef748a759be79`**。主要GPU计算固定 **`8171d3ae7e339cf8f9819bceb3ead9b2d73af363`**，资源Pilot为`6722390`；最终核心兼容性回归单独使用 **`f2518bcbeb5d09f23ee91f42ea48dd11452dff40`**。外部算法/表/规格另有commit与SHA，不把不同快照合并排名。旧5060 Ti及跨GPU五份归档SHA均复核一致，旧192条Anchor未重测。

## 1. 环境、预算与真实覆盖

本轮新GPU数据来自获准空闲的单张 **RTX4090/sm89/24GB**：驱动550.54.15，Python3.11.11、Torch2.6.0+cu124、Triton3.2.0、NumPy2.2.2。原生C++使用现有CUDA toolkit12.4.131/GCC11.2.1；driver、Torch runtime、编译toolkit分别记录。SMAC环境额外固定ConfigSpace1.2.2、SciPy1.15.3、scikit-learn1.6.1、pandas2.2.3、pymoo0.6.1.6。没有升级全局栈或修改原环境。

本地5060 Ti有显示占用，严格准入未启动本轮GPU任务；H800检查时八卡均有大额显存占用，未加入任务。H800的官方vLLM0.8.5组件只完成CPU加载，不能称GPU已通过。原有三卡Anchor仍有效，本轮不产生新的Hopper/多GPU性能结论。

统一主协议：FP16，Graph **128次完整GPU执行计划/批**，warmup10，15个原始samples。prepare、参考、一次性setup、Graph capture不在latency内，但都计候选/实验墙钟；initialize、逐调用workspace reset、所有GPU步骤均在Graph内。host查表/选择不由Graph重复执行，单独计proposal成本。严格参考比较、输入完整性、输出检查与既有容差保持不变，详见[PROTOCOL](PROTOCOL.md)。

| 实验 | 新研究评估 / cache观测 | 独立验收评估 | 结果 |
|---|---:|---:|---|
| 资源Pilot，12定向候选 | 10 / 0 | 12 | 10个配置通过，2个验收OutOfResources保留 |
| 开放W4，2 Shape×2实现×5方法×B8/B32 | 614 / 186 | 40 | 800观测完成；611条新评估成功，3条编译资源失败，缓存后共5次失败观测 |
| 原生Marlin Smoke | 12 / 0 | 6 | 全通过 |
| 六Shape生产矩阵，2 input seed | 60 / 12 | 36 | 72/72通过，复用原Smoke，不重测 |
| AI开发数据，初次+修复恢复 | 72 / 72 | 6+9 | 首次独立验收预算不足保留；修复后72cache/0new通过 |
| MoE grouped/persistent×Split-K1/2 | 4 / 0 | 4 | 全通过，USE_TMA=False |
| 初步L3所选配置复核 | 6 / 0 | 6 | 全通过；此项是离线映射后的执行 |
| 两Shape共同输入，受控+原生五实现 | 16 / 0 | 10 | 全通过，逻辑输入hash跨实现一致 |
| 冻结v1/v2的两Shape测试 | 32 / 0 | 4 | 全通过，测试前已冻结两份源文件 |
| 独立确认：已测近优与Heuristic配置 | 16 / 0 | 16 | 全通过，seed7/19 |
| 最终核心回归：W4 fused/pipeline、原生、MoE | 21 / 0 | 4 | 全通过；原生4点与MoE单点正常耗尽 |
| 真实ShapeDispatcher→原生调用，固定FP32-reduce请求 | 6 / 0 | 6 | 六个查表、实际native flags及GPU结果均核对 |

合计 **869次新研究候选评估、159次独立验收评估、270次cache观测**；评估不是单次Kernel launch。研究记录866成功/3编译失败；另有Pilot的2次验收编译失败，没有数值失败。物理失败、成功计时、cache、CPU/Mock契约与Replay不互相替代。原始记录恢复后76份SQLite quick_check通过。

## 2. 空间审计及约束边界

复用`ConfigSpace/SpaceView/SpaceDefinition`，未新增数据库或Runner。条件参数只规范化为声明的不激活默认值。`Relation/Forbidden`可提前传播；任意Python回调仅作完整配置检查，不能自动推导其约束。部分赋值不能判断时保持unknown。完整有限集合、结构化构造、原生有限候选分别支持，不强制进入统一池。

下表为4 CPU开发机、RTX4090**规划profile**的CPU审计，不能冒充实际probe或GPU合法率；每例独立进程测RSS，静态构建上限15秒：

| 实现 | 原声明组合 | 硬件上下文组合 | 条件规范化 | 静态可行数 | 构建时间 | 进程峰值RSS |
|---|---:|---:|---:|---:|---:|---:|
| fused | 3,840 | 3,840 | 3,840 | 3840 | 0.99s | 24.1MiB |
| pipeline | 15,360 | 15,360 | 15,360 | 15360 | 4.07s | 30.2MiB |
| moe | 2,488,320 | 1,244,160 | 691,200 | 31975（未完成） | 15.00s | 46.2MiB |

MoE的USE_TMA在本上下文关闭；其余Hopper候选仍保留。MoE计数只完成前缀，不计算虚假覆盖率。请求32个候选的实测生成耗时和拒绝原因见[审计JSON](data/space-production/space-audit.json)。旧d7冻结源码在相同规划profile上静态保留fused3456/3840、pipeline15200/15360。旧活动池512点在B32通常尚未触及四分之一补充门槛；这限制低预算下的可访问候选，而非证明空间本身稀疏。

现在：50,000点以下尝试有预算完整构建；不完整/更大空间明确用算法自己的活动池。Surrogate每轮至多评分4096个新子集，披露近似；外部SMAC/DE直接提议完整声明空间。管线两Shape的每个B32原生方法均有 **30/32个GPU验证候选不在seed7的512点原始随机样本中**。这是真实可达性证据，不是重跑旧优化器或证明旧池性能更差。

测试独立核对小空间全集、条件嵌套、3亿组合中3点的依赖构造、不连通区域、联合变参、opaque回调、超时、原生非笛卡尔候选，以及旧池外候选通过真实Runner/SQLite的合成目标回归。合成目标不是GPU性能证据。

`complete/proven_empty`、`budget_exhausted`、`exhausted`、`unsupported`分开；有界未找到不证明空集。输入/运行时不适用也不再错误缓存为proven_empty。字节预算是估算保留数据量，外层进程期限与实测RSS另计，不能称OS内存硬限制。

共享内存估算带 **sm120/Triton3.7校准范围**，作为advisory；精确workspace、索引、线程与实际编译器资源检查保留。4090/Triton3.2的一条真实pipeline通过记录使用98304-byte共享内存，旧公式预测131072。以编译器实际报告的101376上限重算，3条成功观测会被旧估算拒绝，见[反事实诊断](data/space-production/estimator-counterfactual.json)。**原Torch2.6 probe字段为0/unknown，所以旧运行时当时未真的筛掉它们**；没有改写原HardwareSpec。规划profile、保守预测、实际编译资源、未知值不能混用。

## 3. 两个成熟优化器与真实搜索

- **SMAC3 2.4.1 / ConfigSpace1.2.2，BSD-3-Clause**：原生RF ask/tell；初始化4点、单CPU模型工作者。条件、类别/有序域与可表达禁配映射原库，其他约束最终检查，不能无损映射时拒绝或明确披露。重复原生提议内部反馈已揭示值，不换成无关配置。固定进程`PYTHONHASHSEED=0`恢复。
- **Kernel Tuner1.5.0，Apache-2.0**：直接调用原生constraint-aware DE与CostFunc，population8/F=.7/CR=.6/best1bin。完整可构建的静态可行集合只提供坐标、没有性能标签。仅适配原版QMC seed及等量LHS的np.shuffle缺陷，DE主体未改。同步库通过固定seed与已揭示反馈计算重放，所有重放成本计wall；无法有界完整构建时明确拒绝，不回退Random。

原生一致性与7→12预算恢复只新增5次合成评估的CPU测试通过。GPU上 **SMAC每个B32有28次提议时RF已拟合**，原生acquisition/random origin均保留；DE的B8均为初始化，B32均实际完成3代。失败的CRASHED/罚值仅为优化器内部编码，不写成真实latency。最终反馈与pending未测提议均落盘。

B32已观测best latency，单位us；只有一个算法seed7，参考是各实现/Shape的**已观测并集best-known**，不是完整声明空间的全局最优：

| M / 实现（N512 K1024） | Random | Heuristic | Surrogate | SMAC RF | KT DE |
|---|---:|---:|---:|---:|---:|
| 4 / fused | 3.110 | 3.112 | 3.088 | 3.296 | 3.088 |
| 4 / pipeline | 4.967 | 5.096 | 4.784 | 4.752 | 4.744 |
| 128 / fused | 6.184 | 6.432 | 6.064 | 5.952 | 6.344 |
| 128 / pipeline | 6.416 | 7.336 | 7.392 | 6.326 | 6.952 |

[曲线及完整成本/阶段诊断](data/space-production/controlled.json)；例：[管线B32曲线](data/space-production/controlled-7.svg)。B32为静态声明3840/15360的0.83%/0.21%，GPU真正可行总数仍未知，不能据此声称1%成本达到全局最优。Random每case32个不同采样均通过，但相同seed的两M复用了相同参数序列，不能当128个独立随机配置估计全空间成功率。

资源失败是114688/126976超过实际101376，保留3条原记录；它们在其他硬件/软件上不被全局禁用。现有research在任何候选失败时总体返回failed/exit2，即使所有trial正常完成；本轮保留这个行为及失败报告，不能把800观测说成800全通过。

受控研究job wall608.28s；新评估wall179.44s（其中build/JIT编译148.59s）；算法factory/proposal/feedback累计0.066/161.33/5.96s，独立验收39.28s，其余含worker启动、预检等。分项有包含关系，不相加冒充独立总成本。下载/安装及Marlin的一次性原生源码build669.54s另列。共享测量/JIT缓存的对照不证明独立冷启动更快。

8个事后冻结的近优/Heuristic配置进行了两input seed独立确认：两个input run中位数的描述性CV（ddof0）最大0.126%，不是置信区间。融合M4原先不足1%的差距仍应视作近似持平；管线M128配置差距仍约15.7%。这支持继续研究规则失配，不构成算法总体显著性结论或硬件瓶颈因果证明。

## 4. 真实生产Marlin及同层L3

固定 **vLLM0.8.5 GPTQ-Marlin**。4090用原生源码build profile：固定官方sdist，数值CUDA和内部模板选择器原封保留，只注册两个原生schema；所有该实现模板仍在。实际库10,069,160 bytes / SHA `099d5481a21aa35750b18deb9324e944fd6828d90dfff0338b49263d98502c67`。这是sm89源码build，**不是**H800已CPU加载的官方wheel相同二进制，也不是自写Triton替身。上游许可证、文件SHA和[准备/构建脚本](../scripts/build_marlin_native.py)随仓库提供。

`w4a16_gemm@2.0.0`固定W4、group128、零点8、FP16、连续布局、K%128/N%64，无隐式padding/act-order。host counter生成可核验逻辑字节；标准GPTQ4 K-word与Marlin布局独立变换，GPU原生repack逐字核对，scales也按上游排列。输入与实际packed数据分别有hash。原canonical受控@1.0.0及group32保留；不声称可读任意GPTQ/AWQ checkpoint或等同生产模型布局。

原生默认及atomic opt-in使用固定上游 **apply_gptq_marlin_linear / should_use_atomic_add_reduce** 本身，并追踪实际调用。默认atomic关闭；opt-in规则也作为更强基线保留，不能只展示默认。C++内部tile选择继续原生。四个显式选项组合为atomic请求×FP32-reduce请求，不是四个tactic或四个二进制。实际allocation/符号观察可用，逐partition有效atomic分支仍unknown。

**主L3 cohort固定USE_FP32_REDUCE=True**，只比较两个atomic请求选项；原生算术路径仍按同一FP16/参考误差契约验证，不声称逐位一致累加。额外FP32-reduce=False只在[四选项扩展分析](data/space-production/l3-extended.json)另列，不混称固定累加策略对照。

六Shape覆盖M1/16/32/64/128/257，N512/1024/2048，K1024/2048；均匀和[8,4,2,1,1,1]都是**合成频率**。原生真实flags映射到同一显式候选矩阵评分，默认调用实测延迟另存，不跨实现混表：

| 主cohort：相对两选项已测参考的期望延迟损失 | 均匀 | 合成偏斜 | 实际配置数 |
|---|---:|---:|---:|
| 原生默认规则 | 60.04% | 97.17% | 1 |
| 原生atomic opt-in规则 | 18.04% | 8.97% | 2 |
| Exact / Greedy / Local swap，K1 | 2.06% | 1.02% | 1 |
| K2 | 0 | 0 | 2 |

四选项扩展中K1/K2/K4均匀损失2.15%/0.0045%/0；这里只是有限选项与固定旧版本/小工作负载的结果，不是生产服务加速比。原生policy不强行伪装等K；数值安全、其他输入、其他设备/新版本可能需要不同取舍。

真实部署例通过 **ShapeDispatcher.select_config→SDK→原生调用** 执行六个覆盖输入，并核对6次lookup、实际native flags和GPU正确性。工厂/host lookup计proposal wall，GPU Graph不包含host查表。未知Shape明确拒绝，无偷偷超预算fallback。绑定硬件、算子、native source/binary指纹与表SHA，换设备须重新测量建表。

计数分开：主cohort2个调用选项、扩展4个；整个六Shape观察到4个Marlin符号；加载1个库；per-kernel CUBIN/code bytes与部署Graph预算均未可靠求得。库容器10MB不能当单变体代码大小。K2不会裁剪库中的预编译模板，也不会自动减少代码体积。**本轮解决的是有限Config Budget对照能力，未声称解决Compiled Variant/Binary Budget。**

两Shape共同输入例的逻辑hash在受控fused/pipeline与原生实现间一致；原生ABI返回新C，平台额外copy到固定输出，且这一步对默认/显式都在计时内。控制实现直接写固定输出，因此不报告跨实现“生产加速比”。预打包在prepare，逐调用清零、reduction及copy保持在实际发生范围内。

## 5. 三条外部AI接口

- **专用模型产物**：实际加载有SHA的手写线性接口产物，记录feature/model身份与CPU推理成本，可提出部分配置；明确未训练，不是学得的性能模型。此路径在三个开发Shape共24次新候选评估。
- **本地小LLM进程/端点**：stdin/stdout与localhost HTTP契约均可运行，完整/partial/region输出有限重试、超时与大小检查；无大候选列表Prompt。进程GPU隐藏，GPU实验的HTTP须声明CPU/独立设备，实际放置须外部核验。已运行的decision_fixture不是LLM；**无现成模型，真实LLM未运行**，没有下载模型或调用付费API。
- **外部生成程序**：v1契约通过→train/validation GPU反馈72条→仅据反馈导出边际参数偏好，提交审查后的v2→两份完整源文件冻结→M32/M64、seed19各B8实测。导出反馈先核对roster及SQLite索引Shape/split，拒绝打开test payload；两个版本的SHA与反馈SHA进入trial。v2包含联合变参和每四步全局探索，不是返回硬编码最优配置。

冻结测试v1为5.120/6.416us，v2为3.967/4.672us。这里只证明接口/冻结/真实反馈流程；仅两个Shape、一个搜索seed，v2还消耗72条开发观测及15次独立验收，不是公平冷启动“AI胜利”。实际LLM生成模型、原始Prompt与平台外生成成本未采集；不能补造为已测成本。生成程序的运行/候选成本已记录。可信Python程序不是恶意代码沙箱。

## 6. 安装与一条命令

给定示例从仓库根运行；wheel用户可使用独立插件并给模型/程序文件提供自己的绝对路径。原生依赖不要安装到其他项目共享环境。

```bash
git clone --branch research/space-production-integration https://github.com/MingASA/KernelTuningLab.git
cd KernelTuningLab
git checkout --detach 46a33442f307fee1451ff4ef6cfef748a759be79
python3 -m venv .venv-search
.venv-search/bin/python -m pip install -e '.[optimizers]'
PYTHONHASHSEED=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 .venv-search/bin/ktlab research --spec examples/space_production/native-cpu.json --output artifacts/native-cpu
.venv-search/bin/ktlab research --spec examples/space_production/ai-cpu.json --output artifacts/ai-interfaces
```

CPU例仅验收流程，域太小不用于BO优劣结论。在已验收兼容GPU私有环境中，以`PYTHONHASHSEED=0`运行：

```bash
python -m kernel_tuning_lab research --spec examples/space_production/controlled-search.json --output artifacts/open-w4 --write-only
# 检查资源后去掉 --write-only；相同命令恢复，报告包含物理/cache/失败成本。
KT_VLLM_COMPONENTS=/private/verified-components python -m kernel_tuning_lab research \
 --spec examples/space_production/production-smoke.json --output artifacts/marlin-smoke
python scripts/marlin_rule_study.py prepare --smoke artifacts/marlin-smoke \
 --spec examples/space_production/production-matrix.json --output artifacts/marlin-matrix
# 对production-matrix.json运行research，再分析、生成自己的硬件绑定表并实际执行：
python scripts/marlin_rule_study.py analyze --study artifacts/marlin-matrix --fixed-fp32-reduce --output artifacts/l3.json
python scripts/marlin_rule_study.py dispatch-spec --analysis artifacts/l3.json \
 --template examples/space_production/production-smoke.json --runtime-table --output artifacts/dispatch.json
KT_VLLM_COMPONENTS=/private/verified-components python -m kernel_tuning_lab research \
 --spec artifacts/dispatch.json --output artifacts/dispatch-run
```

`prepare_marlin_components.py`验证固定官方wheel；`build_marlin_native.py`使用已存在的CUDA/GCC从固定sdist构建源码profile。两种profile都在cache查询前绑定binary SHA，运行时重验。不能将本轮sm89编译库/缓存带到H800冒充可执行。GPU执行遵循现有[门禁与机器手册](REMOTE_GPU_TROUBLESHOOTING.md)。

完整声明空间的有界CPU审计：

```bash
python scripts/audit_config_spaces.py --hardware 4090 --seconds 15 --output artifacts/space-audit.json
# 可用 --source-root 指向冻结的d7 worktree，复核旧规则；不调用GPU。
```

[SDK空间与hooks](EXTENDING.md)解释normalize/partial/complete/联合neighbors、预算状态及runtime_identity。SMAC/KT作为模块插件加载，无需改核心注册表。实验恢复重建原生随机状态并反馈已揭示cache，不能重复tell；所有报告只排名reference-verified结果。

## 7. 问题、验收与剩余边界

| 问题/验收 | 状态及证据 |
|---|---|
| 硬估算跨架构/编译器被当合法性定理 | 已修为有范围advisory；实际98304 vs预测131072、3条反事实拒绝；精确workspace/编译资源检查保留 |
| 稀疏空间在搜索前被盲抽512误挡 | 已修验收依赖构造；穷举/稀疏/联合移动/unknown/timeout回归 |
| 运行时缺失被缓存为proven_empty | 已修为unsupported，不缓存空集合；CPU最小复现及最终GPU有效上下文回归 |
| 原生SMAC有限空间耗尽异常 | 已修；CPU9cache/0new恢复通过，GPU原生4点与MoE单点正常耗尽 |
| AI独立验收1点预算覆盖2 Shape不足 | 已修为2；首个partial与72原记录保留，恢复72cache/0new及9独立验收通过 |
| 轮询控制字符串缩进错误、SSH中断 | 私有控制最小复现与原receipt恢复；未重测GPU。现在有限独立session/短SSH轮询及可续传SHA验证 |
| 搜索真实空间与成熟库原生阶段 | 通过800观测纵向闭环，有限2 Shape/实现、一个算法seed；非完整GPU可行性穷举 |
| 生产路径、packing、原生规则与真实查表 | 通过有限FP16六Shape；非完整checkpoint/推理服务，未解物理变体预算 |
| AI三接口 | 契约+真实GPU反馈通过；产物/HTTP fixture不是真LLM，真实LLM未执行 |
| 其他硬件 | 旧验证不变；本轮H800被占用、5060显示准入拒绝，Hopper生产路径未执行 |
| 数据 | 76 SQLite恢复检查、三套回传SHA通过；最终封存见[archive.json](data/space-production/archive.json)。无已授权异机备份目标，仍待办 |

已审查失败缓存与观察预算、输入/输出正确性、源码/二进制/软件指纹隔离、独立验收和Graph范围。仍有限制：KT需要有界完整可行集合；大空间Surrogate评分有近似；Torch2.6共享内存probe为unknown，不能用0当容量；瞬时环境变化须重新准入；有限表未覆盖输入拒绝；原生每函数代码体积未知。没有绝对无Bug或跨硬件性能已验证的声明。

最有价值的后续问题：①用少量反馈纠正可撤销先验，尤其量化pipeline中的规则失配；②联合学习性能/可行性与真实调优成本，保留编译器/硬件作用域和缺失特征；③在真实原生候选与实际编译身份上研究L2–L3联合选择。先增加冻结的留出Shape/seed与兼容GPU配对证据，再接真实外部模型；目前不需要训练平台或给Random套LLM名称。

来源及许可证：[SMAC2.4.1](https://pypi.org/project/smac/2.4.1/)、[ConfigSpace1.2.2](https://pypi.org/project/ConfigSpace/1.2.2/)、[Kernel Tuner1.5.0 DE](https://github.com/KernelTuner/kernel_tuner/blob/1.5.0/kernel_tuner/strategies/diff_evo.py)、[vLLM0.8.5 Marlin](https://github.com/vllm-project/vllm/tree/v0.8.5/csrc/quantization/gptq_marlin)。实际使用源码/依赖另有SHA，链接不替代GPU证据。
