# 可行空间、成熟优化器与生产W4A16接入

## 实施决策（2026-10-28，实验前记录）

起点为`research/crossgpu-anchor`的`d7eaca5`；本轮在`research/space-production-integration`开发。旧Anchor和封存不改写、不重测。新协议和源码分别产生新实验身份。

- 保留`ConfigSpace/SpaceView`、SDK、v2 Runner、SQLite/JSONL、正确性与Graph流程。增加有预算的完整构建、依赖感知补全、可显式表达的关系、原生有限候选及诊断；任意Python约束继续最终检查，不能自动解析。原始笛卡尔积上界与规范化/合法数分开。
- 共享内存估算移至带来源/适用范围的advisory；精确workspace和语义/线程上限保留。编译后的真实资源继续校验，失败仍按实现/硬件/软件/协议身份保存，不全局禁用。
- **SMAC 2.4.1 + ConfigSpace 1.2.2**：BSD-3-Clause，原生ask/tell和随机森林；该版本默认不要求pyrfr/SWIG，避免为旧版本编译器依赖修改系统。条件和可表达forbidden映射原生ConfigSpace，任意回调保留最终检查。初始化4点，正式B32验证是否实际进入模型阶段。
- **Kernel Tuner 1.5.0**：Apache-2.0，直接调用其原生constraint-aware differential evolution。使用完整可构建的可行集合/参数坐标，不接收旧512随机池；population8，B8如仅初始化则明确标记，B32检查进化。不可有界构建的空间明确拒绝此适配，不冒充Random回退。
- 优化器可在隔离的可选依赖环境运行；不修改Torch/Triton/驱动。原生重复提议在优化器内部用已经揭示的结果反馈；失败罚值仅为内部编码。恢复使用固定seed、原提议/反馈日志及原测量缓存，禁止把罚值写入latency或重复物理测量。
- 首个生产对象固定**vLLM 0.8.5 GPTQ-Marlin**（Apache-2.0）。版本元数据要求Torch2.6，与既有4090/H800私有cu124栈匹配；0.7.3要求Torch2.5.1，故未选。先复用可合法取得的缓存组件，否则仅提取官方wheel的算子/源文件，不安装vLLM服务或下载模型。**当前仅完成接口调查，CUDA尚未验收**。
- 生产默认调用保留上游`apply_gptq_marlin_linear`与原生atomic选择规则；显式变体先使用该版本确有的`use_atomic_add`与`use_fp32_reduce`。只有实际执行/追踪证明分支后才声明覆盖，不虚构tile参数。默认环境禁用atomic的事实也如实保留，可另列上游atomic opt-in规则，不能混称默认。内部tile/binary不可获取时记unknown。
- 固定共同逻辑量化权重：W4、group128、对称零点8、FP16，host counter生成可校验的跨设备一致字节；canonical/GPTQ/Marlin packing分别记hash并核对逻辑值。预打包在setup，逐调用初始化/reduction在完整调用中。原生分配/输出适配开销显式记录，不把非等价计时边界作加速比。
- AI三路线只做标准插件/进程接口、身份/成本/超时及生成策略版本冻结，不训练或自动下载模型。无现成可用本地LLM时保留真实模型not_executed；契约fixture明确标注。

## 首批验收预算

1. CPU：W4 fused/pipeline完整静态过滤各15秒/50,000节点上限；复杂MoE与合成稀疏空间按需构造，返回超时/未知而非伪称空集。记录wall、峰值内存、拒绝原因与候选覆盖。
2. 本地GPU：先12个有代表性W4候选Pilot（600秒），再最多4 case × 5方法 × (B8+B32)=800搜索观测；验收另计、单worker、分批每批≤1小时，先根据Pilot缩放。目标是公开声明空间搜索，不是固定C16池。所有新协议独立目录。
3. 生产目标优先H800：先默认/显式分支各2配置的小Smoke（600秒），再4 Shape×4候选×2输入run=32主观测及默认基线；在线优化/分派小复核另设明确预算。单卡、每批≤1小时、显存预估≤3GiB，实际以probe/Pilot为准；不并发同卡LLM推理。
4. L3使用上述小实测矩阵，K=1/2/4、均匀及明确合成偏斜权重，与原生规则同层对照；只验证已覆盖输入，未知Shape原生回退/拒绝并披露预算变化。

资源现状：本地4 CPU、7.8GiB RAM；远端沿用已验证Python3.11/Torch2.6+cu124/Triton3.2独立环境。远端只读检查已有空卡，但不是预留，每次负载重新准入。环境、源码、实际GPU与原始记录位置只保存在私有日志。

来源：[SMAC ask/tell](https://automl.github.io/SMAC3/latest/advanced_usage/5_ask_and_tell/)、[ConfigSpace conditions](https://automl.github.io/ConfigSpace/latest/reference/conditions/)、[KT 1.5.0 DE源码](https://github.com/KernelTuner/kernel_tuner/blob/1.5.0/kernel_tuner/strategies/diff_evo.py)、[vLLM 0.8.5 Marlin规则](https://github.com/vllm-project/vllm/blob/v0.8.5/vllm/model_executor/layers/quantization/utils/marlin_utils.py)。实际使用包和提取文件另记SHA；这些链接本身不代表已运行。

## 首轮CPU验收与资源调整

- 核心394 passed / 12 skipped（11项GPU opt-in及未安装可选优化器模块）；可选环境7项原生优化器测试通过。SMAC/KT分别与独立原生调用候选轨迹一致，固定PYTHONHASHSEED=0的两个独立进程轨迹相同，7→12观测恢复只新增5次模拟CPU目标评估。上述是CPU契约证据，不是GPU可行率。
- SMAC需进程启动时固定`PYTHONHASHSEED=0`以重建原生随机状态；原生Random探索与acquisition origin分别记录，不把所有RF拟合后提议都叫模型引导。失败CRASHED内部罚值与真实latency分离。
- KT 1.5.0原生QMC未传seed，且全量LHS分支有np.shuffle缺陷。适配层仅注入QMC RNG及等量permutation，保留原始DE/CostFunc/repair；同步优化器以已揭示反馈计算重放，所有重放/反馈成本计wall。16观测CPU例实际完成一代，并flush最后反馈。
- 本地5060 Ti准入发现显示占用/低持续负载，未启动GPU负载。先推进合法空闲H800，之后重新检查本地，不暂停/加入任何既有任务。
- H800官方0.8.5 wheel全SHA通过；标准库下载先有CA路径错误，使用现有系统CA后验证通过但传输慢。curl IPv4保留9MiB前缀续传成功，未关闭TLS。实际_C SHA与固定wheel一致，CPU-only加载成功，CUDA尚待下一步验收。
