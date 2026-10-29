# H800真实LLM对照与目的审查

## 协议冻结前的审查记录

起点fcbe767，开发分支research/h800-local-llm-review。原始归档、旧GPU数据及main不修改；Git身份MingASA。完整协议见[protocol.json](../examples/h800_llm_study/protocol.json)，后续按实际证据更新本页。

| 目的 | 审查结论/处理 |
|---|---|
| 原生优化器而不是库名或随机池 | SMAC/KT真实ask/tell或原生DE、完整静态可行集合、模型/代数诊断均保留；复核默认及warm初始化轨迹 |
| 所有推荐应来自正确性通过的观测 | 复现通用best_config会选中更快的未验证claim；b653726修复主字段及Local/Surrogate incumbent，原观测保留为诊断。44项相关测试通过 |
| 旧AI示例不应当作真实模型结论 | 旧报告明确是fixture；本轮需要完整本地权重、实际GPU forward/token输出、真实开发反馈及冻结目标测试 |
| B应逐次直接决策 | 新增真实模型看到已揭示历史后提出完整配置；无启发式替换。静态portfolio仅作消融 |
| C应由真实LLM生成并修改 | 27B原文表达式通过有限DSL合同后执行；训练输出反馈给模型，不能由编码Agent改策略冒充模型 |
| A专门训练性能/决策模型 | 按用户要求延期；SMAC内部原生RF仍是传统优化器，不计作本轮A |
| GPU计时不能混入LLM计算 | 采用单GPU串行；B每步推理结束将完整模型移回CPU、清KV/allocator并核对CUDA分配后才准许Kernel。推理/upload/offload分别计时 |
| 模型规模与方法差异不能混淆 | B比较已有Qwen3.5-0.8B/4B，27B在相同任务/历史做预算受控的配对消融；C固定Qwen3.8-27B。版本/规模仍是混杂因素，明确报告 |
| 冷/热启动与历史成本 | rule0无测量反馈；rule1只用train，validation选版本；target最后冻结测试。native warm-SMAC获得同样train配置先验，所有历史成本另计 |

## 实际资源与模型选择

H800为8×80GB、sm90；本轮每次只使用一张实际空卡。初始空卡后来被其他任务占满，准入拒绝；一次模型加载中出现新任务，guard只终止本次已核实进程。原始拒绝与中断保留，不能计为模型或Kernel通过。随后H800官方vLLM0.8.5原生Smoke已实际通过。

共享模型库存在完整Qwen3.8-27B BF16（18shards，55.56GB）、Qwen3.5-0.8B/4B等。只读核对完整性、全文件SHA与tokenizer，不下载新权重。选择27B用于C：适合单80GB卡且有代码/反馈能力；使用固定Transformers5.8.1及原Torch2.6/cu124，不升级全局GPU栈。候选Qwen2.5-Coder32B作为兼容回退，尚不因目录名宣称实测。

实际27B已加载27,356,728,560参数并完成3个中性推理Pilot（算术、合法配置、表达式接口），峰值分配约59.8GB；81-token配置响应约5.58秒。表达式输出仍须静态合同，不能把模型建议当可执行正确程序。Transformers5 loading report含set导致记录异常已复现并修复，原始失败保留。

来源：[Qwen3.8模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[固定Transformers5.8.1实现](https://github.com/huggingface/transformers/blob/v5.8.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py)。官方模型能力陈述只作选择依据，不替代本项目测量。

## 当前证据边界

三类方法的目标性能对照尚未完成；不能从推理Pilot推断LLM胜出。B的小模型实时决策隔离Pilot、C开发/迭代及冻结后的共同任务比较正在执行。旧5060 Ti/4090有效实验不重跑。异机备份仍无授权目的地。

## 详细复核：实现与研究目的

| 项目 | 本轮核对与证据 | 支持边界 |
|---|---|---|
| 权威空间而非公共随机池 | B约束解码从既有`SpaceView.build_feasible`构造完整输出语法；H800 fused Pilot为3840点/22875 token节点，三个真实模型各4次新测量通过。CPU测试证明不连通区域不会拼出非法混合点、旧512池之外的点可达 | 只证明静态集合完整；未逐点编译3840点。完整构建超预算则拒绝，不推断空集 |
| 原生优化逻辑与反馈 | SMAC2.4.1原生ask/tell、Kernel Tuner1.5.0原生DE保持；默认轨迹、恢复、失败编码及warm初始化等10项可选依赖测试通过 | KT为同步原生算法的计算重放适配，额外CPU成本计入；不能将B8初始化称为进化性能 |
| 实际GPU与生产实现 | 新H800 stock vLLM0.8.5默认/显式调用Smoke为12次研究、6次独立验收；binary SHA与此前4090源码构建产物分开 | 未运行完整服务/大模型checkpoint，不从两个调用选项推断物理二进制预算已解决 |
| 真实B模型 | 原始freeform贪心及采样轨迹保留重复/漏字段失败。完整静态语法约束后0.8B/1.7B/4B Pilot共12次模型调用、12次研究测量、3次独立验收通过 | token分支由模型决定，无启发式替换；语法约束会改变解码分布，须与freeform分开命名 |
| 真实C迭代 | 27B使用408条新训练研究记录的摘要：45条具体观测及由完整源数据产生的共享配置先验。两个原文规则均通过契约，验证6任务各B16后冻结选择 | 不是“只用45条数据”；源数据、初次失败推理及生成成本均需报告。没有专用模型训练 |
| 推荐正确性 | 通用`best_config`不再接受未验证claim；实际Runner/Store回归保留更快未验证记录但只推荐验证过的配置 | GPU测量继续强制Torch参考及输入/输出检查，没有降低容差 |
| 测量口径和成本 | Graph128/15samples/warm10，完整GPU计划；LM生成/搬移不进入Kernel事件。原记录保留build/setup/graph setup、校验及测量阶段信息 | Runner的`compile_time_ms`包括adapter build/setup/Graph准备，不能直接当作纯编译器CPU时间；子项不可重复相加 |
| 失败与恢复 | 首次set序列化、输出截断、重复建议及warm预检失败均留原记录；阶段终态/原receipt/完整StandardStore记录只读回传并核对id、UUID | 算法失败与资源冲突分开；原命令失败不改成全套通过；他人进程出现只停止自己 |
| 冻结与数据隔离 | train-only反馈入口拒绝混合validation/test；C在target启动前冻结，B只见本任务已揭示历史；原始模型输出、token和Prompt SHA保留 | B的采样seed与输入seed耦合，模型家族/规模也不独立；不能据此声称模型缩放规律 |
| 安装与扩展 | 新模块在`python -S`禁用site依赖下可导入，不预先加载Torch/Triton/Transformers/NumPy；准备器只生成既有research JSON | 真模型仍需可选推理依赖及已存在权重。CPU契约fixture不是模型/GPU结果 |

warm SMAC在目标测试中发现额外SDK集成缺陷：目标先验被套到合成singleton空间，四个job在GPU搜索前失败。原失败保留；`92fd52b`仅在未绑定trial且明确为SDK合成singleton/mixed空间时省略不适用的先验点，实际目标的非法先验仍拒绝。原生一致性及真实SDK回归通过。剩余同类失败检查被显式移出主批次并登记运行时修正，原C策略/提示/预算不变；修复后的warm GPU补测单列源码版本，不能把缺失值伪装成测量。

对两版生成程序还有一个精确检查：它们提到M/N/K，却只是给所有配置加相同的Shape偏置；当可行候选集合相同时，排序与Shape无关。两版也没有使用已提供的在线历史向量特征。见[`audit_rule`](../examples/h800_llm_study/analyze.py)。这不修改模型程序，也不否定接口闭环，但说明“rationale提到了Shape”不能作为实现了Shape-aware搜索的证据。目标性能结论仍等待冻结实验及复测。

最短复现入口见[真实模型示例](../examples/h800_llm_study/README.md)。模型选择核对了[0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)、[1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)、[4B](https://huggingface.co/Qwen/Qwen3.5-4B)及[27B](https://huggingface.co/Qwen/Qwen3.8-27B)官方模型卡。B主实验使用固定采样子集T=.7/p=.8/k=20；未假装支持服务API的presence penalty。C最终两版保持相同非thinking贪心输出设置，最早thinking截断仍算真实开发成本；本轮不据单次解码设置评价模型能力上限。
