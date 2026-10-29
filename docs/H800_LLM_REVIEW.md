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
