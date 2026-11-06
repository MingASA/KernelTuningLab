# 科研评测入口

规范细节以 **[PROTOCOL.md](PROTOCOL.md)** 为准；资产接口见[EXTENDING.md](EXTENDING.md)，多case命令见[RESEARCH_WORKFLOW.md](RESEARCH_WORKFLOW.md)。本页不另设冲突协议。

## 开始采集前冻结

记录资产版本/源码、实际GPU与软件栈、Shape/布局/dtype、候选集合、seed、重复、计时、预算和指标。按整个Shape或GPU划分train/validation/test。先Pilot估算编译、校验、测量和内存成本，再扩大；同GPU串行。

已研究过的几何是历史探索或训练/对照，不能重新包装成盲测。目标标签开始采集前冻结固定先验。改变cohort/实现产生新数据集身份，保留旧表；C48不是C64的完整超集，不能静默混成“全局最优”。

## 观测单位

| 单位 | 解释 |
|---|---|
| latency sample | 一个record内部重复，不等于独立实验 |
| 输入seed/run | 可追溯物理重复；两轮不足以给精确置信区间 |
| 搜索RNG seed | 同一物理标签上的选点变化，不是新GPU实验 |
| Replay reveal | 揭示已测record，另报原source成本和CPU回放成本 |
| 在线观测 | 可能是新物理执行或缓存，分别计数 |

v2的失败和缓存都消耗观测预算，验收另计。默认性能排名只使用独立参考通过的结果；提交者声明和导入动作均不升级GPU证据。输入、初始重复调用和最终实际输出检查不可省略。

## 计时与后端

v1单launch Event、v2同步完整调用、v2 Graph完整GPU计划是不同口径。Graph注明batch size、热缓冲、warmup/repetitions及host步骤不重放；它不代表服务端到端延迟。多步骤反量化、reset、GEMM/reduction按完整计划计时；编译、参考、setup/capture在latency之外但计入实际调优成本。

库对照匹配数学语义、布局、dtype、输出和步骤范围，记录实际后端/版本/参数。回答长度、scheduler、KV格式、paged/continuous或FA2/FA3路径均可改变结果，不能仅从kernel加速推断服务吞吐。canonical W4和连续Attention没有生产格式加速声明。

## L2、L3和成本

regret以同一有限已测集合的正确最好值为分母。未知格子保持None，预测不进入observed矩阵。搜索器只见已揭示历史，oracle仅作最终评分。训练内容和获取成本单列；等目标观测预算的热启动优势不证明冷启动总墙钟优势。

原Solver优化Config Budget。编译身份、去重cubin、ELF容器/指令段bytes和部署常驻体积是不同量，未知即未知。Shape-generic必须用实际编译键、对象、PTX/CUBIN和多Shape执行验证复用。混合实现部署仅在明确标记的有限POC中对齐语义，原始矩阵仍按实现隔离。

交付保留配置、精确record IDs、samples、正确性/失败、commit/内容、GPU环境、训练清单、缓存/物理/验收/墙钟、覆盖和归档校验。观察、波动、机制假设与因果解释分开；无计数器时不把瓶颈推测写成事实。

实测案例见[第一阶段](RESEARCH_PHASE1.md)、[第二阶段验收](RESEARCH_PHASE2.md)，跨硬件先读[迁移检查单](GPU_MIGRATION.md)。
