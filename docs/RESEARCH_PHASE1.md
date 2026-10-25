# 第一阶段配置空间研究：T1 / RTX 5060 Ti

本研究从已验证的 `9393d17` 创建独立分支 `research/shape-config-matrix`。原验证分支和 `main` 保持不动。研究问题是配置偏好如何随 Shape 改变、有限观测预算能否利用这些规律，以及少量配置覆盖多个 Shape 时到底限制了什么。

## 已冻结的设计与协议

T1 是 `triton/t1_gemm@1.0.0`，通过现有 SDK 自动接入，复用原 GEMM 的输入与独立 FP32 参考。保留 `triton/gemm@1.0.0` 受控基线，增加固定的 `torch/mm@1.0.0` 库基线。T1 使用 Tensor Core 分块、递增指针、软件流水线和 grouped program ordering，所有搜索均由平台显式执行，没有内置 autotuner、融合激活或数学语义变化。

实现参考 [Triton 官方教程 03](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)，保留 [MIT 许可](../third_party/triton-tutorial-LICENSE)。实际代码和依赖文件指纹由资产系统绑定，每条记录保留执行时的真实环境。

受控空间为六个实际参数各两个值，共 64 配置：BM=32/128、BN=32/128、BK=32/64、warps=4/8、stages=2/3、GROUP_M=1/8。原始实现的更大 SDK 空间未被宣称穷尽。GROUP_M 改变多行 tile 的执行顺序；单个 M tile 时会退化，不能将该 Shape 下两个 GROUP_M 值的微小波动解释为调度收益。大 tile/高 stage 的理论操作数暂存量可达 98,304 bytes，最终仍检查实际编译 shared memory，未仅依赖估计。

10 个 Shape 的理由、固定的整 Shape train/validation/test 划分与输入规模在 [shape-design.json](../examples/research_phase1/shape-design.json)。低 M 覆盖 B 位于 L2 内、接近 L2 和超过 32 MiB L2 的情况；其他 Shape 覆盖计算型、长宽矩阵、N/K 比例和三轴尾块。最大原始 FP16 输入/输出约 64 MiB，另计 FP32 参考、输入完整性检查、编译器内存与 Graph；Graph 复用缓冲，未分配 512 份输入输出。

主协议：物理 RTX 5060 Ti 16GB / sm_120；Torch 2.13.0+cu132、Triton 3.7.1、CUDA Runtime 13.2、驱动 616.92；Graph 每次重放 512 次完整调用、warmup=10、31 个样本。三轮独立输入 seed=7/19/31；每个 trial 内候选顺序由独立 SDK 插件可复现地打乱。原始样本、配置、dtype、stride、正确性、源码与真实 GPU 身份保留在既有 v2 journal。不同 seed 是不同输入；另有同 seed 的前后锚点检查时间漂移。

Pilot 的 4 Shape × 12 配置 × 2 seed，共 96 次新测量全部通过，另有 32 次独立验收，总 job 墙钟 236.39 秒。两轮中位数相对差的 P50/P90/最大值为 0.82%/1.99%/3.85%。这些只有两轮，不能视作精确置信区间。超过 L2 的 decode 每候选平均实际成本 5.305 秒，因此正式 trial 上限由 180 调到 600 秒，保留样本质量；整轮采集仍用 90 分钟硬期限。

## 复现

沿用 [5060TI_RESULTS.md](5060TI_RESULTS.md) 的项目独立环境，不升级全局驱动或其他项目的软件栈。先检查上轮数据，再串行执行：

```bash
(cd artifacts && sha256sum -c 5060ti-evidence-20261025.tar.gz.sha256)
(cd artifacts && sha256sum -c research-workflow-evidence-20261025.tar.gz.sha256)
.venv/bin/python examples/research_phase1/design.py
.venv/bin/ktlab research --spec examples/research_phase1/matrix.json \
  --output artifacts/research-phase1/plan --write-only
.venv/bin/python scripts/collect_phase1.py --spec examples/research_phase1/correctness.json \
  --output artifacts/research-phase1/correctness
.venv/bin/python scripts/collect_phase1.py --spec examples/research_phase1/pilot.json \
  --output artifacts/research-phase1/pilot
.venv/bin/python scripts/collect_phase1.py --spec examples/research_phase1/library.json \
  --output artifacts/research-phase1/library
.venv/bin/python scripts/collect_phase1.py --spec examples/research_phase1/anchor-before.json \
  --output artifacts/research-phase1/anchor-before
timeout --signal=INT --kill-after=30s 5400s .venv/bin/python scripts/collect_phase1.py \
  --spec examples/research_phase1/matrix.json --output artifacts/research-phase1/matrix
.venv/bin/python scripts/collect_phase1.py --spec examples/research_phase1/anchor-after.json \
  --output artifacts/research-phase1/anchor-after
.venv/bin/python scripts/analyze_phase1.py --matrix artifacts/research-phase1/matrix \
  --library artifacts/research-phase1/library --simple artifacts/research-phase1/simple \
  --output artifacts/research-phase1/analysis
```

先运行对应的 `simple.json`、`bf16.json`、`w4.json` 才能分析这些补充实验。主矩阵是 1,920 次研究测量，独立验收最多 120 次；不是“预算64，整套只测64次”。所有 GPU 执行串行；采集脚本只为 `ktlab research` 增加每两秒的只读温度/频率/功耗/利用率采样，没有第二套 Runner 或数据库。重跑同一命令重新验收、缓存已完成候选，缓存不算新物理测量。

## 数据选择与研究纪律

分析只读取当前 suite 各 trial 明确揭示的 record ID，不扫描累计 journal 捡最好成绩。原始记录通过现有 `ExperimentStore` 导出到按记录 ID 集合 SHA256 版本化的标准 SQLite/JSONL，入口是分析目录的 `dataset-manifest.json`。可用 `ExperimentStore.import_jsonl` 导入其他实验数据库；原 record ID、计时/正确性/源码语义不变。`coverage.json` 明确列出每个 Shape/config/seed 的缺失与失败。若任何正式格子缺测，脚本保留部分矩阵并拒绝宣称完整矩阵。

矩阵按每格三轮共 93 个原始样本的中位数聚合，独立 run median 的 CV/范围另列。Replay 每次仅使用一个真实测量轮次，三轮分别重放，避免用三轮测量的总成本冒充一次观测。比较 Random、Grid、已有 Heuristic、已有 Surrogate 和 Transfer；预算4/8/16/32，搜索 seed=0..9。Transfer 只读预先划定的四个 train Shape；validation/test 从未提供训练标签。已测全表最优仅用于最终评分，不交给搜索器。其成本是原记录成本的 Replay，新增 GPU 测量为0，不等同于冷启动真实调优墙钟。

L2–L3 联合探索也是离线选择模拟：4 个 test Shape、K=2、32/64/128 个已揭示 cell 上限。独立 L2 流程为候选补测预留预算；公共面板方法每个候选在所有目标 Shape 测量。训练先验只用 train Shape，所有目标标签均通过收费 reveal。未揭示的已测全表只在最后评分。不同方法实际花费可低于上限，必须同时报告。

## L3 的三个预算

现有 Solver 限制 **Config Budget**：至多 K 个配置参数字典。新的 `config_count` / `budget_kind=config_count` 为正式字段；旧 `variant_count` 保留为兼容别名，含义仍是配置数。

**Compiled Variant Budget** 限制真实编译特化。T1 的 M/N/K、stride 和 tile 都是 constexpr；同一配置跨不同 Shape 不能假定共用二进制。实际 Triton compiler cache identity、PTX/cubin SHA256 及产物 bytes 由 `record_compiled_kernel` 保存。多步骤量化路径分别记录每个 Kernel。Torch 内部 dispatch 的身份未知，保持未知。

**Binary / Code Size Budget** 限制部署代码体积。本轮可统计去重 cubin ELF 容器大小，但这包含元数据，不等于纯指令段、驱动常驻内存或最终部署包体积。已有 Solver 未求解后二者。分析分别统计“选定配置 × 所有已知 Shape 全保留”和“仅按最终分派所需特化保留”，不能将一种部署策略的计数冒充另一种。

正式 L3 使用10 Shape，显著多于K=1/2/4/8。均匀权重与明确合成的 decode 偏斜权重分开；没有真实线上概率。精确 oracle 可先删去在所有已测 Shape 上被另一配置支配的列，这是有限表的精确简化；仍超枚举上限时明确未计算，另有预先确定的16配置子集 exact。报告同时给出加权绝对延迟目标和逐 Shape 相对损失。

## 跨硬件准备与备份

[cross-hardware](../examples/research_phase1/cross-hardware) 保留同一 T1 源码、Shape、dtype、64候选与计时协议，分别输出到独立目录。H200/H800 规格选择现有明确的 SXM Profile；4090 选择桌面版。H20 没有已验证 Profile，配置 `hardware=null`，使用实际 probe 保存真实属性，不能填一个猜测的 H20 规格。

本轮没有其他 GPU 实测。共同合法子集应由各 GPU 的真实预检/编译/正确性结果取交集；未经执行的配置保持未测试。当前 T1 不使用 Hopper 专属机制，硬件扩展候选另建有标识的 cohort，不能覆盖当前公共候选表。未来迁移统计应分别报告公共已测子集中的 Top 配置迁移损失、Spearman 相关、覆盖和失败率，并检查实现/协议身份。仅有一张 GPU 时不执行 Leave-One-GPU-Out 训练评估。

旧归档均做 SHA256、单次流式读取和本地 SQLite quick_check：最初 GPU 验证归档2,440文件/99个现存 journal，工作流归档3,445文件/19个现存 journal。未发现损坏。没有已授权的外部备份目的地，因此未自动上传。可在确认挂载目的地后执行：

```bash
KT_BACKUP_DIR=/your/authorized/mounted-backup bash scripts/backup_research_artifacts.sh
```

该脚本先验证源归档，复制后再次验证，不覆盖已有文件，不访问外部上传服务。原始产物保持在被 Git 忽略的 `artifacts/research-phase1/`；最终另做归档及校验，公开 Git 只保存配置、代码、精简证据和图表。

完整实测结果及研究判断将在本轮采集和分析完成后补入以下部分。
