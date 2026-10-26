# 交给目标GPU执行者的检查单

本轮只有RTX 5060 Ti实测。H20/H200/H800/4090配置、静态计划和本页都不是这些设备的验证证据。不要传输5060Ti的PTX、CUBIN或Triton编译缓存作为其他架构的可复用实现。

## 1. 准备源码与环境

1. 检出双方约定的commit，记录干净状态/改动、Kernel和参考源码指纹、实验JSON SHA256。使用独立分支与独立Python环境。
2. 记录可用CPU/RAM/swap/磁盘、GPU型号/实际资源、驱动、Torch/CUDA Runtime、Triton及编译器。先probe再分配大输入；系统nvcc不代替Torch runtime支持。
3. 5060Ti的[固定依赖](data/5060ti-requirements.txt)是已验证本地栈。目标服务器先核实驱动兼容性；无法保持相同栈时，单列软件差异，不将差距全部归因于硬件。不升级全局驱动或覆盖其他项目venv。
4. 每张物理卡使用独立输出目录。通过作业调度器申请独占GPU，或人工确认没有其他性能任务；必要时设置 `CUDA_VISIBLE_DEVICES` 使一个执行者只看到指定卡。本平台没有跨输出目录的GPU调度器。

## 2. 静态准备与最小实测

下面第一条可以在无Torch/Triton的CPU环境执行，生成原生research/v2规格及预算预览：

```bash
python scripts/prepare_target_gpu.py --target h20 --output artifacts/h20-smoke
# 在真正目标设备、兼容独立环境内才加 --execute
python scripts/prepare_target_gpu.py --target h20 --output artifacts/h20-smoke --execute
```

可选target为h20/h200/h800/4090/5060ti。该小套件有两Shape、FP16/BF16、Random/Heuristic，每trial两个观测；独立验收另计。默认只生成，不调用GPU。H20没有猜测的Profile，使用hardware=null和真实probe，执行时还核对设备名称确为H20；H200/H800使用已有SXM Profile，4090使用桌面Profile。PCIe、MIG或其他变体不能冒称这些规格，应另存真实probe和实验身份。

随后在支持命名Profile的设备执行原四类严格gate：

```bash
KT_PYTHON_BIN=.venv/bin/python bash scripts/run_gpu_smoke.sh H200 artifacts/h200-core
# H800 / 4090替换相应Profile；H20须先基于真实probe核实Profile，不套用H200数值。
```

逐类确认数值/边界、资源和计时闭环后再扩大。T1普通dot路径不要求Hopper专属TMA/WGMMA接口；编译器实际lowering与资源仍以目标编译为准。Shape-generic对照另需支持 `do_not_specialize_on_alignment` 的Triton JIT API，本轮API验证版本3.7.1。

## 3. 同协议研究复测

从 `examples/research_phase2/matrix.json` 复制一份，调整明确的硬件选择，保留T1源码、Shape定义、dtype、C48、输入seed、warmup10、31样本和Graph512。先write-only检查总量，再Pilot、训练分区、冻结先验、验证/测试分区。当前模板的C48固定GROUP1/stage3，不等于192候选全空间，也不是C64超集。

实际硬件约束可能排除候选，保留invalid/unsupported/compile/numerical/runtime/missing状态。共同合法集合应取各卡**已编译且正确性通过的Shape/config格子交集**；额外硬件专属空间另设有标识的cohort，不删除其他架构可用配置。不要用缺失预测填充完整实测表。

跨硬件按相同实现/内容、语义、dtype、布局和measurement scope连接，各GPU矩阵分别保存。在共同已测集合上计算源卡Top配置的目标迁移损失、Spearman、Top重叠、近最优覆盖及失败/缺失率。存在多卡真实数据后才做Leave-One-GPU-Out；单张5060Ti不构成多硬件训练集。

## 4. 原始结果回传与验收

- 保存source/parsed spec、assets.lock、当前准确record IDs、SQLite/JSONL、原samples、首次失败、完整日志、环境/probe、温度频率/负载及编译资源。库对照还应保存实际后端/版本/dispatch与参数。
- 同时交付目标观测、新物理测量、缓存、验收、编译/校验/capture/运行和job墙钟。编译器cache热态与真正首次编译分开，避免重复加成本。
- SQLite使用标准备份或关闭连接后归档，不能只拷贝活动WAL的主文件。导入导出后比较完整record ID和语义；原记录不重写为新设备。
- 生成SHA256及逐文件清单，传输到已授权目的地，接收端复核。`scripts/backup_research_artifacts.sh`可复制本地已封存归档，不自动上传公网。
- 验收表分别写通过、部分覆盖、未执行和已知限制；没有目标原始证据时状态保持未执行。
