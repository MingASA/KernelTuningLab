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

注册不导入生产依赖或启动GPU。执行时加载固定原生包/源码与二进制身份，不提供Torch计算替身。`native_sources.py`只提取原始host函数；数值计算始终使用原生实现。后续还有Attention、统一AI动作/驻留、持久原生优化器和L3分派集成，不能将当前阶段写成最终完成。
