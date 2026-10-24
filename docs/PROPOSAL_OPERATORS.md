# 项目书算子适配与生产工作负载

已对照根目录立项书“合作技术点 2.(a)”和原 PPTX 核对：缺项为 **W4/W3/W2A16 反量化 GEMM、Attention、MoEGroupedGEMM**。本轮另外提供三种独立反量化算子，共新增八个语义资产；连同原 GEMM/BMM/Softmax/LayerNorm，目录现有十二类算子。

## 直接使用

```bash
# 无 CUDA：真实 CPU 实现、独立参考、配置搜索与归档
.venv/bin/ktlab proposal-suite --runtime python --output artifacts/proposal-cpu

# 验证机：FP16/BF16、所有新算子及融合/多步骤实现
.venv/bin/ktlab proposal-suite --runtime cuda --hardware 5060ti16g --output artifacts/proposal-5060ti

# 生成生产规模配置，上传目标 GPU 后执行；生成本身不运行 Kernel
.venv/bin/ktlab proposal-suite --runtime cuda --scale production --hardware H200 --write-only --output artifacts/proposal-production

# 单个新算子模板，同样使用标准资产接口
.venv/bin/ktlab scaffold kernel --operator w4a16_gemm@1.0.0 --runtime cuda --output my_w4
```

H800/4090 使用对应硬件别名；不指定 `--hardware` 会探测实际 CUDA 设备，因此也可在 H20 等 sm80+ 设备上采集实际属性。本机缺 CUDA，GPU 项保持 `deferred`。

`--operators`、`--algorithms`、`--dtypes` 可缩小实验；`--variants` 加入算子声明的语义参数案例。默认严格模式会将实际被测配置的失败计入整套失败；研究时可显式使用 `--allow-config-failures` 保留失败继续比较。结果入口为 `suite_report.md`、`suite_summary.json`，每项继续保留标准 lock/预检/验收/SQLite/JSONL/trials 档案。

## 已接入的实现

| 算子资产（版本均 1.0.0） | CUDA 实现与开放参数 | 生产工作负载 |
| --- | --- | --- |
| `dequantize_w4/w3/w2` | 跨 word 解包、分组 affine 反量化；`BLOCK_SIZE`、warps | K/N=4096，K=11008 等 |
| `w4a16_gemm/w3a16_gemm/w2a16_gemm` | 融合解包与 Tensor Core dot；M/N/K tile、warps、stages、`GROUP_M`、`SPLIT_K` | M=1/128/512，N=4096，K=4096/11008 |
| 同上 `triton_pipeline/` | 真正 dequant→GEMM→可选 Split-K reduce，额外开放独立 `DEQUANT_BLOCK_SIZE` | 对照融合路径，完整步骤计时 |
| `attention` | 在线 softmax、因果 KV tile 裁剪；M/N tile、warps/stages、`USE_EXP2`、`PRE_LOAD_V` | MHA/GQA/MQA；2K/16K prefill、16K decode；D=1…256 |
| `moe_grouped_gemm` | 紧凑 tile map、grouped/persistent 调度、L2 tile 排序、Split-K、真实 TMA load；另开放 map block/warps、persistent CTA 数、reduce tile | M=128/4096，N=4096/14336，K=4096，多专家、倾斜与空专家 |

各资产都有 `python/<operator>` 和 `triton/<operator>`；量化 GEMM 另有 `triton_pipeline/<operator>`。CPU 代码只用于小规模数学、格式和平台验收，结果明确标为真实 CPU 测量，不替代 GPU 证据。

声明域在硬件与输入裁剪前：融合量化 GEMM 3,840 个候选，量化 pipeline 15,360，Attention 384，MoE 2,488,320。它们是包含条件与资源约束前的上界，平台采用有界采样，不展开百万级笛卡尔积。

## 量化格式与正确性口径

格式名为 `canonical_unsigned_affine_k_bitstream`：`packed` 是 signed int32 存储的 `[ceil(K*bits/32), N]` 矩阵，每个输出列沿 K 连续 LSB-first 打包。W3 的代码可跨两个 32-bit word。缩放和零点为 `[ceil(K/group_size),N]`，末组允许不足。

`weight[k,n] = (code[k,n] - zero_points[k//group_size,n]) * scales[k//group_size,n]`。中点零点可表示对称约定；GPU 在 dot 前将反量化权重舍入 FP16/BF16，参考采用相同舍入口径后做 FP32 matmul，并禁用 TF32。

`kernel_tuning_lab.quantization` 提供 `pack_bits`、`unpack_bits`、`dequantize_packed`。这是明确版本化的规范格式，**不自动冒充 GPTQ/AWQ checkpoint 布局兼容**。`group_size` 是实验语义参数，改变它需匹配输入的 scale/zero-point 布局；它与可调的反量化执行 tile 不同。

## Attention 与 MoE 语义

Attention 输入 Q 为 `[Batch,Heads,SeqQ,D]`、K/V 为 `[Batch,KVHeads,SeqKV,D]`，要求 Heads 可被 KVHeads 整除。默认 bottom-right causal 对齐，允许 top-left 或 noncausal；全遮罩行输出零。当前为连续、非分页、无 dropout 的推理 forward。CUDA 参考按 query 分块，不分配完整 S² 矩阵；BF16/FP16、尾维、大 batch 和极值指数转换有明确处理。

MoE 接收已经分组的 A `[M,K]`、专家 B `[Groups,K,N]` 和 `group_offsets[Groups+1]`，支持零行专家。`distribution` 支持 balanced/skewed/empty_first/empty_last，也可在实验参数中给出 `group_sizes`（非负、长度 Groups、和为 M）。路由、Top-K 和 scatter 不包含在这个 GroupedGEMM 算子中。GPU 的 counter reset、tile map 构建、matmul、可选 reduction 都在同一计时流程；大权重通用寻址使用 int64，TMA 使用三维专家描述符。

## 空间接口同步扩展

`SpaceDefinition` 现在支持上下文 resolver；新增只读 `SpaceContext(hardware,operator,shape,environment,workspace_memory_limit_bytes)`。新签名为 `resolver(context)` / `constraint(config,context)`，原三/四参数签名继续支持。环境和工作区预算独立于硬件身份，进入解析与测量协议。

有效空间取 Kernel 声明、上下文子域和实验范围的交集。resolver 只能缩小，不得增加未知参数或值；裁剪记录在 `space.resolution`。参数可声明 `description`、`unit` 和 `active_when`。TMA 在 sm90+、TensorDescriptor 软件接口存在且矩阵步长满足 16 字节对齐时才开放；其他平台保留普通指针路径。TensorDescriptor 路径要求对应 Triton API，3.3 基础路径仍可运行非 TMA 实现。

`workspace_memory_limit_bytes` 可在实验中指定，套件也支持 `--workspace-limit-mib`。Split-K、物化反量化等按该预算裁剪，默认 256 MiB 并非所有大 GPU 的固定上限。编译后可调用 `ctx.record_compiled_kernel` 记录实际共享内存、寄存器和 warps，超出已知物理限制即拒绝。

新增 `OperatorDefinition.validation_case_factory(parameters,requested_shapes)`，可按显式专家分组等参数构造合法边界输入。研究限制只作用于 requested Shapes；平台生成的额外案例使用各自合法的 Kernel 声明域，并明确记录 `kernel_defaults` scope。这样 TMA-only/Split-K-only 生产实验不会被不适用的小边界配置误拒绝，非法 requested 输入仍会失败。

## 优化参考与证据边界

- 量化融合、按列连续访问、分组 scale 广播、Split-K 参考 [vLLM AWQ 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq_triton.py)；L2 分组次序参考 [Triton GEMM](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
- 在线 softmax 与执行结构参考 [FlashAttention-2](https://arxiv.org/abs/2307.08691) 和 [官方 fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)。
- MoE 参考 [vLLM fused MoE](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/fused_moe/fused_moe/)、[grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) 和 [persistent/TMA GEMM](https://triton-lang.org/main/getting-started/tutorials/09-persistent-matmul.html)。

这些是独立实现的优化结构，未导入库内写死的最优配置表，也未使用内嵌 autotuner 抢走平台预算。性能参数由本平台算法选择并进入日志。当前证据包含 CPU 数学/格式回归、接口及空间检查、GPU 源码静态检查和生成的生产规模配置；**尚无本机 GPU 编译、正确性或性能实测，不能声称已达到专家库速度**。实机结果须在 5060 Ti、4090 或 Hopper 上独立采集。
