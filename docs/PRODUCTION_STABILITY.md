# 生产平台稳定性收尾与交付

本轮分支 `platform/production-stability-delivery`，基线 `62e77a8`。在下述固定版本、准备契约和运行检查下，可以作为研究基础设施使用；没有发现阻塞已验收路径的问题。不能据定向验收推断任意配置、输入或设备全部正确。新候选始终经过严格参考验证，编译失败、资源拒绝和数值失败仍是搜索的真实反馈。

本轮只修复和验收，没有模型调用、API 调用、训练、算法排名或新的性能矩阵。原 640 次补充研究、5060 Ti 数据、旧暂停研究及其失败均未改写或重测。完整机器信息、原始数据库和控制日志只留在 Git 忽略的私有交付包。

## 实际问题、根因与处理

### GemLite W2 Split-K16：非活动分片越界读取 scale

原失败为 FP16、group128、M97/N768/K768、seed41 的四个显式配置，均含 Split-K16。原逻辑输入与 packing 完整，输出出现非有限值。上游 INT Split-K 对 A/B 尾部有 mask，对 group scale 的加载没有相应 K/N mask；每个分片执行 `ceil(K/(BK*split))` 次循环，因此非活动 K 分片仍会访问逻辑 scale 末端之外。

适配层现在按真实 BK/BN/Split-K 发射范围分配有限零填充的 scale 元数据，保留逻辑 K/N、量化编码、原生数值 kernel 和所有配置参数。通过 tensor 参数传递准备结果，不修改原生注册 Parameter。原生默认路径使用完整原生 proposal 范围及当前 Shape 的 cache override 求安全界；跨越量化 group 的原生 cached BK 明确拒绝，准备内存仍受资源上下文预算约束。该修复仅覆盖已审计的 group-scale/scalar-zero 契约。

四个旧失败配置现在均通过原 Runner 的严格验证与每点四次额外执行；另一个 M17 的真实原生默认 Split-K 通过。安全分配后将非活动 scale 尾部置 NaN 的受控负对照产生 74,496/74,496 个非有限输出，支持上述读越界机制。同轮未经填充的原始对照恰好通过，说明旧问题取决于越界位置内容；不声称每次必现。首次修复错误地向 Parameter 赋 Tensor 的四个准备失败完整保留，随后仅修正参数传递。

### FLUTE：原 Mixed 的低精度跨 CTA 归约

固定 0.4.2 源码的 Mixed 分支使用 FP32 MMA，但 StreamK 中间归约类型为输入 FP16/BF16，并有原子归约顺序差异。BF16 显式模板8与原生默认在完整清空 workspace 后仍失败；逻辑与准备输入 hash 未改变，排除了以旧 workspace 内容解释这些失败。旧 FP16 模板31的测后失败保留；本轮 Mixed 重复诊断全部通过，因此该 FP16 失败具有间歇性。

新生产构建使用上游已有的 High 分支：FP32 MMA、FP32 StreamK 归约、最终输出转换。只改变生成 dispatch 的积累模式枚举，不修改原生数值函数或 header。High 与 Mixed 的二进制、manifest、配置属性和缓存身份不同，不能将旧 Mixed 数据重命名为 High。构建入口现在默认 High；`--accumulation mixed` 仅用于显式复现旧实现，其结果仍须通过相同严格数值检查。

24 个 P32 W3 模板针对 sm89/sm90 构建完成，CPU 构建耗时 441.252 秒，扩展 SHA256 为 `a12e9f5f8f87644429e052ae551fd2e8c4d6d0eacbae35ace64c45757d96cdbb`。未把“24 个已编译模板”解释为全空间数值通过。

- H800：旧 FP16 模板31、BF16 显式8、BF16 原生默认，以及旧默认实际几何6/7，共五点严格通过；每点八次复用/清空 workspace 检查均零错误，最大绝对误差为0。
- 4090：High BF16 显式8、FP16 显式8、FP16 原生默认通过；每点八次附加检查均零错误。
- 原 Mixed 保留为上游复现 profile。其已知失败由严格 Runner 拒绝，不能承诺原 Mixed 的任意 BF16/FP16 候选都满足平台容差，也不将个别失败解释为整个 FLUTE 不支持相应 dtype。

数值容差仍为 FP16 atol/rtol=0.01/0.01、BF16=0.02/0.02，没有修改。Mixed/High 对应诊断点的逻辑输入身份完全一致。

### FLUTE：编译成功不代表设备可发射

4090 上 High 模板31的 CUDA invalid argument 已继续追查：CPU 用原始 Config/CUTLASS header 编译 `sizeof` 探针，得出该模板实际动态 shared-memory 请求为 **119,072 bytes**，而该卡 CUDA driver opt-in 上限为 **101,376 bytes**。H800 的实际查询值为 **232,448 bytes**。模板8仅需49,440 bytes。

新增有 header SHA 绑定的精确资源 facts；执行时与真实 driver 上限比较，超限返回 `unsupported_preparation`，在 packing/候选发射前拒绝，不使用架构名称或性能估算作全后端禁用。4090 模板31的新拒绝路径已经实际通过；H800 上限仅查询，没有重测已通过 kernel。group128 下精确超出本轮4090上限的模板为6/7/18/19/30/31；完整 group64/128、两 dtype 数值见 [资源事实](../examples/production_stability/FLUTE_RESOURCE_FACTS.json)。未知 driver 上限不能被伪造为已知支持。

### 身份缓存和实际导入源码

最终 CPU 检查发现，本机同长度文件快速原地改写可能具有完全相同的纳秒 mtime/ctime。仅依赖 stat 的缓存可能复用旧配置表身份。现在 ≤1MiB 的源码、表和清单直接按内容取 SHA；大二进制作为 worker 的不可变输入，缓存同时绑定 device/inode/size/mtime/ctime。确定性回归冻结 stat 仍能检测内容变化。修改已经加载的扩展或原生 selector 后应启用新的独立 worker，不能靠磁盘身份变化替换进程内已加载代码。

另发现一次控制层沿用旧 PYTHONPATH：新脚本实际导入旧 `4f32c78` 包。原三条记录和失败均未改写；通过模块路径与 kernel fingerprint 确认实际来源，两个有效数值点只归属实际 `4f32c78`。控制层现将当前 checkout 放首位，并在 GPU 工作前核验实际 package root；修复后的 `a8f5912` 才计入新资源拒绝门禁验收。没有将先前的失败当作新门禁已通过。

## CPU 空间构建与访问

同一 H800 主机 CPU、同一真实 MoE 组件和历史任务上下文，CUDA 不可见且未导入 Torch。每版本一个未加 profiler 的完整构建；另存一轮 cProfile，不能把 profiler 时间当普通墙钟或 GPU 时间。

| 指标 | 原始 `f9b5b38` | 初次修复 `ed1a488` | 最终身份修复 `f310d21` |
|---|---:|---:|---:|
| 完整3200点构建，未加 profiler | 47.9258 s | 0.52055 s | 0.50050 s |
| 相对原始加速 | 1× | 92.07× | 95.75× |
| 完整配置集合摘要 | `c7df7841d66abac427658c1b` | 相同 | 相同 |
| task support 检查 / 构建 | 3201 | 1 | 1 |

原 cProfile 总墙钟123.847秒，其中 execution_identity 累积117.920秒，调用3201次、file_identity 518,562次，约156万次 stat。根因是逐候选重复执行任务级组件身份/支持检查。现在每次构建检查一次任务级支持，并在前后绑定运行时身份；每个候选的参数、结构规则、opaque 配置约束仍执行。完整集合缓存绑定硬件/算子/Shape/声明和运行时身份；文件、表、环境变化后失效，不把采样前缀缓存为完整空间。

空间访问审查覆盖五个 BM 区域（每个四点，包含 stage5）、完整集合64次均匀无放回抽样、Cartesian 末端点及全排除耗尽。原全排除请求会无效随机重启到预算；现仅在已有完整集合能证明全部被排除时立即耗尽。审查总 CPU 时间0.359秒。大型未完整构建的空间仍按依赖/结构规则按需生成，明确为非均匀采样；没有通用512随机池、固定前缀或将构建预算耗尽冒充空空间。完整数据与 profile 函数统计见 [CPU_PROFILE.json](../examples/production_stability/CPU_PROFILE.json)。

这些是单次定向工程数据，未增加 CPU/GPU 性能对照矩阵；最终缓存加固没有削弱身份失效以换取速度。

## 最终支持边界

“静态合法”“编译成功”“数值验证通过”分别记录。下表的 GPU 结论仅覆盖实际任务、固定私有运行环境与记录中的配置，新增候选照常验算。

| 生产算子 / profile | 已有与本轮有效 GPU 证据 | 已知约束和未验证范围 |
|---|---|---|
| GemLite0.6.0.post2 A16 W4/W2，原生/显式 GEMM、Split-K | 4090 FP16/BF16 定向任务；本轮 W2 FP16 Split-K16 四点及原生默认修复通过 | uniform signed/group-scale 契约，不是任意量化 checkpoint 导入；新尾部修复 GPU 证明为 group128 FP16。GEMV/RevSplit-K 仍仅源码/CPU审计，未实机验收；缓存BK跨group拒绝 |
| FLUTE0.4.2 W3 High、24个P32编译模板 | H800 指定 FP16/BF16 原生及模板；4090 FP16/BF16模板8、FP16默认 | sm89/sm90 编译不保证所有模板可发射或数值正确；精确shared-memory上限逐设备拒绝；P64逻辑uint3准备不支持；不是W2/W4验收；原Mixed有上述数值限制 |
| FlexAttention Torch2.6 兼容 TRITON forward | H800 dense/causal FP16/BF16；实际launch参数记录 | 不是Torch2.14、所有AUTO/FLASH/DECODE路径或backward验收；其他硬件未验证 |
| SGLang paged decode，关闭PDL兼容路径 | H800 等长/不等长分页 FP16/BF16，原生/显式 | 原分页、两阶段Split-KV与merge保留；cap8及关闭PDL是明确profile；不是完整服务端、PDL开启或其他设备验收 |
| vLLM0.8.5原生MoE兼容profile | H800 FP16/BF16 balanced/skewed；两次GEMM、对齐/SiLU/加权sum；已有L3分派 | 官方固定binary + Torch2.6/CUDA12/ABI0；router网络不计。0.30为另一Torch2.13来源/CPU加载证据，未在当前目标GPU验收，不能混称 |

上轮有效 FP16/BF16、原生默认、优化器和模型证据见 [核心验收](PRODUCTION_SUITE.md)、[工程审查](PRODUCTION_SUITE_AUDIT.md)及[补充报告](PRODUCTION_AI_FOLLOWUP.md)，本轮只读复核，未重跑。

## 搜索、AI、预算与恢复审查

SMAC、KT DE 与三种 AI 接口仍共用权威生产配置空间和原 Runner/Store；本轮未修改搜索策略。原生优化器 CPU 回归12项通过，完整 CPU 回归覆盖预算耗尽、失败反馈、恢复、测量缓存、身份变化、区域补全及程序证据复用；可选依赖另在 CI 安装验证。SMAC 使用原 RF/条件空间，KT 保留原生DE持久callback和恢复前缀；编译/数值失败不能换成成功值，acceptance预算与研究揭示预算分开。有限native集合按结构处理，完整MoE空间可被这些算法访问。

A 为既有专门模型接口，不训练；B 可发完整配置/partial/region/query；C 为审查、开发、冻结后使用的完整SDK程序。CPU transport fixture 与先前真实本地模型记录严格区分。本轮模型/API调用均为0，未再次测量已有真实B/C轨迹；外部API、额外推理卡和非合作HTTP服务仍未实机验收。常驻推理仍要求实际生产者的合作门禁及资源上下文绑定，不能用一次利用率或 empty_cache 证明静默。

## 复现与交付检查

生产 FLUTE 新建私有 profile 的顺序如下，路径指向自己的固定源码/CUTLASS/toolkit；不覆盖旧 Mixed runtime，不修改全局环境：

```bash
PYTHONPATH=. python scripts/probe_flute_resources.py \
  --source "$FLUTE_SOURCE" --cutlass "$CUTLASS_SOURCE" \
  --nvcc "$PRIVATE_NVCC" --compiler-bindir "$PRIVATE_COMPILER" \
  --output artifacts/flute-high-resource-probe
PYTHONPATH=. python scripts/build_flute_subset.py \
  --source "$FLUTE_SOURCE" --cutlass "$CUTLASS_SOURCE" \
  --work artifacts/flute-high-build --architectures '8.9;9.0' \
  --templates 4,5,6,7,8,9,10,11,16,17,18,19,20,21,22,23,28,29,30,31,32,33,34,35 \
  --accumulation high --resource-facts artifacts/flute-high-resource-probe/resources.json \
  --cuda-home "$PRIVATE_CUDA" --build
```

设置 `KT_FLUTE_BUILD_MANIFEST` 为对应已核对二进制的 component manifest；本轮归档另提供绑定原manifest SHA的派生resource profile。全新进程加载对应runtime，不能沿用旧PYTHONPATH或进程内扩展缓存。验收case列表 [acceptance_cases.json](../examples/production_stability/acceptance_cases.json) 只描述已验证点，不是需自动重跑的队列。

本轮完整 CPU **529 passed / 12 skipped**；12项可选原生优化器通过（6条上游warning）。实际 GPU 共25条尝试：15 success、3 correctness_error、6 compile_error、1预期 invalid_config。其中13条为修复接受点，2条为诊断通过，另有116次附加诊断/验算。八个GPU门禁全部passed/eligible；没有隐藏失败或将所有尝试写成通过。公开逐点结果见 [verification.json](../examples/production_stability/verification.json)。

新原始封存166文件、7个SQLite恢复成功，25条导出记录与恢复数据库逐条相符；10个被引用的PTX/CUBIN已找回并核SHA。High构建原文、CPU profiles、首份失败、源码归属更正与机器排错记录一起保留。最终分支CI、独立wheel实际安装/CPU运行、完整源码bundle恢复与交付包SHA均以私有最终receipt为准，入口 `artifacts/PRODUCTION_STABILITY_CURRENT.json`；main与已有标签不变。

已验收profile无已知阻塞使用的问题。剩余边界是未验收路径/设备、原Mixed精度、设备逐模板资源限制及外部API/额外设备；这些均不应被表述为已经通过。异机备份没有授权目标，仍列待办，未将本地私有交付上传共享GPU服务器。
