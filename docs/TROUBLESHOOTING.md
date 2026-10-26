# 排错入口

先保留原目录。manifest定位job，`preflight.json` / `acceptance.json`定位阶段，`worker.log` / `failures.jsonl` / `trial_failures.jsonl`定位case/config。原始invalid/compile/correctness/runtime状态不可改成成功。重跑归档旧attempts，无需删库。

| 症状 | 核查与处理 |
|---|---|
| venv提示ensurepip缺失 | 完整Python或已有 `uv venv --python 3.12 --seed .venv`，无需改系统CUDA |
| CPU安装成功但无Torch/Triton | 核心包无GPU依赖；在独立环境显式安装，再probe --gpu |
| CUDA deferred / sm120不支持 | 核对设备、Torch runtime/架构、Triton和驱动；nvcc不能代替wheel支持，见[5060Ti说明](RTX5060TI.md) |
| wheel找不到examples | 使用scaffold，或取得匹配commit的仓库examples |
| 插件不存在/版本歧义 | 文件相对JSON，模块须安装；使用完整id@version和正确entry-point组 |
| 同版本内容冲突 | 核对版本/source_files，行为改变递增版本，勿隐瞒源码变化复用缓存 |
| 非法配置 | 看约束与实际shared/线程/workspace；不将单卡限制全局施加到Hopper |
| 编译失败/主机OOM | 保存配置、traceback和栈；单worker先小cohort，查WSL RAM/swap，避免同时升级整个CUDA栈 |
| 数值失败 | 最小化到Shape/config/seed；查tail/mask、布局、跨word、reset/reduction、输入修改和实际输出，保持原容差 |
| Graph capture失败 | 需稳定预分配缓冲和可捕获GPU计划，host回调不重放；full-flow另存，不静默换口径 |
| 短Kernel波动 | 查Graph batch、samples、重复run、频率温度和后台；单Event可能含Host间隙，噪声内差距不作稳健加速结论 |
| 固定实现仅一个配置 | `{}`合法；预算超过1时正常耗尽，不伪造参数 |
| Replay全部无最优值 | 查split、证据、实现/协议及资产匹配。SDK CPU Kernel不用v1 CUDA约束；缺插件时只对应已观测cohort |
| 历史库原地改变 | 新trial记录训练IDs/hash/历史成本，报告隔离数据身份；保留训练库快照 |
| 重跑新增测量 | 查源码、协议、空间、环境、seed、experiment ID；被终止候选会重试，不人为改cache key混用结果 |
| 中断/超时 | 保留interrupted诊断，相同命令恢复；杀进程不承诺复位失去响应的GPU驱动 |
| L3多种计时协议 | 按source context/实现分开求解，不平均进一张矩阵 |
| 成本与latency不一致 | 样本、measurement callback、compile、验收、job墙钟不同；子项不可重复相加 |

先用 `research --write-only` 检查展开维度，再用一个小v2案例check。`--report-only`不重测GPU。同一张卡不同输出目录也须串行。
