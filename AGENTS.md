# 实验接续要求

2026-10-29真实LLM审查任务接续还要先读`artifacts/H800_LLM_CURRENT.json`，再读其root的`state.json`、`H800-state.json`及`private/SESSION_NOTES.md`。新任务分支`research/h800-local-llm-review`；A专门训练模型延期，B为真实小模型逐步决策，C为Qwen3.8-27B生成/迭代规则。旧接口fixture不算真实模型证据。留出测试策略已冻结，不能据目标性能修改提示/规则；已经有效的GPU阶段只读恢复，不重测。

开始或压缩上下文后恢复实验时，先读：

1. `docs/REMOTE_GPU_TROUBLESHOOTING.md`。
2. 若本机存在，读 `artifacts/cross-gpu-run2/private/MACHINE_NOTES.md`、对应机器的 `H800_ISSUES.md` / `4090_ISSUES.md`。
3. 读 `artifacts/cross-gpu-run2/state.json`、`4090-state.json` / `H800-state.json` 及相应私有 deployment/environment/receipt 日志。状态与原记录优先于对话中的推测。

不要重做证据有效的5060 Ti 192观测。跨GPU Anchor采集源码固定为 `6ecde2ed6c6778bec4c68be0c326a3320212f43c`；外层资源包装器独立记录其commit/SHA。修改实现/协议后不得无说明地合并旧数据。

Git作者和提交者必须为 `MingASA <mingasa7@gmail.com>`。在研究分支工作，可推送，不合并main、不移动既有发布标签。

用户已授权4090/H800当前账户使用空闲单卡，无额外固定配额；正常有界安装、编译、验证、Pilot、Anchor和分析无需重复申请批准。每台只用一张卡，性能任务串行；每批有明确数量和超时。仍须遵循实际管理员规则。

启动前检查显存、利用率及compute/graphics进程。绝不加入、停止、修改、抢占他人任务；未知PID不能因同一登录UID就视为自己。冲突时只停止身份已核实的本次任务，保留证据并推进独立工作。GPU UUID是身份，CUDA ordinal不是。

不使用sudo，不改全局驱动/CUDA/Python/Docker/SSH安全配置，不读他人实验目录。采用登录入口的当前账户私有workspace。Docker历史排错线索不是本轮已验证结果，也不是一律添加参数的理由；当前优先宿主私有venv。

GitHub不稳定的机器优先复用已核对的Git bundle；不要反复clone。所有新故障及时更新机器手册，写清复现、错误分类、原因是否证实、有效/无效处理、下一步及原始日志位置。长任务状态必须落盘，不能依赖本次会话的进程ID记忆。

服务器路径、地址、账户、UUID、他人进程及认证资料只放本地私有且Git忽略的记录。公开手册去敏化。原始大数据单独封存与校验，不覆盖旧归档；异机备份没有授权目标，单列待办，不上传共享GPU服务器，也不阻塞实验。
