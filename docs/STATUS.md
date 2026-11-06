# 稳定版本状态

稳定入口为main，基础设施版本v1.0.0。后续研究在独立分支开展；历史研究的冻结源码、数据和标签保留。

| 能力 | 当前状态 | 证据/入口 |
|---|---|---|
| 核心CLI/SDK/CPU/SQLite/报告 | 可独立安装运行，无GPU依赖 | QUICKSTART.md、CPU CI |
| GPU Runner、参考验证、编译/缓存/恢复 | 5060Ti、4090、H800定向验证 | VALIDATION.md、各GPU报告 |
| 五类生产算子 | 固定profile下定向通过，边界明确 | PRODUCTION_STABILITY.md |
| SMAC RF / KT DE | 原生引擎、预算/失败/恢复检查 | SPACE_PRODUCTION.md、可选优化器CI |
| A/B/C三种AI接口 | 工程接口与既有本地模型证据分开 | PRODUCTION_SUITE.md |
| 外部付费API / 额外推理卡 | 本轮未真实调用/实测 | 显式授权与相应验收后使用 |
| L3配置集合与分派 | 实现、导出与定向真实分派证据 | L3.md |
| Web | 本地只读、既有实验数据、无第二套数据库 | WEB.md |
| 环境与数据恢复 | bootstrap / doctor / archive；私有资产库保全 | DEPLOYMENT.md |
| 人工上手 | 提供独立练习，用户执行后再确认 | QUICKSTART.md |

## 使用边界

静态合法、编译成功、数值通过是不同状态。Flute原Mixed可能违反严格精度契约；生产新建profile使用独立High。模板资源、packing、ABI和未验证硬件限制见生产报告，不作整个后端的泛化禁用。

核心和生产支持没有已知阻塞已验收路径的问题。新的GPU/版本需要新验收；历史性能不能自动成为新环境证据。共享模型权重、外部认证与资源许可由实际来源决定。

## 历史材料

历史阶段报告保留当时的分支、commit、未完成项和失败。安装与日常使用以当前README、QUICKSTART、DEPLOYMENT为准，不把旧计划自动当作当前队列。旧H800 LLM研究保持暂停，不恢复模型规模/算法大矩阵或路线A训练。
