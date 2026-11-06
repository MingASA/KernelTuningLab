# 生产平台审查与补充实测入口

后续稳定性收尾已完成调查与定向修复，见 [PRODUCTION_STABILITY.md](PRODUCTION_STABILITY.md)。下文保留补充研究开始时的历史审查，数值根因与最终支持profile以新报告为准。

审查基线 `b97649e`，原始要求为恢复会话保存的核心长 prompt。审查不重测有效 FP16 数据，不以 CPU fixture 代替真实模型或 GPU 证据。原始交付和失败保持独立封存；补充研究使用新目录与新冻结协议。

| 原始要求 | 审查结论与证据入口 |
|---|---|
| 五类生产算子、默认与显式配置 | GemLite W4/W2、FLUTE W3、Flex、分页 SGLang、原生 vLLM MoE 均有实际适配与 FP16 实机记录，见 `examples/production_suite/verification.json`；兼容版本、二进制身份与限制独立记录 |
| 上游空间、静态空间、编译集合和实测集合分开 | NativeCandidate、硬约束和 FLUTE build manifest 分开；逻辑 uint3 packer 当前只支持 P32，24 模板不冒充全部 36 个 |
| packing、输入完整性、初始化与归约范围 | suite_inputs、ExecutionPlan、prepared input hash 与各后端实际调用路径；没有 Torch 数值替身，Split-K 清零与分页合并仍在完整执行范围 |
| 大空间按需访问、条件/类别/原生模板 | ConfigSpace/SpaceView、space_tools 的有界构造；没有通用 512 点池；CPU 覆盖、稀疏约束和恢复检查已存在 |
| 原生优化器与成本 | SMAC RF 与 KT DE 使用原生引擎；已有真实 GemLite 32 次尝试进入模型/进化阶段；11 个编译失败保留，不能全部算通过 |
| 三条 AI 路线、API/本地正交 | A 提供既有外部模型接口，不训练；B 支持完整配置、partial/region/query；C 支持完整 SDK Searcher 提交、审查、开发、冻结与测试。真实 API 因费用/出站授权边界未执行，属于明确允许的待验收项 |
| 常驻和静默证明 | 同进程合作门禁与实际生产者同步；六条固定诊断输入 hash 配对恢复通过；旧 command_failed 批次未改为 eligible |
| 独立推理设备模式 | 审查发现 LocalModelClient 之前只暴露常驻/卸载，独立设备仅有协调器原语；现补显式 inference_device_uuid 路径，设备必须已可见并独立准入，不自动开放设备。CPU 设备身份检查通过，本轮仍每台只实际使用一张卡，没有双卡实测声明 |
| C 历史证据完整性 | 发现复用入口未核对父评测 spec 与 manifest.spec_sha256。先保留能复现的失败，再补 SHA 检查；已有 C1 原始 spec/summary 全部 SHA 相符，无需重测。没有发现已有证据损坏 |
| 原生规则、表与 L3 | 真实 MoE 两 Shape/两配置矩阵、K=1/K=2 表及 seed29 实际分派通过；未知 Shape 拒绝；K 不代表物理二进制/Graph/packing 预算 |
| 新用户安装、CI 和封存 | 原交付 core wheel 独立安装、源码 bundle 实际恢复、515 项 CPU 与三项 CI 成功；1,491 文件/47 SQLite 原始证据已回传并恢复核验 |

BF16 是本轮审查的实机覆盖缺口，尤其 Flex 的明确 FP16/BF16 契约。使用同一原生路径的新 BF16 输出做小规模定向验收，既有 FP16 记录不重跑。GemV/RevSplit、vLLM0.30、额外设备与外部服务的未实测状态仍需逐项标明，不能从兼容路径泛化。

补充比较从新生产任务开始，先冻结输入/配置空间、模型/生成策略、预算和反馈边界，再打开测试性能。B 采用真实已有小模型；C 用已有 Qwen3.8-27B 生成并依开发反馈迭代。Random、SMAC、KT 的配置权威空间与 AI 一致；失败、耗尽、费用和不同资源模式分别分析。核心工程中的两点 grammar 和相同 C 程序证据复用不作为补充研究结果。

私有恢复入口为 `artifacts/PRODUCTION_FOLLOWUP_CURRENT.json`。旧研究的 PAUSE_REQUESTED 不删除、不自动恢复；任何后续研究不能把已见测试重新标成盲测。

## 本轮实际审查结果

两项修复的源码为 `d306b93`，完整 CPU 回归 518 passed / 12 skipped。父 spec SHA 缺口已用实际失败复现，旧 C1 的原始 check/development spec 和 summary 均与原 manifest SHA 相符，因此未重测旧证据。独立设备显式 loader 仅作 CPU 身份验证，本轮没有使用额外卡。

BF16 新定向验收：4090 GemLite W4/W2 原生/显式共6条研究观测、6条独立验收通过；H800 Flex dense/causal、SGLang equal/ragged、MoE balanced/skewed 对应12个 job 通过。FLUTE 默认及显式各1条独立数值验收失败，最大绝对误差均为0.125，输入及packed hash保持不变，既有 atol/rtol=0.02 未修改。两个模板均为原生 kMixed 路径，但该数值误差的具体根因尚未证实；不将其写成全架构 BF16 不支持，也不把失败改为通过。H800 整批18条研究观测、28条独立验收保留为原 command_failed 状态，外部进程冲突计数0；已通过的 job 不重跑。

据此，核心平台可在明确支持边界下开展补充 FP16 研究；不能宣称 FLUTE BF16 或任意版本/设备全覆盖通过。新研究分支 `research/production-ai-followup`，冻结协议与执行组合入口见 `examples/production_ai_study/`。

补充轮现已完成，见[实测报告](PRODUCTION_AI_FOLLOWUP.md)：新任务640次揭示、80次真实B调用、两次27B程序生成，全部原始失败和有界运行时修正保留。另有4个FP16 GemLite W2 Split-K16配置及1个FLUTE配置数值复核失败，Runner严格拒绝，未修改容差或借测试结果修剪冻结空间。空间静态可行、编译通过、参考验证通过仍是三个不同层级。
