# KernelTuningLab 工作规范

## 当前入口

当前用户任务为 Web v2 远端实验工作台，分支 feature/web-v2-workbench；先读 artifacts/WEB_V2_CURRENT.json 及其 root 的 state.json、private/WORKPLAN.md、SESSION_NOTES.md。不自动恢复旧研究队列。
本轮明确允许经可信版本确认的上传 Python 在独立受限 Worker 中运行；下文 v1 只读 Web 限制只适用于保留兼容的 ktlab web。新入口 ktlab serve 始终 loopback，SSH 隧道、会话、来源与请求保护；队列元数据与原 Runner 测量库分开。不得提供任意 Shell 接口。

main为稳定使用入口，v1.0.1基础设施交付后，科研工作在独立分支开展。开始或恢复会话先读本页、docs/STATUS.md、docs/REMOTE_GPU_TROUBLESHOOTING.md。

本机有私有状态时，优先读取artifacts/FINAL_RELEASE_CURRENT.json及其root的state.json、private/SESSION_NOTES.md、private/WORKPLAN.md；不存在时再读取PRODUCTION_STABILITY_CURRENT.json和PRODUCTION_SUITE_CURRENT.json。状态、原始receipt和记录优先于聊天推测。

当前最终发布已获用户明确授权：审查和验收后正常合并main、推送并创建新稳定标签。保留历史分支和标签，不force push。后续main变更遵循当轮用户授权及CI审查，不从本条推断可发布任意后续实验。

## 数据与证据

- 本机主要保全位置E:\GPT\KernelTuningLab，WSL实际路径先核实。私有源路径、账户、UUID、他人PID及认证资料只放Git忽略的私有记录。
- 不删除远端原始数据、不覆盖旧封存；传输可续传，核SHA、清单与SQLite，实际恢复后再标完成。
- 已有效的GPU实验只读恢复、不重测。5060Ti192点和跨GPUAnchor冻结源码6ecde2ed6c6778bec4c68be0c326a3320212f43c不因平台发布改变身份。
- 区分静态合法、编译、CPU验证、GPU数值通过及未验证。修复不放宽容差，不删除失败或改写旧记录。修改协议/实现后不混合旧数据。
- 旧H800研究的PAUSE_REQUESTED不得删除或绕过。模型/优化器大矩阵、模型训练和真实付费API不能由基础设施任务自动恢复。

## GPU与环境

- 只在获准空卡上做必要、有数量和超时的定向验收；每台一张卡，性能任务串行，不无限轮询占用卡。
- 启动前检查显存、利用率及compute/graphics进程。UUID是身份，ordinal不是。未知PID不能仅因同UID视为自己。
- 不加入、停止、修改或抢占他人任务。发生冲突只停止已经通过argv/父子关系/start_ticks核实属于本轮的进程树，保留证据。
- 常驻推理采用可验证的合作门禁并绑定资源上下文；一次利用率和empty_cache不是静默证明。无逐候选全模型卸载的全局要求。
- 不使用sudo，不改全局驱动/CUDA/Python/Docker/SSH安全配置。使用私有新环境，不复制旧venv宣称跨机可用；固定源码、二进制ABI、架构与重新验证范围。
- GitHub不稳定时复用SHA核对的Bundle，不反复clone。依赖故障使用可信wheel/source缓存，不修改共享环境。

## 工程交付

- Git作者/提交者：MingASA <mingasa7@gmail.com>。
- 以后新提交、推送、PR 和发布使用 MingASA 身份，不添加 Codex 作者名或代署名；不重写历史提交。
- 实验长任务状态、receipt、日志落盘，不能依赖会话PID记忆。新故障记录复现、分类、已证实原因、有效/无效处理和原始日志位置。
- 本轮目标是远端Web工作台、真实创建/上传/执行链路和工程验收；不新增研究Kernel、搜索算法或排名。小问题不补机械测试；数据完整性、访问边界和真实接口增加有意义的回归。
- `serve`只监听localhost，独立受限Worker复用原Runner；`web`保留只读兼容。上传静态解析不执行Python，精确版本可信确认后才允许Worker执行；无任意Shell接口，不建立第二套测量数据库。
- 用户亲自上手需用户执行后确认；自动模拟不能标成用户已验收。
