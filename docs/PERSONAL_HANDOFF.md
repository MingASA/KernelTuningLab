# 个人上手：远端实验工作台

这份练习由你亲自执行。自动浏览器验收另有证据，不能代替人工验收。Web v2 使用稳定 `main` / 1.1.0；历史 v1.0.1 标签与开发分支保留，获取源码时核对实际 commit。

## 1. 服务器启动，本机连接

```bash
# 服务器；已有专用 Web 环境时直接使用它
 git clone --branch main https://github.com/MingASA/KernelTuningLab.git
 cd KernelTuningLab
 python3 -m venv "$HOME/.venvs/ktlab-web"
 "$HOME/.venvs/ktlab-web/bin/python" -m pip install '.[web]'
 "$HOME/.venvs/ktlab-web/bin/ktlab" doctor
 "$HOME/.venvs/ktlab-web/bin/ktlab" serve --workspace "$HOME/ktlab-workbench"
```

首次启动在服务器终端设置工作区密码，预期随后显示 loopback URL。浏览器使用该密码登录；之后的重启和恢复都保留密码，不需要读取 token 文件。服务无 GPU 也能启动。

```bash
# 本机新终端；ktl-gpu 替换为自己的 SSH 别名
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8765 ktl-gpu
```

浏览器打开 <http://127.0.0.1:8765> 并登录。新服务器的 GPU 环境先按 [DEPLOYMENT](DEPLOYMENT.md) 安装，再登记私有 `server.json`；已有远端部署的实际路径在私有操作页，不写进公共文档。

## 2. 浏览器提交一次 CPU 实验

**新建实验 → CPU · GEMM 模板**，保持 `core` 环境，算法 Random / Grid，观测预算各 2、墙钟 15 秒。点击 **预检并继续**，核对 2 条搜索试验、最多 4 次算法揭示和 1 次独立验收，再提交。

预期状态依次为待预检、可提交、排队、运行、完成。CPU 很短，曲线可能直接呈现最终状态。搜索过程显示真实 Best-so-far；配置结果可展开正确性和执行记录。缓存使物理执行数可能少于揭示数。

关闭页面再打开，任务仍在；刷新不创建第二项任务。服务器重启 API 后任务记录与结果仍可查看，独立 Worker 不因 API 退出而停止。

## 3. 一次小预算生产 GPU 搜索

仅在管理员允许、目标单卡空闲时执行：

1. **设备与环境** 核对实际解释器、依赖状态、获准 GPU 的物理 UUID；环境由服务器管理员登记，不在页面输入 Shell。
2. **新建实验 → Production MoE**，选择已经安装的 `moe-cu124` 环境与目标 GPU。保持模板 Shape `M=17,H=128,I=256,E=4,TOP_K=2`、FP16、CUDA Graph 完整调用协议。
3. 选择 Random 和 SMAC RF，每方法观测 2、墙钟 90 秒，独立验收 1。SMAC 缺依赖时明确修复环境，不换名冒充。
4. 预检并核对预算、配置空间、冻结身份，再提交。资源忙时保留排队，获准空闲后明确重试。
5. 打开实时搜索过程、配置结果、成本和日志。正确性以参考验证字段为准；编译/数值失败保留为反馈，不降低容差。

如需练习取消，点击取消并等待明确终态，再恢复同一任务。恢复可能重放缓存给算法，原始记录与各次尝试保留。不要重跑历史大矩阵或为了截图反复测量。

## 4. 查配置和选择性下载

- **搜索过程**：切换观测/墙钟轴，缩放或切换图例；多方法仅在相同测量上下文内对比。
- **配置结果**：按参数、状态或 ID 筛选，展开真实 launch / 模板、正确性、来源与失败。
- **成本**：查看候选提议、编译、测量、缓存及已有模型/API 字段；没有保存的成本保持缺失。
- **文件与下载**：默认选报告、summary 和规格；需要原始数据时另选 SQLite、日志、JSONL。下载 ZIP 后核对 `EXPORT_MANIFEST.json` 的 SHA。

完整实验默认留在服务器。浏览器可选择保存到 E 盘；也可在实验结束后用已有 SSH / rsync 方案归档。无需服务器持续挂载 E 盘，不能只复制运行中的 SQLite 主文件。

迁移整个工作台时，完成或取消任务，停止 API 与 Worker，再执行 `ktlab workspace archive`；新服务器 `archive restore --source workspace` 后重新登记环境。见 [WEB](WEB.md)。

## 5. 上传一个自定义 Searcher

在本地或服务器的核心环境：

```bash
ktlab scaffold algorithm --output my_search
```

将生成目录中的 `assets.py` 和 `experiment.json` 一起拖入 **新建实验**（也可 ZIP）。核对静态发现的资产、依赖和 SHA，填写代码来源并确认精确版本可信，再预检并提交。模板使用公开 SDK 的 `suggest`，不依赖 GPU 或 API。

CLI 也能独立执行：

```bash
ktlab check --spec my_search/experiment.json --output artifacts/my-search-check
ktlab run --spec my_search/experiment.json --output artifacts/my-search
ktlab web --root artifacts/my-search
```

## 6. 出错时从哪里查

| 现象 | 入口 |
|---|---|
| 页面连接失败 | SSH 转发、本机/远端端口、服务日志 |
| 预检失败 | 页面最近错误、Worker launcher 日志、原生 schema 与环境 |
| 上传后没有资产 | 静态解析清单、插件相对路径、SDK 注册入口 |
| 排队且资源忙 | 准入 receipt；不要抢占其它用户，明确重试 |
| 无曲线 | 原始逐观测轨迹；旧数据可能只有最终记录 |
| 恢复后环境变化 | 复制规格、重新登记并预检，保留旧计划 |

执行后记录：源码 commit、任务短名称/ID、CPU 与 GPU 环境、正确性结果、下载包 SHA 和遇到的问题。**人工验收在你执行前始终为待执行。**
