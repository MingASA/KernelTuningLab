# 远端GPU机器速查

恢复实验前先读本页，再读本机私有机器手册与持久化状态（入口见 [AGENTS.md](../AGENTS.md)）。这里只保留可公开的机制和版本；服务器地址、账户、路径、UUID及他人资源信息不入Git。每个新故障先留首份日志，再写分类、已证实原因、无效尝试、有效恢复和下一步，不把猜测写成根因。

## 本轮已经复现

| 机器/阶段 | 现象与证据范围 | 本轮处理与防止重犯 |
|---|---|---|
| RTX4090 / 源码 | GitHub连接超时 | 开发机生成已核对的源码bundle，经正常SSH传输；目标机本地clone/fetch该bundle，再checkout准确commit。没有重试大型远端clone |
| H800 / 源码 | ls-remote成功，clone仍在90秒内未完成 | 小请求成功不能证明大fetch稳定。保留日志，改用同一bundle；确认本次失败clone没有遗留进程，不操作无关进程 |
| 两机 / Python传输 | 33 MB独立Python发行包在短SSH传输期限内只收到约10–12 MiB | 已验证部分的SHA前缀后有界续传，最后核对完整SHA再解包。不能把部分文件当完整安装包，也不重新从零反复传输 |
| RTX4090 / 默认Python | Python3.6不满足项目≥3.10要求，PATH无合适解释器 | 在私有数据盘workspace安装新的独立Python，显式使用venv解释器；不升级系统Python、不复制旧venv |
| H800 / 默认环境和磁盘 | 默认Python3.9低于要求；系统home盘可用空间少 | 使用登录入口对应的私有数据盘目录，缓存/环境均留在此处。另有系统Python3.10可作为只读引导备选，不修改它 |
| 两机 / 源码bundle | clone bundle提示remote HEAD不可用 | 随后显式checkout目标commit，核对HEAD与clean状态；提示本身不代表源码丢失。未来增量fetch须先确认目标已有基commit |
| RTX4090 / pip | Network unreachable后显示No matching distribution | 后者是网络失败的次级错误，不能误判Torch版本不存在。IPv4也失败后停止重复直连pip；只读寻找已有运行时组件，不绕过文件访问控制 |
| H800 / 资源监测 | 严格core阶段被未归属PID拦截，未完成整gate | 保留首份partial与guard日志，不计通过。原现场不足以完全归因；已CPU复现并修复NVML退役PID与忽略父级SIGINT的边界：独立追踪自己子进程、仅忽略已证实退役的自己PID、停止已证实自己的session，补充冲突现场。未知PID仍拒绝 |
| 本地 / 控制进度 | 两个非GPU阶段并发写小JSON，成功状态被旧快照覆盖 | 原stdout/receipt可证明成功，不重跑GPU。控制元数据改为文件锁、只合并本阶段字段与原子replace；原错误快照保留。不是测量数据库损坏 |

当前共同环境候选为Python3.12.12、Torch2.5.1/cu121及配套Triton；**安装完成、CUDA可用、编译通过是三个独立状态**。不能将环境候选写成GPU验收通过。驱动535/550不套用5060 Ti的cu132依赖锁，参考[官方PyTorch版本](https://pytorch.org/get-started/previous-versions/)和[NVIDIA兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。实际版本以各次原始记录为准。

## 用户提供的H800历史线索

以下由用户在其他实验中验证或观察，**本轮尚未复现**，需要遇到相应症状后再核对，不自动套用所有参数。

| 症状 | 历史线索与检查顺序 |
|---|---|
| GitHub HTTP/2 stream error、SSL timeout、Empty reply | 优先本地生成增量bundle并传输，在目标已有仓库本地fetch。保持正常TLS/SSH验证，不反复clone大仓库 |
| Docker CUDA Error304 / cudaGetDeviceCount失败 | 既有可用容器配置曾包含 `--security-opt seccomp=unconfined`；检查获准的runtime、`--gpus`及设备映射，容器内分别核对nvidia-smi和Torch。不要据此改全局Docker服务/驱动，也不要未遇到问题就更换当前venv方案 |
| Docker Rust无法创建worker thread、Operation not permitted | 历史上遗漏上述已验证容器安全配置触发过此症状；先对照批准的容器配置，不直接重装Rust。任何明确管理员限制仍优先 |
| OpenBLAS大量pthread_create失败 | 可参考 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。这是症状相关的线程排错方案，不是所有机器的强制模板；本轮已有OMP/OpenBLAS限制是资源控制设置，不证明复现过该错误 |
| Docker数字UID没有passwd用户名 / getpwuid失败 | USER/LOGNAME变量不创建passwd条目；优先核对镜像用户映射和可写缓存。用户历史方案涉及容器临时HOME，但本轮不改宿主或当前会话HOME，不冒充已创建用户 |
| Docker通过heredoc接收Python却无输出 | 检查是否缺少 `docker run -i`，再检查脚本；不要先认定Python或CUDA损坏 |
| 宿主GPU编号与容器CUDA编号不同 | 单卡容器通常看到CUDA0。按物理UUID选择并在容器内复核；`--gpus device=<UUID>`的设备身份不能用CUDA ordinal替代 |

本轮空卡包装器核对宿主的进程树，**尚不提供Docker daemon下容器进程的身份关联**。若转用容器，需要核验自己的明确容器ID及host PID关系；不能通过忽略未知PID、只看同一UID来绕过冲突检查。

## 接续时的最短检查

1. 读取对应机器私有手册与state；核对上次成功阶段、source commit、当前环境和raw output。已通过阶段不重测。
2. SSH采用BatchMode及连接超时。认证/MFA/未知主机密钥问题停止该连接，禁止降低信任验证。
3. 网络传输查已验证bundle/部分包和SHA，包未完整不安装；安装只用指定私有venv。
4. 每个GPU命令重新进行空卡准入；预选不等于预留。若别人开始使用，只停自己的进程并隔离受污染数据，不能从该次缓存继续当作干净测量。
5. Pilot有效且资源允许时直接继续有界Anchor，再分析/Replay；不请求重复批准。全程保留独立验收、缓存、失败、实际GPU观测与离线Replay的不同计数。
