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

当前两台主实验环境已分别安装并通过probe、T1及原四类严格Smoke：Python3.11.11、Torch2.6.0+cu124、Triton3.2.0、NumPy2.2.2。4090采用既有官方镜像组件的私有离线拷贝；H800采用新私有venv。早期H800的2.5/cu121验证及失败保留在独立目录，不混入主矩阵。**安装完成、CUDA可用、编译通过是三个独立状态**。驱动535/550不套用5060 Ti的cu132依赖锁，参考[官方PyTorch版本](https://pytorch.org/get-started/previous-versions/)和[NVIDIA兼容性](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)。实际版本以各次原始记录为准。

## 本轮补充问题

- **4090完全离线的依赖恢复**：获用户只读复用授权后，从本地缓存的官方PyTorch镜像创建自己的未启动、无GPU/网络临时容器，将运行时复制到私有目录，核对Python ABI与pip安装前缀后在宿主运行。只删除自己的这个临时容器；原镜像、共享环境和他人容器不变。没有通过访问被拒绝的`/proc`文件绕过权限，也不把cp311扩展塞到cp312环境。
- **H800安装ENOSPC**：仅指定PIP_CACHE_DIR不改变pip的`/tmp`暂存。保留首份失败，将TMPDIR/TEMP/TMP设为自己的大盘临时目录后，同一版本安装成功；不清理共享/tmp。
- **两机GPU成功后的SSH后处理超时**：原20秒元数据deadline不足。先读原始guard与测量记录，确认exit0、eligible且物理UUID吻合，再幂等补写相同字节的sidecar；不启动CUDA、不重跑已有观测。控制状态区分postprocessing_pending与测量失败，后处理允许有界重试。不能只改成功标志而没有原receipt证据。
- **4090 Pilot噪声门槛**：首次24条数值正确，但一处tail cell两run CV6.51%超过预设5%。保留全部样本，冻结两轮各24条的独立同协议稳定性诊断；两轮都通过才继续Anchor。不删除首轮、不挑最快样本、不放宽正确性或噪声门槛。频率/输入等原因尚未证实。

## 已完成后的补充与证据限制

- 两机最小T1、十组合80条严格core、24条Pilot、192条正式Anchor和36次在线观测均已完成，主数据使用6ec，外层guard使用cbccd9d；不要重复执行已有有效证据。
- 两机没有`rg`：原编译归档脚本报FileNotFoundError。523486b helper增加仅扫描指定自有cache、拒绝跟随symlink的标准库fallback，两端均95/95真实CUBIN保存。归档修复不需要重测GPU或系统安装。
- cbccd9d guard原receipt未序列化command，分析器不能假定字段存在。保留原字节，只为确切旧SHA派生单独control binding：原driver state、原console/manifest.run_id、原guard SHA全部核对。新guard记录command/cwd；不能把新字段伪装回填进旧观测。
- SQLite封存顺序必须是quick_check→checkpoint→显式close→枚举清单→tar。4090追加noise的首个脚本反序枚举，close后WAL消失导致FileNotFoundError；首份错误保留，修复后仅重做CPU封存，不重测。
- 三卡相同CUDA随机seed的实际输入hash不同；同Torch2.6的4090/H800也不保证相同。源码在CUDA上randn，保留真实hash，不宣称输入字节或纯架构因素已隔离。
- 主4090矩阵仍有4/96 cell CV>5%。追加12观测只作诊断，tail一处仍5.50%；保留原样本及不确定性。不要无限复测、挑稳定/最快一次或改写主矩阵。

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

## 空间/生产接入轮的补充（2026-10-28）

- 4090离线优化器wheelhouse跨Python下载漏掉`python<3.12`的importlib_metadata条件依赖；补齐固定importlib_metadata/zipp并全SHA核验后，安装到新私有环境。网络错误不等同于版本不存在，原GPU环境未改动。
- 固定vLLM0.8.5官方wheel在H800仅CPU加载；其卡被占用未执行GPU。4090以既有CUDA12.4/GCC11、单编译worker构建原封数值源码及两个schema注册，真实Marlin已验证。源码build与官方wheel分别绑定binary SHA，不能混称同一产物。
- 4090长SSH丢失后，原受控研究实际已完成800观测；只读恢复terminal guard和SQLite，未重测。私有poll模板另有缩进错误，最小CPU复现后修复并保存poll stderr；不能把这个错误归因CUDA或全部归因网络。
- 大传输采用有界4/8/32KiB确认帧与prefix SHA续传，最后核完整SHA才解包。既有配置publickey认证、正常known-host验证均保留；没有改密钥/全局SSH/驱动。此措施改善本次恢复，不把网络具体根因写成已证明。
- AI同一job有两个Shape却只给一个独立验收点，初次partial保留；只增加验收预算，已有72研究观测全部cache恢复。观测预算不等于验收预算。
- Torch2.6实际probe的shared memory字段仍为0/unknown，编译器报告本卡101376。原记录不补造规格；校准于sm120/Triton3.7的估算改advisory，真实编译资源另验。规划profile与实际probe不能混用。

本轮所有新GPU批次已结束。继续前仍先读本轮私有SESSION_NOTES、state和对应receipt；不要因为控制/下载失败而重测已有证据。原始大数据只回传开发机并私有封存，未经授权不作为共享服务器备份。

## H800真实本地模型轮（2026-10-29）

本节为后续独立实验；上节“已结束”只指空间/生产接入轮。接续入口为`artifacts/H800_LLM_CURRENT.json`。

| 实际症状 | 有效处理及证据边界 |
|---|---|
| Transformers5完整模型加载成功，loading report中的set使JSON记录失败 | 递归保留set内容为排序列表；原失败保留。不能据此重装CUDA或声称推理已通过 |
| thinking输出达到8192tokens仍无最终程序 | 推理完成与程序契约通过分开；保留全部原文/成本。模型收到静态错误后自行给出简短最终JSON，执行器不手改表达式 |
| 0.8B复制嵌套参数说明，随后省略外层config | 输入改为完整scalar choices。只对键集合完全一致的标量对象补外层包装，记录原文；绝不替模型补参数或选择值 |
| 小模型反复推荐已测配置，静态重试仍重复 | 原freeform轨迹保留为失败/消融。可显式启用HF prefix约束解码：完整静态可行集合，不提供性能标签，模型logits决定每个分支，排除已揭示配置。构建不能完成时明确拒绝，不换随机池或启发式 |
| 安装SMAC2.4.1时only-binary找不到包 | 固定版本需要sdist/已核对构建wheel；复用已有纯Pythonwheel并校验SHA，其余依赖装到新私有环境，不改原GPU栈 |

H800单80GB卡已实际完成完整Qwen3.8-27B BF16推理；小模型0.8B/1.7B/4B的全空间约束Pilot共12次真实研究测量通过。B每步权重上传、生成、完整CPU卸载和CUDA分配检查单独记录后才调用Runner。该部署的搬移成本不能冒充模型本身的生成成本；SDPA/Transformers推理速度也不代表优化后的服务后端。正式结果见[本轮审查报告](H800_LLM_REVIEW.md)，不从Pilot推断搜索胜负。
