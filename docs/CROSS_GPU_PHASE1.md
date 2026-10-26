# 5060 Ti交付收尾与4090/H800跨硬件Anchor

状态：本地审计完成，Anchor在目标GPU性能观测之前冻结；远端GPU执行等待资源使用授权。此文随实际验证更新，未执行项目不作为GPU证据。

## 基线、数据与交付

起点为 `research/5060ti-readiness` 的 `35ac36ed66becf19271c9ed3e97950fb67a03d87`，工作区干净。开发在独立 `research/crossgpu-anchor`，不合并main。运行时T1、参考、Runner、缓存和存储继续使用该已验证代码；实验commit另由每个manifest记录。

四个正式归档重新校验SHA256均通过：基础GPU验证、Research Workflow、Phase1及Phase2 **v2**。Phase2原始目录和已替代的初次归档不作为新实验输入。异机备份目前 **blocked：没有明确授权的目的地**；不会把共享GPU服务器用于私人备份。可执行备份方式是将四个指定归档及sidecar复制到授权路径，并在源/接收端各运行 `sha256sum -c`。未提供目标不阻止本地实验和远端只读准备。

README/Quickstart已改为checkout验证分支及准确35ac commit，避免普通clone误用旧main。建议拆分交付审查：

1. PR1：`research/5060ti-readiness` → main。标题建议“Validated CUDA research workflow, T1 datasets and readiness acceptance”。主要变更为Graph/正确性/预算/缓存加固、SDK研究入口、T1与L2/L3、CPU/wheel/plugin验收和精简研究证据；大型原始产物不在Git。验证为355 CPU通过、11 opt-in跳过，另11实际GPU案例及80严格Smoke，35ac的CPU CI通过。保留有限覆盖、无Hopper验证、Config Budget不等于compiled budget等限制。由维护者审查并授权合并。
2. PR2：本跨GPU分支，待PR1落地主线后再审。只包含本轮Anchor、必要兼容修复、去敏化结果和报告；不把新实验结果悄悄写入旧Phase2表，不创建或移动公开发布标签。

## 只读远端验收与门禁

使用现有SSH别名、正常主机密钥验证、普通登录用户，未使用sudo、agent forwarding、GPU reset/锁频或任何GPU负载。未读取其他用户文件。敏感主机、账户、UUID、内部路径和原始资源日志仅在本地private记录，公开表不含其他用户的进程信息。

| 目标 | 已确认的只读信息 | 使用门禁 |
|---|---|---|
| RTX4090服务器 | 8张4090，可见单卡约23GiB，驱动550.54.15；部分卡显存已被占用；home所在盘约33GiB可用 | blocked：尚无允许的单卡索引/UUID、执行方式及配额说明 |
| H800服务器 | 8张H800，可见单卡约80GiB，驱动535.216.03；只读时GPU空闲；系统Python3.9无Torch/Triton；home所在盘仅约3.1GiB可用 | blocked：空闲不等于已分配，尚无GPU/容器/调度和workspace授权 |

初始PATH未发现Slurm/PBS命令，没有据此认定服务器无调度规范。继续检查系统/当前用户提供的说明；在门禁满足前不安装环境、不部署仓库、不选择空卡运行，也不改CUDA_VISIBLE_DEVICES绕过分配。已向用户请求每台单卡授权、允许执行方式、最长时长/CPU内存额度和可写workspace。H20本轮不连接。

## 冻结Anchor与预算

[冻结规格](../examples/crossgpu_anchor/frozen-design.json)及[运行示例](../examples/crossgpu_anchor/README.md)定义6个Shape、16个显式配置，使用原SDK cohort插件与v2 Runner。Shape来自5060 Ti历史探索，因此不是该卡上的新盲测；选择在4090/H800延迟数据之前冻结。

Shape：M1/N2048/K2048；M4/N8192/K4096（64MiB权重）；M33/N2048/K2048；M512/N1024/K1024；M1024/N2048/K2048；M63/N1031/K1025。分别检验低M、缓存工作集、tile边界、中等/较大计算量和全轴尾块。以实际probe确定每张卡的L2及资源，未知规格不猜测。

配置包括旧通用BM32/BN128/BK32/w4/stage3/G1、BM16/32/64关键tile、BN64/128、BK32/64、warps4/8，以及配对stage2和GROUP8控制。16个配置不是192点笛卡尔域的穷举；独立验收可采样域内其他配置，但不进入锚点矩阵。

协议：FP16、连续row-major输入、seed7/19两个独立输入/物理run、warmup10、31原始样本、Graph batch512、完整GPU计划及原严格正确性；无提交者声明。

| 每张获准GPU | 观测 | 验收额度 | 硬墙钟上限 |
|---|---:|---:|---:|
| Pilot：3 Shape×4配置×2seed | 24 | 24 | 600秒 |
| 正式：6 Shape×16配置×2seed | 192 | 48 | 1200秒 |

正式采集包含3,047,424次计时内完整计划调用，不把192次record误当只有192个launch。初估正式8–12分钟，单编译/测量worker，GPU显存及主机内存各按2GiB预算；用本地Pilot校准。若远端已知额度不足，先缩小或请求确认，不自行长期占用。断线/超时保留attempts和标准SQLite，再以相同commit/spec/output恢复；不使用未经许可的常驻会话。

## 跨硬件分析协议

每GPU/软件上下文独立矩阵。共同候选仅取各卡所有选定Shape/input-run都已编译且数值通过的交集；非法、编译/数值/运行失败及未执行保留原状态。不用静态Profile替代实测，也不把不同软件栈的数据平均为同一矩阵。

逐Shape比较绝对延迟、各GPU内部排名、Spearman和Top5%/10%/20%重叠；迁移regret以目标GPU同一共同集合已测最好值为分母。source Top-K在target上的最好覆盖是事后oracle评分，不冒充B0可选出最好的预测器。

L2使用硬件留出：物理原记录不改，派生fold明确标记原record ID及train/test角色；训练不含目标GPU标签。B0仅选择source Top1；B1/2/4/8比较Random、Heuristic、source静态优先序、兼容的既有Transfer及Surrogate。算法RNG与物理输入重复分别报告；历史采集成本单列。不同Python/依赖可能使资产指纹不兼容，Transfer应标为不适用而非改写指纹强行使用。小16候选下B8已占50%，不支持“1%穷举成本”的主张。

软件栈尽可能一致，但535/550驱动不能无条件使用本地cu132依赖锁。先遵循已授权容器/模块，再选择独立兼容环境；有软件差异就记录混杂因素，不声称纯架构因果结论。

## 当前验收及后续

| 项目 | 当前状态 |
|---|---|
| 本地基线/四正式归档SHA | passed |
| 异机备份 | blocked，无授权目标 |
| Anchor配置/SDK静态资源检查 | passed，CPU证据 |
| 新最终代码5060 Ti Anchor | not_executed，下一步Pilot |
| 4090 / H800部署、Smoke、Anchor | blocked，等待明确资源授权 |
| 三GPU配对、迁移Replay、在线复核 | not_executed，依赖目标数据 |
| H20/H200 | not_executed；H20本轮不连接，之后按真实probe和单卡配额重新验收 |

只有实际新增测量才能将上述目标升级为passed。若目标权限持续缺失，保留已完成本地Anchor并交付可执行规格，报告不产生模拟的远端结论。更大矩阵需先依据Anchor的差异/噪声和服务器规则给出独立预算并取得所需授权。
