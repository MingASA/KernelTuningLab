# 从安装到亲自运行实验

Web v2 / 1.1.0 的当前交付分支为 `feature/web-v2-workbench`；既有稳定 `main` / `v1.0.1` 保留。默认将平台、源码和完整记录放在服务器，本机通过 SSH 隧道使用浏览器；CLI / SDK 仍可单独运行。

## 0. 先从浏览器跑通远端 CPU

```bash
# 服务器：获取本轮交付源码
 git clone --branch feature/web-v2-workbench https://github.com/MingASA/KernelTuningLab.git
 cd KernelTuningLab
 python3 -m venv "$HOME/.venvs/ktlab-web"
 "$HOME/.venvs/ktlab-web/bin/python" -m pip install '.[web]'
 "$HOME/.venvs/ktlab-web/bin/ktlab" serve --workspace "$HOME/ktlab-workbench"
# 本机另一个终端
 ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8765:127.0.0.1:8765 ktl-gpu
```

浏览器打开 <http://127.0.0.1:8765>，使用服务器工作区 `access-token` 登录。**新建实验 → CPU 模板 → 预检并继续 → 提交实验**，实时查看曲线。创建入口同时支持上传 SDK 插件和原生实验 JSON；上传 Python 要确认精确版本可信。下载仅在明确选择时发生，全部数据默认留在服务器。

生产 GPU 环境须先在服务器 `server.json` 登记实际解释器、组件、合作锁及允许的 GPU UUID，再在页面中选择。完整流程见 [WEB](WEB.md) 和 [DEPLOYMENT](DEPLOYMENT.md)。

## 1. CPU 五分钟上手

这是独立 CLI 起点；已按第0节安装 Web 环境时，直接使用该环境运行下方实验，不重复 clone 或建环境。

```bash
git clone https://github.com/MingASA/KernelTuningLab.git
cd KernelTuningLab
python3 scripts/bootstrap.py --profile core --env .venv
source .venv/activate-ktlab.sh
ktlab doctor
ktlab run --spec examples/quickstart/cpu.json --output artifacts/first-run
ktlab web --root artifacts/first-run
```

Python最低3.10。缺少venv/ensurepip时可用已安装的uv：`uv venv --python 3.12 --seed .venv`。Windows原生CPU使用 `.venv\Scripts\python.exe`，GPU使用WSL2/Linux。

成功时 summary 的 `status` 为 `passed`，独立验收通过，共四次算法观测。两种方法可能访问相同配置，真实执行数和缓存数以实际输出为准。浏览器打开 `http://127.0.0.1:8765`；退出查看器用 `Ctrl+C`。

## 2. 实验 JSON 与预算

`examples/quickstart/cpu.json` 是 schema_version=2 的单算子实验。

| 字段 | 控制什么 |
|---|---|
| `operator / dtype / parameters` | 算子语义；位宽、量化group属于任务 |
| `kernels` | 实现及可选空间约束 |
| `algorithms` | 搜索器引用与options |
| `shapes / seed` | 规模与输入协议 |
| `budget` | 每个算法/实现/Shape的揭示次数与墙钟；缓存和失败也计入 |
| `validation` | 独立正确性验收预算 |
| `warmup / repetitions / timing_mode` | 预热、采样与计时范围 |

多算子/dtype/seed矩阵使用 `kind: research`：

```bash
ktlab research --spec examples/research/cpu.json --write-only --output artifacts/cpu-plan
ktlab research --spec examples/research/cpu.json --output artifacts/cpu-study
```

`run` 与 `research` 的JSON结构不同。`--write-only` 只检查和展开预算。

## 3. 找结果与恢复

| 文件 | 内容 |
|---|---|
| `run_summary.json` | 状态、执行数、观测、缓存、验收 |
| `measurements.sqlite / measurements.jsonl` | 原始样本、正确性、失败、环境和配置 |
| `trials.json`、`trials/` | 搜索器、预算、逐步轨迹、墙钟与成本 |
| `validation_report.json` | 独立验收、覆盖、失败 |
| `experiment.json / assets.lock.json` | 实际规格、资产内容身份 |
| `report.md` | 可读报告 |

相同命令和输出目录可恢复；新实验使用新目录，保留原失败和规格。

```bash
ktlab run --spec examples/quickstart/cpu.json --output artifacts/first-run
ktlab export --database artifacts/first-run/measurements.sqlite --output artifacts/first-run-export.jsonl
ktlab report --database artifacts/first-run/measurements.sqlite --output artifacts/first-report
```

## 4. 本机5060 Ti

保留原环境，使用独立profile。原验证栈为Python3.12、Torch2.13.0+cu132、Triton3.7.1；驱动和设备以实际probe为准。

```bash
python3.12 scripts/bootstrap.py --profile 5060ti --env .venv-5060
source .venv-5060/activate-ktlab.sh
ktlab doctor --devices
ktlab probe --gpu --hardware 5060ti16g
python scripts/prepare_target_gpu.py --target 5060ti --output artifacts/5060-ready
```

最后一条只生成短验收规格。确认当前卡获准独占且无冲突后，才增加 `--execute`。不要自动恢复旧192点Anchor；5060栈也不能直接套用H800的FLUTE二进制。

## 5. SSH到H800/新服务器

下文 `ktl-gpu` 是你配置的SSH别名，账户、地址、UUID留在私有记录。

```bash
# 本地：源码传输
git bundle create ktlab-source.bundle --all
scp ktlab-source.bundle ktl-gpu:~/ktlab-source.bundle
ssh ktl-gpu
# 远端：新私有目录
mkdir -p ~/work/ktlab
git clone --branch feature/web-v2-workbench ~/ktlab-source.bundle ~/work/ktlab/source
cd ~/work/ktlab/source
```

能正常访问GitHub时直接 clone 本文交付分支即可；main/v1.0.1 保留已发布 CLI 与只读兼容。SSH负责命令，Git/Bundle负责源码，rsync/scp负责数据和部署资产；CLI 可按需执行；Web 模式由独立 API 与 Worker 管理远端任务。

选择MoE0.8.5兼容环境，`KTL_COMPONENTS`指向恢复的官方组件目录：

```bash
python3.11 scripts/bootstrap.py --profile moe-cu124 --with-optimizers \
  --env "$HOME/venvs/ktlab-moe-v1" --components "$KTL_COMPONENTS"
source "$HOME/venvs/ktlab-moe-v1/activate-ktlab.sh"
ktlab doctor --devices
```

离线安装增加 `--wheel "$KTL_CORE_WHEEL" --wheelhouse "$KTL_WHEELHOUSE" --constraints "$KTL_LOCK"`。组件提取与选择见[部署指南](DEPLOYMENT.md)，不能以Marlin-only子集代替MoE组件。

运行完整生产空间上的短预算搜索：

```bash
python scripts/with_idle_gpu.py --gpu auto --expected-model H800 \
  --receipt artifacts/moe-v1-resource.json --timeout 360 -- \
  python -m kernel_tuning_lab run --spec examples/quickstart/moe-h800.json --output artifacts/moe-v1
```

Random与SMAC每方法最多四次揭示，独立验收另计。候选失败会保留；没有空闲获准卡时包装器拒绝执行，不停止他人进程。

复制JSON为新文件，修改 `algorithms` 即可切换：

```json
"algorithms": ["random@1.0.0"]
```

或：

```json
"algorithms": [{"asset": "smac_rf@1.0.0", "options": {"initial_design_size": 2}}]
```

SMAC需要 `kernel_tuning_lab.optimizers` 插件与可选依赖。修改输入、算法或预算时用新输出目录。

## 6. 按需归档到E盘与只读兼容查看

在线使用优先通过工作台选择报告、配置、SQLite 或轨迹下载，ZIP 自带 SHA 清单，SQLite 使用一致快照。完整原始记录留在远端；E 盘是离线资产库。也可在实验结束后按需 rsync 整个目录；SQLite可能有WAL，不能仅复制运行中的主库。

```bash
# 本机WSL
export KTL_VAULT=/mnt/e/GPT/KernelTuningLab
mkdir -p "$KTL_VAULT/records/user/moe-v1"
rsync -a --partial --append-verify ktl-gpu:work/ktlab/source/artifacts/moe-v1/ \
  "$KTL_VAULT/records/user/moe-v1/"
ktlab web --root "$KTL_VAULT/records/user/moe-v1"
```

远端workspace不同则替换为自己的实际路径。SHA和恢复检查见[部署指南](DEPLOYMENT.md)。远端查看可用：

```bash
ssh -L 8765:127.0.0.1:8765 ktl-gpu
# 远端另一个会话：ktlab web --root <实验目录> --port 8765
```

## 7. 创建自定义Searcher

```bash
ktlab scaffold algorithm --output my_search
ktlab check --spec my_search/experiment.json --output artifacts/my-search-check
ktlab run --spec my_search/experiment.json --output artifacts/my-search
```

编辑 `my_search/assets.py` 的 `suggest`。生成模板用于接口学习；研究可使用 `space.sample_feasible`、partial/region与条件空间，不把模板策略当成平台性能结论。算法只收到已揭示反馈，失败与尝试过的配置也属于历史。

## 8. AI与L3入口

- A：注册已有预测器，平台不代训练。
- B：参考 `examples/production_suite/local-provider.example.json`、`api-provider.json`，统一返回配置、区域或查询。
- C：`ktlab ai-program submit/review/check/development/freeze/test`；原文、审查、反馈与测试曝光分开保存。
- L3：`ktlab l3 --database <SQLite> --output artifacts/l3`；K是配置数，不能代替物理二进制或Graph数量。

API示例默认不授权真实付费调用。模型先核对来源、revision、文件SHA与访问许可，接口fixture不能代替实际模型效果。

## 常见错误

| 现象 | 先检查 |
|---|---|
| unknown research experiment fields | schema2用run，kind=research用research |
| 找不到smac_rf | 优化器插件与匹配依赖 |
| Torch/CUDA/ABI不匹配 | doctor、profile、binary manifest；新建兼容环境 |
| native default准备失败 | 实际模板、packing契约和资源原因，不改容差 |
| 资源拒绝/未知PID | 保存receipt，等待合法资源，不清理他人任务 |
| Web没有曲线/成本 | 选择完整目录；旧记录可能没有保存轨迹 |

更多见[排错](TROUBLESHOOTING.md)和[远端GPU手册](REMOTE_GPU_TROUBLESHOOTING.md)。
