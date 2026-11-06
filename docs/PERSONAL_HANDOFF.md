# 自己操作一次：v1上手练习

这份练习由使用者亲自执行。自动验收不能代替人工确认。Windows建议在WSL2终端操作；E盘实际挂载先检查：`df -h /mnt/e`。

## 1. 从main安装并跑CPU（约五分钟）

```bash
mkdir -p ~/ktlab-practice
cd ~/ktlab-practice
git clone --branch main https://github.com/MingASA/KernelTuningLab.git
cd KernelTuningLab
git log -1 --oneline
python3 scripts/bootstrap.py --profile core --env .venv
source .venv/activate-ktlab.sh
ktlab doctor
ktlab run --spec examples/quickstart/cpu.json --output artifacts/my-cpu
ktlab web --root artifacts/my-cpu
```

预期：`doctor.version=1.0.0`；实验`status=passed`、独立验收`passed`、四次算法观测。真实执行与缓存数以输出为准。打开`http://127.0.0.1:8765`，选择搜索过程，切换观测/墙钟；进入配置分析点击一个配置ID。完成后`Ctrl+C`。

## 2. 登录并选对生产环境

在本机设置SSH别名`ktl-gpu`或把下面别名替换为已有授权连接。账户/地址/路径见私有交付操作页。

```bash
ssh ktl-gpu
# 远端：进入自己的稳定源码目录和已恢复环境
cd "$KTL_SOURCE"
source "$KTL_ENV/activate-ktlab.sh"
ktlab doctor --devices
ktlab assets list --kind kernel
```

MoE短练习使用`moe-cu124 + optimizers`：Python3.11、Torch2.6/cu124、Triton3.2、官方vLLM0.8.5组件。环境不存在时按[部署与恢复](DEPLOYMENT.md)初始化；不能把5060Ti或vLLM0.30环境误当作同一profile。

## 3. 一次生产短预算搜索：Random / SMAC

```bash
python scripts/with_idle_gpu.py --gpu auto --expected-model H800 \
  --receipt artifacts/my-moe-resource-01.json --timeout 360 -- \
  python -m kernel_tuning_lab run \
  --spec examples/quickstart/moe-h800.json --output artifacts/my-moe
```

该JSON已包含Random与`smac_rf@1.0.0`，每方法最多四次揭示，独立正确性另有预算。观察`trials.json`的`algorithm_ref`、`budget`、`cache_hits`和停止原因。资源不空闲时门禁退出，不运行GPU；保留receipt，在之后获准空闲时用新的receipt重试，不能停止未知进程。

如只想运行其中一种算法，先生成新规格：

```bash
python - <<'PY'
import json
from pathlib import Path
spec = json.loads(Path('examples/quickstart/moe-h800.json').read_text())
spec['algorithms'] = [item for item in spec['algorithms'] if (item if isinstance(item, str) else item['asset']) == 'random@1.0.0']
Path('my-moe-random.json').write_text(json.dumps(spec, indent=2))
PY
```

换成`smac_rf@1.0.0`可仅运行SMAC；新规格使用新输出目录。不要用本练习的几个点判断算法排名。

## 4. 找到实际配置、正确性和失败

```bash
cat artifacts/my-moe/run_summary.json
cat artifacts/my-moe/trials.json
ktlab export --database artifacts/my-moe/measurements.sqlite --output artifacts/my-moe-export.jsonl
```

Web配置详情或JSONL中查看`config.params`、`metadata.correctness`、`execution.annotations`及环境身份；请求参数与实际原生模板分开。成本看trial中的`algorithm_costs_ms`、`algorithm_diagnostics.space`及实际compile/measurement字段。失败保存在原始记录，不能因为最终有一个成功就丢掉。

## 5. 同步到E盘，再本地查看

先退出远端，在本机执行。将`KTL_REMOTE_RESULTS`设置为自己的远端结果路径。

```bash
export KTL_REMOTE_RESULTS='~/work/ktlab-v1/source/artifacts/my-moe'
mkdir -p /mnt/e/GPT/KernelTuningLab/records/user/my-moe
rsync -a --partial --checksum "ktl-gpu:$KTL_REMOTE_RESULTS/" \
  /mnt/e/GPT/KernelTuningLab/records/user/my-moe/
ktlab web --root /mnt/e/GPT/KernelTuningLab/records/user/my-moe
```

服务器上保存已退出实验的目录；复制后检查`run_summary.json`和SQLite，传输完整性可用第二次`rsync -an --checksum`确认没有差异。对于正在写入的SQLite，先结束自己的实验并正常关闭，不能只拷贝一份仍有未同步WAL的数据库。

## 6. 自己创建一个Searcher

```bash
ktlab scaffold algorithm --output my_search
ktlab check --spec my_search/experiment.json --output artifacts/my-search-check
ktlab run --spec my_search/experiment.json --output artifacts/my-search
```

打开`my_search/assets.py`，找到`suggest`，它接收空间、历史和预算并返回配置。模板使用CPU，可以先跑通，再修改策略。模板窗口只是教学策略，不代表平台的大空间访问能力；完整接口见[EXTENDING](EXTENDING.md)。

## 遇到问题从哪里看

| 现象 | 首先检查 |
|---|---|
| 没有python/venv | 使用python3；选择已安装解释器，或`uv venv --seed` |
| 缺少SMAC | 使用optimizers profile或兼容环境加`--with-optimizers` |
| MoE拒绝环境 | Torch/CUDA/ABI、组件SHA、`KT_VLLM085_MOE_COMPONENTS` |
| GPU资源拒绝 | 本次resource receipt；等待获准空卡，不关别人进程 |
| 恢复身份不一致 | 保留旧结果，核源码/规格/依赖，另开输出目录 |
| Web没有曲线 | 检查选择范围及旧数据是否保存observation_trace |
| 原生构建错误 | DEPLOYMENT、生产支持说明、私有机器问题手册 |

完成后由你记录：main提交、CPU结果、GPU环境/receipt、E盘结果目录、Web检查和自定义Searcher结果。人工验收状态在你执行前始终为“待执行”。
