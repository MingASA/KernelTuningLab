# 真实本地模型与原生优化器对照

实际结果和证据范围见[审查报告](../../docs/H800_LLM_REVIEW.md)。计算快照为`af49d80`，后续warm-SMAC的SDK预检修复单独记录为`92fd52b`。A专用模型训练按用户要求留到下一轮；本例不下载模型、不训练、不调用付费API。

B用真实小模型看当前Shape、完整声明空间及已经揭示的历史，逐步选择配置。默认研究配置使用完整静态可行集合的HF token约束，不是随机池；每个参数分支仍由模型logits决定。只约束合法表示和不重复条件，不提供未测性能。构建不完整就拒绝此模式。模型每步完全移回CPU并同步、释放CUDA分配后，既有Runner才测Kernel。GPU上传、生成、卸载分别计时；这种部署的搬移成本不代表模型固有生成速度。

C使用27B的真实原文规则。`rule0`没有项目性能反馈，`rule1`使用train分割的观测摘要和共享源配置先验，validation选版本，目标测试前冻结。执行器只接受有限数值表达式，不改写模型程序。两个实测规则都是静态排序启发式，不应宣传成学习型在线Agent。专用性能模型、完整Python策略插件和外部端点仍通过已有SDK扩展，见[工作流](../../docs/RESEARCH_WORKFLOW.md)。

已有H800环境上的最小复现（在源码checkout中执行）：

```bash
# 仅指向机器上已经存在且获准读取的完整checkpoint。
python -m kernel_tuning_lab.local_llm manifest \
  --model ./existing-checkpoint --model-id Qwen/Qwen3-1.7B \
  --hash-weights --output ./model.json

# --rule使用本轮封存的真实生成artifact，或你自己由真实模型生成并冻结的artifact。
python -m examples.h800_llm_study.prepare \
  --model-manifest ./model.json --rule ./examples/h800_llm_study/frozen/rule1.json \
  --budget 8 --seed 7 --output ./study.json

# 可先预览，完全不启动GPU或模型。
ktlab research --spec ./study.json --output ./artifacts/model-example --write-only

# 正常资源准入；只占用一张获准的空闲卡。
python scripts/with_idle_gpu.py --gpu auto --expected-model H800 \
  --receipt ./artifacts/model-example.guard.json --timeout 1800 -- \
  env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ktlab research --spec ./study.json --output ./artifacts/model-example
```

准备器生成标准research JSON：三个实现、一个低M Shape、六种方法、B8，最多144次研究观测和18次独立验收。B8的DE仍主要是初始化；本轮正式B32轨迹才用于检查原生进化阶段。按`protocol.json`的完整Shape划分扩展前应先检查成本。模型manifest、输出、checkpoint与缓存均由实验账户管理，禁止上传私有服务器路径和权重到公开Git。

H800实际使用Torch2.6.0/cu124、Triton3.2.0、Transformers5.8.1、accelerate1.15.0、safetensors0.8.0；优化器SMAC2.4.1、ConfigSpace1.2.2、Kernel Tuner1.5.0。复用已验证的私有GPU环境，仅为可选推理组件建立私有环境。安装前检查pip计划，不无条件套用5060 Ti的CUDA依赖锁。缺少可读本地模型时停止模型路径，不能用fixture冒充。

`analyze.py feedback`只接受全部为train的原始v2导出bundle；混合validation/test会拒绝。原始记录、资源receipt、模型Prompt/token输出和权重SHA在私有封存中；公开摘要不包含服务器身份。`analyze.py evaluate`使用实际目标观测，拒绝跨协议/设备或不同输入内容混排；缺失和失败保持缺失与失败，不预测补齐。恢复先读仓库`AGENTS.md`和机器手册，不重跑已完成阶段。
