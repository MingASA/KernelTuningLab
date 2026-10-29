# 生产套件工程验收

从仓库根目录执行。当前分支`platform/production-suite-readiness`；状态与边界见[生产报告](../../docs/PRODUCTION_SUITE.md)。这些是小规模验收规格，不是算法性能排名或完整服务压测。

```bash
python3 -m venv .venv-cpu
.venv-cpu/bin/python -m pip install -e '.[dev]'
.venv-cpu/bin/ktlab research --spec examples/production_suite/cpu-provider.json --output artifacts/provider-cpu
.venv-cpu/bin/ktlab research --spec examples/production_suite/gemlite-w2.json --write-only --output artifacts/w2-plan
```

首命令在无Torch/模型时运行Python Kernel与明确标为fixture的CPU进程，通过同一个动作/反馈入口生成报告；不算真实LLM或GPU证据。

GPU环境按后端隔离，先核实driver/可用设备与实际软件，再安装固定依赖。GemLite固定0.6.0.post2，要求Triton≥3.6；本机已有Torch2.13/cu132与Triton3.7可供私有overlay只读复用。vLLM0.30官方包要求Torch2.13；FLUTE0.4.2先用给定源码和CUTLASS3.4.1私有构建，不能将旧ABI扩展塞入不同Torch环境。Flex可用报告中精确核对的2.13或2.14源码。GPU输入生成还需NumPy。不要用一条`pip install -U`升级全部共享栈。

对每类先`--write-only`核对数量，在空卡guard下运行去掉该选项的相同命令。相同输出目录恢复；FP16和BF16分别生成，旧尝试保留。GemLite W2/W4分开；FLUTE只将当前二进制包含的模板视为可执行。原生default可能内部调优很多次，平台B3不是内部调优上限。不要在别人占用的卡上执行。

## B：相同算法，选择API或本地模型

选择插件`kernel_tuning_lab.provider_search`、算法`ai/standard_provider@1.0.0`。provider示例见`api-provider.json`及`local-provider.example.json`。把所选provider对象放进算法options的provider字段；API授权放options.authorization，绝不放key值。API默认配置不授权付费/出站，即使存在环境变量也拒绝调用。小空间trie不是本入口的前提。

本地manifest由`python -m kernel_tuning_lab.local_llm manifest --model <已有目录> --model-id <真实身份> --hash-weights --output <私有manifest>`生成。只读复用合法已有checkpoint，不下载。常驻串行需使用已有GPU准入，并由同进程合作门禁静默；没有可信合作协议的外部同卡服务拒绝。另卡推理先单独获准并准入，不能自动占卡。新资源模式GPU验收状态见报告，不能套用旧卸载协议的结论。

## C：生成完整Searcher，审查后才执行

```bash
# API尚未授权时，此命令默认拒绝；本地模型只在获得空卡准入后执行。
python examples/production_suite/generate_program.py --provider <provider.json> --task <允许的开发任务.json> --output <新生成目录>
ktlab ai-program submit --artifact artifacts/strategy-v1 --response <原文.txt> --program <原程序.py> --algorithm-ref <id@version> --provenance <来源.json>
ktlab ai-program review --artifact artifacts/strategy-v1 --reviewed-by <审查人>
ktlab ai-program check --artifact artifacts/strategy-v1 --spec <CPU或获准GPU的v2规格.json>
ktlab ai-program development --artifact artifacts/strategy-v1 --spec <train或val规格.json>
ktlab ai-program feedback --artifact artifacts/strategy-v1
ktlab ai-program freeze --artifact artifacts/strategy-v1
ktlab ai-program test --artifact artifacts/strategy-v1 --spec <明确split=test的规格.json>
```

来源JSON至少含route=C/provider/model/evidence_level；人工编辑需`--edited-program`与edit_description，原模型版本不改。迭代用新artifact，可通过`--parent`关联冻结父版本。测试曝光在执行前落盘；修改后不能重新宣称同一测试未见。生成程序可调用SpaceView、维护状态并用已揭示反馈，不只限于评分表达式。受审代码不继承任意凭据环境，但仍是可信Python，不是恶意代码沙箱；审查时拒绝凭据/目标表读取。

## 原生规则与有限表

原生default与显式配置是独立Kernel资产。在同实现/语义/协议的实测PerformanceMatrix上使用已有L3，再由`production.dispatch.export_table`导出精确绑定表。用插件`kernel_tuning_lab.production.dispatch`及算法`production/config_table@1.0.0`，options为table_artifact及table_sha256，交回原Runner执行。未知shape拒绝；不能把配置K直接称作二进制/Graph/packing预算。vLLM原生配置JSON有独立读写接口，保留nearest-M tie-break；不得覆盖上游安装的共享配置。
