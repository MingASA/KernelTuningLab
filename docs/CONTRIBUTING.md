# 贡献算子、实现、搜索算法与实验

先完成[Quickstart](QUICKSTART.md)，从工作模板开始。使用独立分支，保留已有实验和用户改动；提交前检查状态、作者身份与暂存内容。

## SDK入口

```bash
ktlab scaffold kernel --operator gemm@1.0.0 --runtime python --output my_kernel
ktlab check --spec my_kernel/experiment.json --output artifacts/my-check
ktlab run --spec my_kernel/experiment.json --output artifacts/my-run
ktlab scaffold algorithm --output my_algorithm
```

Kernel模板复用已有算子的输入、参考和容差，替换build即可。新算子才需提供 `OperatorDefinition`。搜索器实现 `suggest(hardware, operator, shape, space, history, budget, batch_size)`，仅使用已揭示历史；排除失败/已尝试候选，零预算不返回建议。字段、条件空间、工作区、入口与包发现统一见 **[EXTENDING.md](EXTENDING.md)**。

## 身份与可复现性

- 明确 `id@version`；同名同版本不同内容不会静默覆盖。发布行为改变时递增版本。
- 实现、参考、容差、空间约束与helper列入 `source_files` / asset metadata。平台不推测任意Python传递导入图。
- 外部编译器、模型和库版本显式记录。库名不等于实际后端；FlashInfer等对照还须标明FA2/FA3/auto实际dispatch、布局、Tensor Core路径、split/merge和Graph/eager口径。
- 历史数据保留原record IDs、内容指纹、划分与获取成本。内置Transfer及暴露 `training_records` 的插件将实际数据绑定到trial和报告；数据库路径不是身份。私有模型另用版本化metadata/source_files保存训练清单。
- 插件为可信Python代码；进程期限用于故障隔离，不是安全沙箱。

## 验证与提交

先测小Shape、尾块、重复调用和支持dtype，保留首败。输入只读、输出不与输入共享可变存储。多步骤initialize/steps应包含必需GPU工作；setup仅一次。保持原容差与完整计算范围。

研究预算限制已揭示候选，失败与缓存都计数，新物理执行另列。编译/参考/验收/capture/job墙钟可追溯，缓存共享成本不能证明独立冷启动胜出。见[研究协议](RESEARCH_PROTOCOL.md)。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m pip wheel --no-deps . -w dist
```

再在独立环境安装wheel，从源码目录外运行scaffold/check/run。GPU回归显式opt-in、串行；真实错误先最小复现再修复。测试验证实际行为和证据边界。

提交说明写清问题、最终行为、验证、原始数据位置和限制。代码、规格、精简统计和必要SVG进Git；大型SQLite/JSONL、日志和编译产物单独归档校验。检查凭据和绝对本机路径，勿假定Git保存 `artifacts/`。原始数据仅迁移到授权目的地。
