# Cross-GPU Anchor v1

16 个显式配置、6 个历史探索 Shape、FP16、输入 seed 7/19；Graph512、warmup10、31 samples。先 Pilot，再每卡 192 次新观测，独立验收另计。[冻结设计](frozen-design.json)列出选择依据、成本上限和硬件留出协议。

使用报告声明的准确实验 commit。三个设备分别独立输出、独立编译；hardware=null 表示真实 probe。此配置不授予远程 GPU 权限，必须先满足服务器资源分配门禁。

```bash
# 项目独立环境，已确认该 GPU 可用；最多一个编译/测量 worker。
timeout --signal=INT --kill-after=30s 600s .venv/bin/python scripts/collect_phase1.py \
  --spec examples/crossgpu_anchor/pilot.json --output artifacts/crossgpu-local/pilot
timeout --signal=INT --kill-after=30s 1200s .venv/bin/python scripts/collect_phase1.py \
  --spec examples/crossgpu_anchor/anchor.json --output artifacts/crossgpu-local/anchor
```

相同命令和输出目录恢复；原 attempts、失败和兼容缓存保留。不要在无授权的共享服务器直接复制执行，不使用旧 GPU 的编译缓存。`--write-only`可通过原 `ktlab research` 预览展开维度。

`online.json` 与 [online-design.json](online-design.json) 是后续有界在线复核：3 Shape、每种算法4个观测、16个固定候选、输入seed7；Random、Heuristic和仅使用5060 Ti源矩阵的固定优先序。算法插件只包装原搜索器，不读取目标Anchor性能表。源矩阵按SHA256固定；源历史获取成本应另列。每卡最多36次新搜索测量与36次独立验收，算法间允许复用兼容缓存并分别计数，不能据此比较独立冷启动墙钟。一次GPU执行只有一条算法轨迹，多个Replay随机seed不冒充物理复测。

```bash
# 使用独立新输出；共享服务器仍须资源准入、UUID复核和外层1200秒期限。
python -m kernel_tuning_lab research --spec examples/crossgpu_anchor/online.json \
  --output artifacts/crossgpu-local/online --write-only
```

主Anchor GPU采集源码仍固定6ec；在线插件是额外有版本的控制输入，执行时另记录其commit和SHA，不能把它写成6ec原有文件。
