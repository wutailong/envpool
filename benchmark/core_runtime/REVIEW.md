# Core runtime review / 分支导航

核验日期：2026-10-04 UTC。历史提交链接固定到已经发布、核验过的版本；本分支新增的是未采用 player-index 方案的测试与研究记录。
这是累计修改链，不是一组互相独立、需要全部合并的补丁。

## 先看哪个版本

- **累计正确性修复基线**：[fix/core-container-ownership / 15f80b09](https://github.com/wutailong/envpool/tree/15f80b0934689f635c49ec8764e6b5211e5fd218)。包含完成通知生命周期、关闭队列、计时初始化、玩家 discount 和动态 Container 输出释放修复。
- **前一轮分配优化实验**：[perf/core-container-storage / bf2f16c0](https://github.com/wutailong/envpool/tree/bf2f16c0d724d01c480668703012342ab1f0fd4e)。包含前面的全部修改，再减少 Container backing buffer 的重复内存分配。非标量字段每次创建由 5 次堆分配降为 3 次，端到端更快尚未证实。
- **保留的直接 tuple 版本**：[perf/core-state-tuple / ce1c47f2](https://github.com/wutailong/envpool/tree/ce1c47f238a069454f732a70089d1b9857dcc6de)，[详细报告](state_tuple/README.md)。每个 CartPole state 少一次 576 字节的临时分配。固定本轮计时中，多线程 CartPole、Dummy 和 Box2D 诊断客户端有正向信号；小批量 CartPole、HalfCheetah 和完整 PPO 尚不能确认加速。
- **本研究分支**：[未采用的 player-index 优化](player_action_index/README.md)。减少了连续玩家索引分配，但目标负载没有可靠提速，普通配置有不利结果。生产 core 保持 ce1c47f2，候选补丁只作未应用记录保存。
- **原版对照**：[main / 9c31c547](https://github.com/wutailong/envpool/tree/9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3)。保持不变，作为历史对照；没有后续修复。

这些累计版本都基于第一轮 core 性能改动，**不是直接基于未优化 main 的纯修复版**。
历史导航分支 docs/core-runtime-review / b8224909 只增加文档。直接 tuple 版本 ce1c47f2 在其后。本研究分支再添加测试和记录，不应用被否决的 parser 改动。需要修复基线时，请明确选择 15f80b09。
早期 perf/core-runtime 没有后续发现的修复，不应把它的历史速度表当成当前推荐或安全保证。

## 已核验的提交链

每一行的父提交都是上一行，最后一列说明该版本新增的内容。

| 分支 | 固定提交 | 本次新增 |
| --- | --- | --- |
| main | [9c31c547](https://github.com/wutailong/envpool/commit/9c31c5478eb61d8f67f8c9a1ec2b37568f2c7ca3) | 原始对照 |
| perf/core-runtime | [7de9691f](https://github.com/wutailong/envpool/commit/7de9691f724e5739f73c83d52a0ab57bf166d37b) | 第一轮 core 运行时实验，收益和回退随配置变化 |
| fix/core-completion-lifetime | [f1728fec](https://github.com/wutailong/envpool/commit/f1728fec30423f5651584a849a4ea3f3c985674e) | 完成计数发布前缓存 batch，避免发布后继续读取已回收 buffer |
| fix/core-shutdown-wakeup | [9fff99ef](https://github.com/wutailong/envpool/commit/9fff99ef555cb0199748bbda58446d669803fa64) | 关闭时使用停止感知唤醒，避免 sentinel 覆写尚未取走的 action |
| fix/core-timing-initialization | [6685e510](https://github.com/wutailong/envpool/commit/6685e510c1fad428b02152c1a378d6289880c4df) | 初始化计时累加字段；不采用收益不明确的游标改写 |
| fix/core-player-discount | [3b00a186](https://github.com/wutailong/envpool/commit/3b00a18695cbd7b7c28a1c081c5d7c6769818d4b) | 正确初始化所有玩家 discount；零玩家不再写入相邻槽位；保留单玩家标量路径 |
| fix/core-container-ownership | [15f80b09](https://github.com/wutailong/envpool/commit/15f80b0934689f635c49ec8764e6b5211e5fd218) | 回收被丢弃和 C++ 接收后释放的 Container payload，保护 Python 所有权转移异常路径 |
| perf/core-container-storage | [bf2f16c0](https://github.com/wutailong/envpool/commit/bf2f16c0d724d01c480668703012342ab1f0fd4e) | 复用 typed owner 和 shape，取消额外控制块和 shape 分配 |

之后依次为文档提交 [b8224909](https://github.com/wutailong/envpool/commit/b8224909df38650c0f1e7aadc5004ea8d3e2eecd)，以及直接 tuple 版本 [ce1c47f2](https://github.com/wutailong/envpool/commit/ce1c47f238a069454f732a70089d1b9857dcc6de)。本研究分支以 ce1c47f2 为父提交，生产代码没有再变更。

选择后面的提交就已经包含前面的提交。不要再重复 cherry-pick 整条链。
没有修改 main，没有创建 PR、合并或发布 release。

## 测试证明了什么

Container 所有权和存储分配两轮均重新执行了 64 个 native 测试；ASan/UBSan、TSan 各 62 个测试重复 3 次；
16 个既有 Python 测试；50 次带 sanitizer 的 NumPy 生命周期/部分转换失败测试。
还检查了 5,499 个 rollout 数组、8 个 CPU XLA 记录，以及完整同步 CartPole PPO：
101 个 checkpoint、4,925 个 tensor，比较范围内最大绝对差为零。
Box2D 的 4 个测试模式场景覆盖 260 个输出记录、1,040 个内部 Container 数组和
8 个在 pool/外层数组释放后仍保留的内部数组。

直接 tuple 一轮增加 12 个用例，共通过 76 个 native 测试，ASan/UBSan 和 TSan 各 74 个用例重复 3 次；上述 Python、rollout、XLA、完整 PPO 和 Box2D 检查也全部重跑通过。

这些结论有边界：只重建了 Classic Control、MuJoCo Gym、Dummy 和 Box2D 测试客户端。
Box2D 的动态诊断字段要求 ENVPOOL_TEST；普通 release spec 不包含它们。
不是全部环境、全部平台、GPU/Container-valued XLA 或一般异步训练等价性的证明。
LeakSanitizer 在此环境未启用；泄漏修复另有精确析构计数回归，不等于整个程序 leak-clean。
玩家 discount 修复有意改变此前错误的多玩家/零玩家行为，不能笼统说所有旧结果都不变。

## 如何看速度数字

- 第一轮 [历史报告](README.md) 有 234 个样本，包含收益和回退。它不是后来累计版本相对原版的最终矩阵。
- 后续各轮主要比较上一版和新候选，不能把不同轮次的百分比相加。
- 前一轮 Container 存储分配实验保存全部 224 个吞吐样本。7 个主效应区间都包含零；Dummy 点估计为 +6.96% / +5.13%，8 线程 CartPole 为 -3.11%。不能只挑正数。
- 初始 8 次 PPO 计时中，候选中位耗时约增加 0.88%，范围不重叠。这个不利结果完整保留。
- 只追加了一次预先固定的 16 次同二进制 A/A 对照确认：主效应速率 +1.37%，区间 -0.11% 至 +2.86%，耗时范围重叠，未复现固定变慢。仍不能据此保证更快或零成本。
- 确定的收益是 Container 非标量字段每次 backing-buffer 创建少 2 次堆分配，不是整个训练内存下降 40%、RSS 下降 40% 或训练加速 40%。

- 当前直接 tuple 实验另有完整 224 个吞吐样本和 16 次同二进制 A/A PPO 对照：CartPole 256/256/4 速率 +26.10%，八线程异步 +10.11%，Box2D 诊断模式 +5.02%，Dummy 单/多玩家 +17.30% / +9.47%。这些是本窗口的配对估计，同二进制波动仍较大；完整 PPO 为 -0.55%，区间 -2.30% 至 +1.22%，不能声称训练也加速。各项区间和负向 block 均保留在报告中。

- 未采用的 player-index 方案保存 256 个吞吐样本与 16 次 PPO：目标普通多玩家 Dummy 为 -3.98%（区间跨零），小批量 CartPole -4.04%、八线程异步 -4.45%、诊断 Box2D -1.22%；HalfCheetah 的 +1.08% 也完整保留。没有为找正结果追加采样，也不把这些数字解释成已证实的精确因果回退。

## 报告入口

- [关闭唤醒修复](shutdown/README.md)
- [计时初始化与未采用的游标方案](timing_initialization/README.md)
- [玩家 discount、单玩家 codegen 和混合计时结果](player_discount/README.md)
- [未采用的 action metadata 缓存](action_metadata/README.md)
- [Container 所有权修复、负对照和成本检查](container_ownership/README.md)
- [Container 分配优化、全部计时窗口和复现方法](container_storage/README.md)
- [直接 typed tuple、逐 state 分配审计与全部速度结果](state_tuple/README.md)
- [未采用的 player-index 改动、散列玩家压力负载和全部结果](player_action_index/README.md)
- [同步 PPO 方法](ppo/README.md) / [CPU XLA 范围](xla/README.md)

每份报告旁边保留原始测量、构建参数、源码/二进制指纹和限制。没有上传二进制、模型权重、
资源包、凭证或私人机器路径。
GitHub workflows 的触发条件是 main push 或 PR（release 另有版本 tag）；当前这些独立分支
没有 hosted CI runs/status checks。这不是 CI 通过，现有证据来自已记录的本地测试。
