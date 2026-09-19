# 故障排查

先找到能够复现问题的最小边界。`vflash doctor` 检查可见 GPU，`vflash profiles` 列出受支持的
profile 合同，`vflash plan` 可以在不加载模型权重的情况下验证设备选择。

## 启动与运行资源

| 现象 | 检查 | 处理 |
| --- | --- | --- |
| 没有显示 GPU | `nvidia-smi`、容器设备选择 | 只暴露预期单卡或卡组，再运行 `vflash doctor`。 |
| profile 与设备不匹配 | 算力架构、显存和卡数 | 选择对应的 SM86 或 SM89 profile。修改名字不能转换 artifact。 |
| prepared assets 被拒绝 | 模型、adapter、schedule 和目标架构身份 | 对最终只读资源快照重新运行 `prepare-pipeline`，不要修改已有 receipt 背后的文件。 |
| Triton 无法加载编译模块 | cache 是否可写且允许执行 | 挂载持久、可执行的 cache；`noexec` 临时目录不能加载 Triton 动态库。 |
| 首个请求明显更慢 | 初始化、shape 编译和冷文件页 | 调用 `pipeline.prepare()`，预热生产 shape，并分开报告冷、热延迟。 |

基础包不包含 PyTorch 或模型文件。根据接口安装 `gpu`、`pipeline` 或 `server` extra；MP4 交付还需要
`ffmpeg` 和 `ffprobe`。

## 内存错误

要区分 PyTorch 已分配显存、PyTorch 保留显存、整卡显存占用和主机 RSS。3080 的 streamed worker
会在主机内存保存大部分权重，因此显存占用低不代表容量充足。

- 在把 OOM 当成 profile 上限之前，先停止无关 worker。
- 已测原生 worker 至少从 64 GiB 可用主机内存起步；完整 pipeline 需要明显更多。这不是任意 shape
  的容量承诺。
- 先使用 profile 已测的画布和时长，再一次扩大一个维度。
- 不要隐式启用第二张协作卡；必须显式选择，并让整个卡组只有一个生命周期 owner。

CUDA 失败后应关闭 session。如果不能确认设备完成或 NCCL 已清理，退出 worker 进程；不要复用该
context，也不要重置其他进程持有的设备。

## 双卡执行

两张卡必须是相同的受支持架构，并使用兼容的 prepared artifact。当前完整 pipeline 的协作策略是
`sequence-head`，其他组合会在加载模型前失败。启动前检查两张卡的现有进程和显存。

`sequence-head` 使用本地 NCCL group；任一 peer 失败都会中止整条 lane。hang、timeout 或不完整清理
都属于 worker 终止错误，不能在同一进程里继续重试 collective。

精确直接重排默认启用。仅进行受控回归诊断时，启动全新 worker：

```bash
VFLASH_H3_DIRECT_RELAYOUT=0 vflash generate ...
```

它只恢复旧的物化拷贝，不改变 collective 布局；这不是推荐的吞吐配置。详见
[Sol-Engine 对齐](../reference/sol-engine-alignment)。

## 输出与质量

成功编码 MP4 不等于提示词质量合格，也不等于数值等价。先检查帧数、时长、尺寸、音频布局和完整结束，
再针对提示词与参考素材检查多帧、正常速度播放和实际音频。

首帧或尾帧细节异常时，应分别比较条件图像、官方 VAE 重建和显式的
`keyframe_delivery_profile`。精确端点交付是 decode 后策略，不能证明中间模型帧保留相同细节。

性能候选对照必须固定模型、adapter、schedule、提示词、参考、seed、画布、时长和设备组。MP4 变化可以
只是实现差异，不一定是质量失败，但在晋级前仍必须完整复核。

## 有用的诊断信息

只在一次归因运行中使用 `--profile-denoise`，不要给所有吞吐运行都加 profile。保留结果 JSON、软件版本、
profile ID、GPU 型号/数量、拓扑类别和有界温度记录。不要公开提示词、参考素材、本机路径、设备 UUID 或
模型文件；也不要在每个请求中哈希数 GiB 的模型，准备 receipt 已经绑定不可变资源。

若问题仍可复现，提交 issue 时请提供最小、可公开的合同，并说明资源清理是否得到确认。
