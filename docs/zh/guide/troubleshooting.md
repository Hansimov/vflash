# 故障排查

先用能复现问题的最小检查：

```bash
vflash doctor                       # 当前进程可见的显卡
vflash plan <profile> --gpu 0       # 配置与设备是否匹配，不加载权重
```

对完整流水线请求，`validate_request` 在不初始化 CUDA、不加载模型的情况下执行与 `generate` 相同的准入规则
（[示例](./complete-pipeline#check-a-request-before-allocating-hardware)）。

## 请求在加载前被拒绝 {#rejected}

Vflash 不会为了"能跑"而缩小画布、改变时长或改走其他路径。报错会指出对应规则：

| 报错提到 | 原因 | 处理 |
| --- | --- | --- |
| 宽高、32 的倍数、宽高比 | 画布不是 32 像素步长或超出 1:4–4:1 | 选择合法画布。 |
| 超过 1,048,576 像素 / 原生高清 | 未使用混合图像条件路径却超过 1MP | 使用[混合模型](./complete-pipeline#extended-hybrid)并提供首帧或参考图。 |
| 最长边 2720 / 像素帧预算 | `宽×高×模型帧数 > 2048²×124`，或最长边超过 2720 | 缩短时长（4MP 为五秒请求）或降低画布。 |
| 6–15 秒需要图像条件 | T2VA 或非混合配置请求了长时长 | 使用混合 I2VA/FL2VA/Ref2VA。 |
| resident I2VA supports… | `weight_residency="resident"` 超出常驻范围（SM90：1536×864；SM120：像素帧预算；SM90采用单首帧；SM120混合流水线同时接受像素帧预算内的L2VA、FL2VA和图像Ref） | 该请求改用 `block-ring`。 |
| 模式与已准备配置不符 | 例如向关键帧配置发送参考图 | 为该模式准备对应配置。 |

## 启动与资源 {#startup}

| 现象 | 检查 | 处理 |
| --- | --- | --- |
| 未列出显卡 | `nvidia-smi`；容器的设备选择 | 只暴露目标显卡；分区容器中设置 `VFLASH_DEVICE_DISCOVERY=cuda-visible`。 |
| 配置与设备不匹配 | `doctor` 报告的计算能力和显存 | 选择匹配配置；改名不会转换已编译资源。 |
| 已准备资源被拒绝 | `prepare-pipeline` 之后移动或修改了文件 | 对最终只读快照重新执行 `prepare-pipeline`。 |
| `trust_local_code` 报错 | 官方解码器代码需要显式信任 | 审阅快照后传入 `--trust-local-code` / `trust_local_code=True`。 |
| 缺少 Sol 或 Veda | 未执行 `python -m vflash.install_sol` / `install_veda`，或版本不符 | 安装固定版本；不会静默回退到其他内核。 |
| 常驻启动被拒绝 | 启动 Python 前未设置 `PYTORCH_ALLOC_CONF` | 在启动环境中导出 `PYTORCH_ALLOC_CONF=expandable_segments:True`。 |
| Triton 无法加载模块 | 缓存目录只读或为 `noexec` | 挂载可写、可执行的缓存目录。 |
| 导入到了另一份安装 | 包装过的解释器替换了 `PYTHONPATH` | 安装到该解释器环境中，或在入口脚本中把路径加入 `sys.path` 前部。 |
| 首个请求明显更慢 | 模型加载、内核编译、冷页缓存 | 先调用 `pipeline.prepare()`，并分开报告冷、暖请求。 |

基础包不含 PyTorch 或模型文件；按所用接口安装 `pipeline`、`gpu` 或 `server` extra，并保证 MP4 交付所需的
`ffmpeg`/`ffprobe` 在 `PATH` 中。

## 内存不足 {#memory}

区分四个数字：PyTorch 已分配显存、PyTorch 预留显存、整卡显存占用和主机 RSS。完整流水线默认的 `block-ring`
把去噪器权重放在主机内存中，显存数字小并不代表容量足够。

- 把内存不足当作配置上限之前，先停止无关的 GPU 进程。
- 主机内存要留足：每个原生 worker 至少 64 GiB 可用，SM90/SM120 上的混合流水线约 128 GB，并发 worker 需更多。
- 从已实测的画布和时长出发，每次只扩大一个维度（见[选择硬件与配置](./profiles)）。
- 任何 CUDA 失败后都关闭流水线或会话；若无法确认清理完成，退出 worker 进程，不要复用其 CUDA 上下文。

## 双卡执行 {#parallel}

两张卡需要相同的受支持架构和同一个已准备的 artifact；完整流水线使用 `sequence-head`。任一卡失败都会中止请求，
挂起或部分清理会结束 worker；不要在同一进程内重试 collective。仅用于受控回归检查时，
`VFLASH_H3_DIRECT_RELAYOUT=0` 可恢复早期的显式拷贝（[详情](../reference/sol-engine-alignment)）。

## 输出与质量 {#quality}

能播放的 MP4 既不代表符合提示词，也不代表数值等价。先检查帧数、时长、画布和音轨，再按提示词和参考检查多个画面、
正常速度播放和实际声音。端点细节异常时，分别比较条件图、VAE 重建以及 `keyframe_delivery_profile="exact-v1"`
的还原结果。比较提速选项时，保持模型、适配器、调度、提示词、输入、种子、画布、时长和设备不变。

## 提交问题时附带 {#report}

用一次 `--profile-denoise` 做耗时归因。附上结果 JSON、Vflash 与依赖版本、配置 ID、显卡型号与数量以及简短的温度记录。
不要公开提示词、输入媒体、本地路径、设备 UUID 或模型文件。
