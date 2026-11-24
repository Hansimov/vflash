# 条件编码、显存与硬件配置

将固定配置流水线接入新设备或修改数值路径时，先保留相同分辨率、配置和种子的可播放对照。

## 混合模型与扩展请求 {#extended-hybrid}

`HybridModel(Path(".../transformer_ref"))` 保留 LightX v0.1 四次评估主干，只把第 25–49 块的 AdaLN 投影替换为固定版本的官方 Ref
组件（约 55 MiB 调制表，不加载第二个去噪器）。关键帧使用普通 FL 图，Ref 使用共享前缀、文本编码器和 VAE 的真实官方 Ref 图。
可在单张 SM89、SM90 或 SM120（SM103 为预览）上运行，配合稠密或显式 Veda 注意力；各显卡的实测范围见[选择硬件与配置](./profiles#heterogeneous)。

配合 Veda 时，单 SM89 混合模型接受 **5–15 秒**图像条件请求，以及最多**九张**有序参考图（`<Picture 1>` … `<Picture 9>`），
仍受[联合预算](./complete-pipeline#modes-and-limits)约束。标准 Ref4/Base16、SM86、双卡执行和参考视频不继承此扩展。

## 文本编码器前缀 {#exact-text-encoder-prefix}

条件编码保留 Qwen3-VL 64 层解码器中的 51 层：固定的 H3 适配器读取 `hidden_states[50]`，因此保留原始中间状态，并在安装卸载
钩子前移除未使用的尾部层。权重、视觉编码器和 BF16 计算不变；完整检查点仍会先加载，所以这不是更小的下载，也不保证冷启动更快。

## 注意力 {#veda-dense-edges}

单 SM89 官方 Base16 默认使用近似的 Sol；其他配置和双卡保持稠密。用 `attention_backend="torch-flash"` 显式选择稠密。
[Veda](./veda) 是显式稀疏后端，默认两端各五个稠密层（40 稀疏 / 10 稠密）；`veda_dense_edge_layers=3` 选择 44 稀疏 / 6 稠密。
每个会话持有自己的分区并报告实际运行的层。这些都是近似推理选项，不是等价声明；Sol 的[选择与限制](../reference/sol-engine-alignment#sol-default)另行说明。

## 条件复用与内存交接 {#memory-conditioning}

- **同组复用。** 可信调度器为同一受理请求的串行 I2VA/L2VA/FL2VA 候选传入同一个不透明 `ConditioningReuseScope`。
  一条 CPU 副本（≤128 MiB，空闲一小时过期）只保存文本输出、token 标记和干净的关键帧 latent；噪声、调度状态、激活和媒体从不复用。
  切勿从公开内容派生作用域，也不要跨所有者复用。
- **内存交接。** `conditioning_transport="memory"` 在单设备上把编码器张量一次性交给原生去噪，不写 bundle。默认的 `file`
  传输保留独立原生调用和服务所需的持久化契约。

## 近似 FP8 加载器 {#fp8-loaders}

`create_row_fp8_native_loaders` 与 `load_row_fp8_text_encoder` 通过 `H3Pipeline` 的可信 `native_denoiser_loader` /
`text_encoder_loader` 回调，加载显式准备、带 FP32 行缩放的 E4M3 权重。行分块限制临时内存，同时保持全序列注意力。W8A8 为近似计算，
需要 SM90/SM120，不会静默切换算法；默认仍是 BF16，准备凭据和命令行不接受 FP8 bundle。

## 计时与诊断 {#timing-and-diagnostics}

`VideoResult.elapsed_seconds` 覆盖一次成功的 `generate`，包括首次模型加载。`stages.initialization_seconds` 是本次调用内的加载时间，
`stages.request_elapsed_seconds` 不含加载，`stages.session_initialization_seconds`（亦即 `pipeline.initialization_seconds`）是会话的初始加载。
编码与媒体阶段报告恢复、挂起和调用时间；`capture_diagnostics` 区分官方调用、钩子等待、bundle 收尾和本进程资源增量。嵌套时间用于归因，不能相加。

## 正确性结果的含义 {#what-a-correctness-result-means}

条件与最终 latent 应与参考实现分开比较，画面质量另行评估，并对照原始请求判断动作、外观、指令和声音。H.264 与 AAC 为有损格式，
MP4 哈希不代表数值等价；官方音频 VAE 在重复请求间可能有细微差异。发布检查覆盖 14 个条件张量、最终 FP32 latent、解码媒体和可播放交付。

## 资格证据 {#qualification-evidence}

下表汇总当前接口实际运行过的范围；每项的数值、范围与限制见[版本更新](../reference/releases)和[性能记录](../reference/performance)。

| 能力 | 证据 |
| --- | --- |
| SM120 常驻像素帧预算（0.15） | RTX PRO 6000 96 GiB、FP8 常驻主干、Veda 44/6、人脸与六步音频修复：1920×1088 五秒 163 秒、2720×1536 五秒 261 秒、1920×1088 十秒 245 秒，显存峰值 42.1 GiB |
| 2720 像素原生高清（0.14） | RTX 4090 48 GB、BF16 混合 Veda 44/6 五条 120 帧 I2VA：2720×1536 两条、1536×2720、2368×1760、2720×1152 |
| 扩展混合模型（0.6.12） | 单 SM89 配合 Veda 的 15 秒与九图案例 |
| 六步音频（0.11–0.13） | RTX PRO 6000 FP8（1024²、1536×1344）与 RTX 4090 48 GB BF16 双槽（512²、1024²、1248×832/10 秒）；两组 512² 隐藏上下文复用 |
| 人脸重绘（0.6.34–0.10） | RTX PRO 6000 Server 完整视频场景，两步与四步；裁剪编码配置最终帧一致 |
| 内存交接 | 六条 H100 SXM 请求，解码画面与 PCM 完全一致 |
| 同组复用 | 双 SM86 A/B/A，码流一致 |

单一 GPU 型号或画布上的证据不能推广到其他情况；服务系统需各自完成资格验证。
