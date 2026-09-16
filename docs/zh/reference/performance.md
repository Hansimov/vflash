# 性能测量

测量应覆盖应用真正关心的过程，并明确计时范围。完整链路输出 MP4，原生去噪器输出 latent 张量，两者的计时边界不同。

## 区分首次加载与后续请求 {#timing}

每次命令行调用都会初始化模型。常驻 `H3Pipeline` 或原生 HTTP worker 可以跨请求复用模型，因此启动、首次使用和后续请求的成本不同。

完整生成应单列 `H3Pipeline.initialization_seconds` 和 `VideoResult.elapsed_seconds`。后者覆盖输入准备、编码、去噪、媒体输出及请求清理。`stages` 同时包含外层时长和嵌套明细，不应重复相加，详见[链路计时合同](../guide/complete-pipeline)。

原生去噪结果的 `session` 包含以下字段：

| 字段 | 含义 |
| --- | --- |
| `request_index` | 当前模型会话中的请求序号，从 1 开始 |
| `initialization_seconds` | 该会话的初始化耗时 |
| `initialization_charged_seconds` | 首次请求计入初始化耗时，后续请求为零 |
| `request_wall_seconds` | 在已初始化会话中执行本次请求的耗时，包含写入 latent 输出文件 |

估算引擎侧首次请求成本时，可将 `initialization_charged_seconds` 与 `request_wall_seconds` 相加。排队、进程启动、HTTP 传输、输入编码和视频解码不在这两个字段的合计范围内；如果应用包含这些环节，应分别测量。

首个任务之前，`/readyz` 通过只表示配置文件和显卡检查通过，不代表模型已经加载。

## 剖析单次去噪请求 {#denoise-profile}

`vflash generate` 与 `vflash denoise` 均支持 `--profile-denoise`。该开关刻意保持为显式诊断选项：它会为每个 evaluation 和每个流式 block 记录 CUDA timing event，因此最终吞吐比较应使用未开启诊断的独立运行。诊断本身不增加 evaluation 围栏；完整 pipeline 原本就会在发布每次进度前等待所有协作设备，报告会把这些既有围栏单列。

返回的 `generation.denoise_profile` 会区分：

- 主卡的行时间步准备、输入打包、invocation 准备、denoiser、final layer 与 latent 更新；
- 每个 rank 的 H2D 活跃时间、计算流等待权重时间、block 执行以及复制/计算跨度；
- 每个 evaluation 与 rank 汇总的 AdaLN、attention norm/modulation、QKV projection、Q/K norm 与
  rotary、attention、attention output、FFN norm/modulation、FFN input 和 FFN output；
- sequence/head collective 的调用次数、传输字节、主机提交时间和主机 `Work.wait()` 时间；
- sequence/head attention 在计算流上的 QKV pack、入站依赖等待、QKV unpack、Flash-SDPA、
  attention 结果 pack、出站依赖等待和最终 unpack；
- 每个 evaluation 的引擎提交、既有进度围栏、回调、最终围栏和 profile 汇总耗时。

H2D、ready wait、block compute 与 rank span 描述的是相互重叠的 CUDA stream，**不能相加**。block compute 包含 collective 临界路径；主机 collective wait 只是提交/同步诊断，不能单独当作 GPU 通信耗时。需要按 kernel 归因 NCCL 时仍应使用外部 CUDA profiler。诊断输出只含设备序号、时间和字节计数，不包含模型张量、提示词、路径或 GPU UUID。

sequence/head 的 `inbound_ready_wait` 与 `outbound_ready_wait` 只测量实际阻塞计算流的依赖，
不代表 NCCL kernel 的完整生命周期；后者可能在另一条 stream 上与 QKV pack 或 Flash-SDPA 重叠。
这些 attention 子阶段已经包含在外层 `attention` 中，不能再与 attention 或 block 执行重复相加。

## 让比较有意义 {#comparisons}

使用相同的条件包、模型与 LoRA 版本、运行配置、显卡和输出边界。分别报告首次使用与后续请求耗时，并重复足够次数以观察波动。

比较内存时，区分 GPU 已分配内存、GPU 预留内存、整张显卡占用和系统 RAM。尤其是 3080 的分块加载权重会占用系统内存，这部分不会出现在只统计显存的图表中。

从 Turbo8 切换到 Turbo4 时，变化的不仅是计算量，也包括蒸馏调度方式。报告速度比较时，应同时说明这个差别。

## 双卡执行 {#parallel}

双卡协作可以缩短单个请求的等待时间，独立单卡 worker 则服务于不同的吞吐需求。使用相同主卡、输入和计时边界评估两者，别把双卡的一次请求与两张卡的总吞吐混为一谈。

已公开的 [a3 双 3080 对照](./benchmarks#sm86-parallel)在一个固定负载上测得 1.725× 加速。它保留原始软件版本、热状态和质量限制，不代表当前版本或其他负载的速度保证。

## 检查输出质量 {#quality}

更快的结果仍需满足实际任务。请根据用户原始要求与参考素材，检查指令遵循、主体与细节一致性、运动、画面瑕疵和音画关系。即使候选结果接近基线，基线本身也可能没有满足要求。

解码后按时间查看多帧，并以正常速度观看视频、试听音频。静帧无法证明运动自然，音频波形也不能证明语义正确。分别记录已确认的发现、无法判断的维度、样本不足和评审分歧。张量误差仅用于排查实现差异，不能用作生成质量分数。

精确注意力不代表蒸馏 LoRA 与基础模型效果等价，也不保证不同 GPU 架构上的浮点结果完全一致。每个配置当前完成的检查范围见[支持说明](../guide/profiles#lora)。

完整链路已支持[列出的配置](./pipeline-profiles)。集成检查证明已测请求能够正确完成，不代表通用的提示词到 MP4 速度或质量保证。
