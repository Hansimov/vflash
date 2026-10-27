# 生成视频 {#generate-a-video}

`H3Pipeline`（Python）和 `vflash generate`（命令行）根据提示词，以及可选的图片或一段短参考视频，生成完整 MP4。
Vflash 负责原生去噪、阶段生命周期、本地媒体读取和 MP4 交付；文本/图像编码器使用固定版本的 Diffusers 与
Transformers 代码，视频与音频由官方 H3 VAE 解码。不需要 LightX2V 运行时或应用服务器。

本页描述当前版本。各版本的改动与对应证据见[版本更新](../reference/releases)，硬件容量见[配置与硬件](./profiles)。

## 模式与限制 {#modes-and-limits}

| 模式 | 输入 | 提示词标签 | 配置 |
| --- | --- | --- | --- |
| T2VA | 仅提示词 | — | `t2va-turbo4-exact-sm89` |
| Ref2VA | 1–3 张有序图片（扩展混合模型可到 9 张） | 按输入顺序 `<Picture 1>` … | Ref4（`ref2va-turbo4-exact-*`） |
| I2VA | 一张首帧锚点 | `<Picture 1>` | Base16 或 LightX v0.1 关键帧配置 |
| L2VA | 一张尾帧锚点 | `<Picture 1>` | 同一组关键帧配置 |
| FL2VA | 首尾两张锚点 | 按时间顺序 `<Picture 1>`、`<Picture 2>` | 同一组关键帧配置 |
| 参考视频 | 一段 2–5 秒视频 | `<Video 1>` | 单 SM89 Ref4；单 SM89／SM120 Veda 混合模型 |

参考图是视觉参考，不是时间轴位置。关键帧配置可用同一个已准备的流水线串行处理 I2VA、L2VA 和 FL2VA，无需重启；
`result.request_mode` 记录实际输入类型，`profile_id` 记录已准备的配置身份。其他配置会在执行前拒绝模式切换。
提示词按原文使用；请说明每张图的主体和作用。

**画布。** 宽高须为 32 的倍数，宽高比在 1:4 到 4:1 之间。普通请求最多 1,048,576 像素。图像条件原生高清最多
2048² 像素、最长边不超过 2720，例如 2720×1536（16:9）、1536×2720（9:16）、2368×1760（4:3）或 2048²。
参考视频输出为五秒：Veda 混合模型上限为 1,048,576 像素，原 Ref4 保持 475,136 像素。

**时长。** Turbo T2VA 与 Ref4 为五秒。Base16 关键帧配置接受 5–10 秒整数时长；[扩展混合模型](#extended-hybrid)
接受 5–15 秒。交付为 24 fps，例如 5 秒对应 124 个模型帧/120 个交付帧，15 秒对应 362/360。

**联合预算。** 画布与时长不能独立组合：`宽×高×模型帧数` 不得超过 `2048²×124`，因此 4MP 画布只能是五秒请求，
2048² 十五秒会被拒绝。三张以上的多图 Ref 限五秒和 960×544。这些是软件上限，服务系统只能对外提供在实际硬件上
通过资格验证的组合。

### 分配硬件前检查请求 {#check-a-request-before-allocating-hardware}

`validate_request` 执行与 `H3Pipeline.generate` 相同的准入检查，但不导入 Torch、不加载权重、不打开媒体，也不初始化 CUDA：

```python
from pathlib import Path
from vflash.pipeline import VideoRequest, validate_request

request = VideoRequest("Describe the motion.", first_frame=Path("first.png"),
                       width=2720, height=1536, duration_seconds=5)
validate_request(request, profile_id="i2va-turbo4-v01-544-exact-sm89",
                 hybrid=True, attention_backend="veda-triton",
                 parallel_strategy="single", weight_residency="block-ring")
```

通过准入只说明软件上可执行，不代表显存、资源完整性、硬件资格或输出质量。

## 安装与准备资源 {#installation-and-assets}

从发布版本的检出目录安装 pipeline 扩展，并确保 `PATH` 中有 `ffmpeg` 和 `ffprobe`：

```bash
python -m pip install '.[pipeline]'
python -m vflash.install_sol   # 仅单张 SM89 Base16 默认的 Sol 需要；不下载模型
```

扩展会固定适配器实现，包括 Diffusers 提交 `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`。先选择
[固定模型配置](../reference/pipeline-profiles)，再用六个显式路径描述资源：

| 字段 | 内容 |
| --- | --- |
| `model_directory` | `MiniMaxAI/MiniMax-H3` 在 `42ed227ee7df40d41602854ae760620d6eb651fe` 的官方 Diffusers 组件，含 `transformer_ref`（Ref）或 `transformer`（T2VA、关键帧） |
| `adapter_path` | Turbo 配置固定的 BF16 适配器（见[运行资源](../reference/runtime-assets)），Base16 为 `null` |
| `decoder_directory` | 同一版本的 `FL2VA` 目录，含 `video_vae` 和 `audio_vae` |
| `artifact` | 该配置完整的 BF16 原生产物 |
| `schedule_overlay` | 匹配的训练 Euler 调度（Turbo 4 次评估；Base16 16 次，偏移 12/3） |
| `auxiliary_tensor` | 与之匹配的原生输入/输出张量 |

后三项可用[官方权重编译](./compile-weights)生成，或对照[资源契约](../reference/runtime-assets)检查已有产物；目前不提供现成模型包。
把资源放到最终的只读位置后执行一次准备：

```bash
vflash prepare-pipeline --assets pipeline-assets.json --receipt prepared-assets.json
```

准备过程会按上游清单或产物 manifest 校验所有被使用的文件，并生成绑定本机文件系统的凭据；之后启动只检查文件身份和时间戳，
不再重复计算哈希。移动或修改资源后需要重新准备。

## 命令行生成 {#generate-from-the-command-line}

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

- Ref2VA：一到三个 `--reference`，顺序与提示词一致。
- T2VA：准备 `t2va-turbo4-exact-sm89`，不传图片参数。
- 关键帧：`--first-frame`（I2VA）、`--last-frame`（L2VA）或两者（FL2VA）；用 `--duration N` 指定时长。不要与 `--reference` 混用。
- 双卡：`--gpu 0 --peer-gpu 1 --strategy sequence-head`（SM86 双卡必需；两张相同 RTX 4090 48 GB 的 Base16 关键帧可选）。
- 显式 Veda 边缘层数：`--veda-dense-edge-layers 3`（见[注意力](#veda-dense-edges)）。

进度以 JSON 行写到 stderr，结果写到 stdout。每条命令都会自行加载并关闭模型；要在多次请求间常驻模型，请使用同一个
Python 流水线。容器用法见 [Docker 生成](./docker#pipeline)。

## Python 生成 {#generate-from-python}

在其他代码初始化 CUDA 之前，于独立进程中运行流水线，并显式选择设备：

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = discover_nvidia_devices()
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))
with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    pipeline.prepare()                       # 可选：接收请求前预加载
    result = pipeline.generate(
        VideoRequest(prompt=Path("prompt.txt").read_text(encoding="utf-8"),
                     references=(Path("subject.png"), Path("setting.png")), seed=1234),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

关键帧请求使用 `first_frame=`、`last_frame=` 或两者。双卡时传入 `peer_device=devices[1], strategy="sequence-head"`，
编码与解码仍在第一张卡上。两张 RTX 4090 处理一个请求能降低延迟，但消耗更多 GPU 秒；同时有两个请求时，应使用两个独立的单卡 worker。

**生命周期。** 构造时在 CPU 上检查配置；首次 `generate` 会先校验输入再加载模型。无效输入不会破坏已加载的实例；执行失败会
退役该流水线并清理临时文件。输出路径不得已存在，只有编码、探测和阶段清理全部成功后才发布视频。`close()` 在 CUDA 完成栅栏后
释放自有模型，绝不重置其他所有者的上下文。原生分块环为编码器和 VAE 留出显存，CPU 上的模型主副本常驻，因此主机内存需要容纳它们。
进度回调同步执行，可抛出异常取消，但不能在回调中调用 `close`。`trust_local_code=True` 会从已校验的本地快照加载官方解码器文件，
请先阅读[许可证](../reference/license)。

在只读容器中，请为 Triton 提供可写且可执行的缓存目录；`noexec` 临时文件系统不能作为该缓存。

管道媒体交付采用120秒停滞时限与600秒编码总时限；RGB输入块写入完成或编码文件增长会刷新停滞计时。
到期后回收自有编码进程并清理临时媒体。直接调用`encode_mp4`时可设置`stall_timeout_seconds`和
`timeout_seconds`，文件输入采用总时限。

## 交付选项 {#delivery-options}

| 选项 | 作用 |
| --- | --- |
| `keyframe_delivery_profile="exact-v1"` | 解码后恢复每个输入端点（LANCZOS 缩放到画布，四帧羽化）；默认保留 VAE 重建 |
| `audio_delivery_profile="web-v1"` | 有界增益，目标 −18 LUFS、−2 dBTP 限幅；默认 `unchanged` 保留解码波形 |
| `audio_delivery_profile="silent-v1"` | 原时长的数字静音立体声轨，只跳过波形解码；仅用于明确要求整段静音 |
| `generate_audio=False` | 仅视频生成：去噪前去掉音频行，不写音轨；同一 seed 下画面可能不同 |
| `generate_rgb(request, Path("x.npy"))` | 仅视频的 uint8 RGB 数组 `[模型帧, 高, 宽, 3]`；`required_frames=` 只解码所需帧对应的 VAE 块 |

命令行支持 `--keyframe-delivery-profile` 和 `--audio-delivery-profile`。`generate_audio=False` 与 RGB 交付仅限 Python 单卡路径。
完整流水线还会在官方音频解码器截断前施加联动立体声余量；可复用的上下文为
`vflash.media.audio_headroom.audio_decoder_headroom`，它是采样峰值保护，不会放大安静音频。以上选项都不能改善音色或同步。

## 使用参考视频 {#reference-video}

```bash
vflash generate --prepared-assets prepared-assets.json --prompt-file video-prompt.txt \
  --reference-video reference.mp4 --gpu 0 --seed 1234 --output variation.mp4 --trust-local-code
```

Python 中使用 `VideoRequest(prompt=..., reference_video=Path("reference.mp4"))`，以 **`<Video 1>`**
指代视频。Veda 混合模型在单 SM89／SM120 上复用共享官方 Ref 图；同一实例交替处理图像和视频。
输入为一段完整的 2–5 秒 MP4、MOV 或 WebM，最多 20 MiB、1344×768 源像素，一路方形像素视频流，
帧率上限 240 fps。源音频丢弃。官方编码规范化为 24 fps，保持固定时间对齐和前缀预算；
120 帧输入按官方网格取前 107 帧并编码为 32 个时间位置。输出是参考引导的五秒新视频。

每个条件编码器拥有最多 128 MiB 的 CPU 参考 latent 缓存，相同解码身份和几何重复使用时复用官方 VAE
结果；文本与随机噪声每次重新计算。缓存随 pipeline 关闭释放，调用者负责素材授权和隔离。
默认 `reference_video_profile="full"` 保留完整空间信息；显式 `"spatial-half"` 在编码后平均池化空间网格，
保留全部已编码时间位置。该近似选项只用于 Veda 混合路径，可能改变人脸、细节与构图。
身份一致性优先使用 full。具体张量协议见[条件契约](../reference/runtime-assets#video-conditioning)。

参考 latent 复用和压缩方案受 MIT 许可的
[RefMod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/517e8c419a8584fbf69fceb487bc39d83e6a6ec8) 启发。
本实现复用官方编码结果并提供显式空间池化；RefMod 的逐素材 MSE 拟合和音色参考属于另外的方案。

### 外观图与动作视频组合（0.21）

0.21混合路径将一份参考视频与一至三张有序图像组合：

```python
request = VideoRequest(
    prompt="<Picture 1> 提供外观；<Video 1> 提供动作和摄影机时序。",
    references=(Path("appearance.png"),),
    reference_video=Path("motion.mp4"),
    width=960, height=544, duration_seconds=5,
)
```

图像职责使用`<Picture 1>`至`<Picture 3>`，视频使用`<Video 1>`。图像沿目标面积、仅缩小的match策略，
视频沿官方尺寸规范；执行路径为单SM89／SM120 Veda混合模型，默认完整空间参考，输出保持五秒、最多1,048,576像素。
图像顺序、身份与目标画布参与参考缓存身份，同一导引搭配新外观时重新编码组合条件。

首条执行证据覆盖48 GiB SM89、一张外观图与五秒导引：960×544完整调用280.7秒，编码60.2秒、去噪201.3秒；
会话初始化另计90.9秒。成片保持人物外观与完整交接动作。同一简单动作的简洁／细化I2VA提示也完成交接，
调用76.1／70.5秒，编码和冷暖状态分别记录。按具体控制需求选择动作参考；更广场景、三图容量与目标硬件性能继续实测。
对于粗糙动作导引，可显式选择`reference_video_profile="spatial-half"`，仅池化最后一份视频latent的空间网格，
此前所有外观图latent及视频时间位置完整保留。相同初始场景的条件行数从32,766降至8,574，
SM89调用168.9秒（编码59.1、去噪90.1），仍完成交接。SM120完整／压缩分别在两个缓存宿主完成于110.3／79.4秒，
包含不同的冷编码成本；去噪为74.0／31.8秒。这里记录单场景的近似控制结果，更广的动作和身份覆盖继续实测。
0.21增加组合输入；此前版本保留图像／视频分别输入的合同。

## 混合模型与扩展请求 {#extended-hybrid}

`HybridModel(Path(".../transformer_ref"))` 保留 LightX v0.1 四次评估主干，只把第 25–49 块的 AdaLN 投影替换为固定版本的官方 Ref
组件（约 55 MiB 调制表，不加载第二个去噪器）。关键帧使用普通 FL 图，Ref 使用共享前缀、文本编码器和 VAE 的真实官方 Ref 图。
可在单张 SM89、SM90 或 SM120（SM103 为预览）上运行，配合稠密或显式 Veda 注意力；各显卡的实测范围见[选择硬件与配置](./profiles#heterogeneous)。

配合 Veda 时，单 SM89 混合模型接受 **5–15 秒**图像条件请求，以及最多**九张**有序参考图（`<Picture 1>` … `<Picture 9>`），
仍受[联合预算](#modes-and-limits)约束。标准 Ref4/Base16、SM86、双卡执行和参考视频不继承此扩展。

## 文本编码器前缀 {#exact-text-encoder-prefix}

条件编码保留 Qwen3-VL 64 层解码器中的 51 层：固定的 H3 适配器读取 `hidden_states[50]`，因此保留原始中间状态，并在安装卸载
钩子前移除未使用的尾部层。权重、视觉编码器和 BF16 计算不变；完整检查点仍会先加载，所以这不是更小的下载，也不保证冷启动更快。

## 注意力 {#veda-dense-edges}

单 SM89 官方 Base16 默认使用近似的 Sol；其他配置和双卡保持稠密。用 `attention_backend="torch-flash"` 显式选择稠密。
[Veda](./veda) 是显式稀疏后端，默认两端各五个稠密层（40 稀疏 / 10 稠密）；`veda_dense_edge_layers=3` 选择 44 稀疏 / 6 稠密。
每个会话持有自己的分区并报告实际运行的层。这些都是近似推理选项，不是等价声明；Sol 的[选择与限制](../reference/sol-engine-alignment#sol-default)另行说明。

## 可选音频补采 {#optional-audio-only-six-step-refinement-source-main}

`AudioRefinementRequest` 在四步混合 I2VA 生成后追加六次原模型音频评估。它冻结已完成的视频，保留全部时间位置，并以原始旋转坐标
均匀采样最多 500,000 像素的空间上下文。视频像素、时长和交付时钟不变。

```python
from vflash.native.h3_audio_refinement import AudioRefinementRequest
from vflash.native.h3_refinement_schedule import refinement_schedule

refinement = AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.5), tuple(audio_tables))
result = pipeline.generate(request, output_path, audio_refinement=refinement)
```

要求混合权重、单设备、Veda 44/6、一张同画布首帧和生成音频，并使用常驻权重加 `veda-triton`，或原生双槽 BF16 `block-ring`
加 `veda-sm89`。静音交付、仅视频输出、其他模式以及同时重绘会被拒绝。默认 `context_pixels=500_000` 以均匀抽样到 0.5MP 的视频上下文作为条件；`context_pixels=250_000` 选择 0.25MP 预算（0.15.1 起），`context_pixels=None` 选择全上下文六步对照。进度报告
4 + 6 次评估，`audio_refinement` 记录额外工作；失败时恢复绑定、注意力、视频和原音频。没有命令行或 HTTP 开关，也不默认启用。

### 原片音轨重建（0.19）

高级原生会话可传入 `encoded_audio_source=EncodedAudioSource(video_rows, audio_rows)`，并显式指定
`AudioRefinementRequest`。输入是有限值、已归一化的模型坐标：视频行排除首帧条件前缀，双声道音频按声道连续排列。
会话复制输入并跳过原视频采样；最终封装复用原视频码流，仅解码新音频。

默认六步方案支持这类原片条件；应用负责源文件所有权、波形编码、版本记录和原画面封装。

### 更长音频采样（0.19）

`AudioRefinementRequest` 支持 `evaluations=12, denoise=0.9` 与 `evaluations=20, denoise=0.95`。
每种方案需要同一步数和强度编译的参数表，例如
`compile_refinement_tables(..., steps=20, denoise=0.95, mode="audio")`。
可在普通四步I2VA生成后执行，也可使用显式原片条件。空间条件预算、隐藏状态／K/V缓存及音画解码重叠
分别配置；按声音自然度、内容对应、瞬态表现和完整MP4交付耗时选择方案。

`dense_attention=True` 为二十步的另一项诊断：音频阶段全部上下文层使用密集注意力，成功或失败后恢复原策略。
主视频仍采用原四步策略；启用缓存时，非音频隐藏状态仍在首轮之后固定。普通请求默认值保持原配置。

### 紧凑参数表（0.19）

`vflash.compiler.refinement_compact.compact_audio_tables(source, output)` 写出新的
`audio-table-dedup-v1` 目录。它对完全相同的整行去重，保留原表全部BF16位，包含尚未使用的行。
`load_tables` 恢复完整连续布局，同时支持原格式；源文件、推理内存和计算保持。
二十步存储由611.00 MB降至400.92 MB。51个真实文件已完成全部表元素的逐位对比；
六／十二／二十步小型检查另覆盖符号零和指数位。

### 隐藏上下文复用 {#audio-hidden-context}

`reuse_hidden_context=True` 复用首次评估的 BF16 非音频隐藏状态；六次音频评估都保留，后续步骤重建 K/V，只处理音频行。
这是显式近似，绝不改变视频 latent，默认关闭。缓存只属于一次补采（最多 40 GiB，并至少保留 12 GiB 空闲显存），成功或失败都会释放。

### K/V 复用与音画重叠 {#audio-kv-overlap}

在常驻 SM120 上，`reuse_kv_context=True` 可复用首次评估的旋转后 BF16 非音频 K/V，后五步重新计算
音频行。它与 `reuse_hidden_context` 二选一，属于显式近似；所选求解方案与视频 latent 保持，缓存拥有相同的
40 GiB 上限及 12 GiB 余量。SM89 的已测方案采用隐藏状态缓存。

`H3Pipeline(..., audio_decode_overlap=True)` 让最终视频 VAE 解码与所选音频阶段在独立 CUDA stream 上重叠，
适用五秒、至多 1MP 的请求；显存余量充足时才提交，余量不足自动串行。pipeline 拥有线程与 future，
成功或异常都会汇合并释放，`stages.audio_video_overlap` 记录实际启用情况。音频补采依赖已完成的视频，
主视频去噪先执行；重叠发生在最终画面解码阶段。主去噪保留联合音频，以保持原动作和口型条件。
完整耗时、设备精度和逐帧检查范围见 [0.18 版本记录](../reference/releases#_0-18-0)。

### BF16双槽权重复用（0.20） {#audio-ring-weight-cache}

`AudioRefinementRequest(..., weight_cache_bytes=20 * 1024**3)`在音频阶段将部分BF16权重保留在显存中，默认值为零。
权重数值、原计算槽地址、注意力及采样策略保持；后续步骤通过显存内复制读取已缓存权重，减少PCIe传输。
作用域在成功或异常退出时恢复原绑定，并为上下文和重叠VAE工作预留20 GiB。常驻主干沿自身已有权重执行。
`weight_cache`诊断记录实际缓存层数、字节数和准备耗时。

初始RTX 4090 48 GiB资格覆盖十二步、五秒I2VA、隐藏上下文复用与解码重叠，画布为960×544和1024²。
四组完整匹配对照节省8.93–11.06秒（5.6%–13.6%），全部交付解码视频帧和双声道PCM样本与各自对照一致。
这是原音频方案内的权重放置等价性证据；其他时长与内存布局沿其独立容量验证。

## 人脸重绘构件 {#explicit-native-face-repaint}

以下都是需显式启用的引擎接口。检测、跟踪、裁剪/关键帧对齐、修复模型、源音频保留和合成由应用负责。

- `VideoRepaint` / `VideoRepaintRequest`：在实际 sigma 上使用编译后的 FL/Ref 调制做低噪声重绘，保留条件行并冻结音频。
  用 `compile_refinement_tables(..., steps=2 或 4, denoise=0.035, mode="face")` 编译调制表。原生请求在预加载的
  `NativeEngineSession` 上运行，需 `generate_audio=False`，仅限单设备。
- `vflash.media.face_encoder.face_encoder(vae, profile)`：方形 320/384/512 裁剪、120–362 模型帧的作用域编码。
  `batch2-fp16-norm` 与 `batch2-fp16-norm-reuse` 为近似 FP16 批处理（后者复用完全相同的片段）；`batch2-fp16-norm-reuse-untiled`
  对 512 方形、124 帧裁剪去除空间分块，显存更高。`original` 保持原编码。
- `H3Pipeline.source_scene_conditioning(source)`（0.19）：为相关的仅视频重绘复用源场景的 Qwen 特征，会刻意改变条件。

流程参考了 MIT 许可的 [ComfyUI H3 FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)。不包含人脸修复或身份识别权重。

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

### 保留原生音视频潜变量

Python 调用方可向 `H3Pipeline.generate` 传入 `latent_output=Path("clip.safetensors")`。
目标必须与 MP4 不同且尚不存在。完整的原生音视频潜变量在媒体解码前原子保存；后续解码或编码失败时仍保留。
成功返回的 `VideoResult` 包含该路径。省略此参数时维持原有临时文件生命周期。
保留文件本身不等于自动重试解码，也不代表某种续接方案已取得质量资格。

### 实验性原生音画目标前缀

`H3Pipeline.generate(..., native_av_prefix=prefix, latent_output=path)` 支持高级接口
`vflash.native.h3_av_continuation.NativeAVPrefix`。使用 `from_latents` 构建，传入原生
`video_latents`／`audio_latents`、选定的 `source_frames`、`context_frames`，以及从同一组模型权重
编译的51个 timestep 1.0 调制张量（50个主块和最终层）。这些资产由调用方显式持有，
引擎不会自行下载或换用检查点。当前仅为Python接口，CLI／HTTP准入保持原范围。

当前执行范围是单SM89、Veda block-ring、四步I2VA原生音频及一个同画布首帧；本组合拒绝额外
音频增强、重绘及性能插桩。源切点和上下文均须满足24fps下 `17n+5`；上下文还须能被3整除，
例如39或90帧，以匹配40Hz音频的精确时长。源切点支持107、124等全部VAE对齐边界；
音频取最近tick，与视频切点最多相差1/120秒。首图对应上下文起点，不是其末帧。上下文占生成目标开头；
应在潜变量层去除重复区间，再对拼接后的序列统一解码。返回的五秒MP4包含上下文，不应整段追加。

`prefix.join(source_latents, generated_latents, generated_frames=124)` 核对实际前缀逐位保留，
去除重叠并返回拼接潜变量。音频长度每次按完整序列时钟重算，最多裁掉一个多余尾tick；音频不足则报错，
避免连续续接累积时钟误差。调用方随资产保留新的帧数 `source_frames + generated_frames - context_frames`。

每次Euler更新后恢复视频0.999条件强度与音频1.0干净上下文，最后恢复逐位相同的干净音画前缀。
调制表和行时刻表由当前请求持有，并在退出时恢复。960×544、五秒的实际SM89完整执行中，视频及
音频潜变量均与已实测的临时原型逐位一致；这证明接口集成保持结果，人物运动及主观音质仍逐片评价。

主动换景时，可用 `preserve_video=False` 构建前缀。每步仍恢复原生音频历史，所有目标画面行
则依据新场景首帧生成；当前同画布和执行范围约束继续适用。将目标开头 `context_frames` 帧
作为预卷去除，用 `prefix.join_audio(source_latents, generated_latents, generated_frames=124)`
取得拼接音频张量并统一解码。画面按同一帧时钟独立剪接；`join` 拒绝此模式，以免将主动换景
误作视频连续性。回执标明保留视频行数为0，`video_prefix_exact=None`。两个实际SM89换景
试验均在39帧预卷下精确保留65个历史音频tick，验证了执行与时钟；配乐连贯、对白及同步仍须试听。

若实际交付在模型补齐尾部之前结束，可显式传入 `join_audio(..., delivery_frames=N)`。
两个独立时长的舍入可能让未采用的拼接尾部少一个音频tick；该参数核对整个实际交付区间都有声音，
并保留已有原生尾部供解码，调用方随后按声明时钟裁切PCM，不补造缺失声音。
省略参数仍要求完整模型拼接时钟，包括模型补齐尾部。

可用 `vflash.compiler.continuation.compile_native_history_tables` 准备这些表。
传入本地基础transformer检查点；混合模型另传本地参考检查点和实际 `reference_blocks` 元组，
例如 `tuple(range(25, 50))`。函数仅读取对应张量，在CPU编译51个干净上下文表，返回的tuple
可直接用于 `NativeAVPrefix.from_latents`，并按模型选择复用。它不修改权重或全局线程数，
可选 `progress(completed, total)` 回调可执行调用方期限。原生前缀接口目前是实验性源码接口，
已发布的0.21.0 wheel尚未包含这些接口。
