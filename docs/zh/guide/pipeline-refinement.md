# 原生音轨细化与人脸重绘

画面达标后可单独选择音频补采。人脸重绘另有遮罩、权重与执行限制，保存成片时应记录实际启用的选项。

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
