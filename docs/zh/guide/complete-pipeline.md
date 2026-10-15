# 生成视频

## 在申请硬件之前检查请求

源码主线从 `vflash.pipeline` 导出 `validate_request`，与
`H3Pipeline.generate` 共用软件准入检查，不导入 Torch、不加载权重、
不打开参考素材，也不初始化 CUDA：

```python
from pathlib import Path
from vflash.pipeline import VideoRequest, validate_request

request = VideoRequest("描述画面中的运动。", first_frame=Path("first.png"),
                       width=2048, height=2048, duration_seconds=5)
validate_request(request, profile_id="i2va-turbo4-v01-544-exact-sm90",
                 hybrid=True, attention_backend="veda-triton",
                 parallel_strategy="single", weight_residency="block-ring")
```

检查应使用实际执行的引擎版本与解析后的选项。尺寸和时长分别合法，
不代表组合合法：2048 × 2048、十五秒仍超出现有像素与帧数联合预算。
准入通过只表示软件允许，不代表显存足够、素材凭据有效、目标硬件已验证或画质通过。


## 0.6.8：显式 hybrid 多参考模型

`H3Pipeline(..., hybrid_model=HybridModel(Path("models/MiniMax-H3/transformer_ref")))`
在原始 FL LightX v0.1 四步配置上，仅替换第25–49块的官方 Ref AdaLN 投影。
FL 的时间 MLP、输出层及适配器不变，新增约55.4 MiB调制表，不加载第二个去噪主干。
这是一种明确标识的模型组合，不是独立官方 Ref4，也不改变未指定该选项时的行为。

从 `vflash.pipeline` 导入 `HybridModel`。当前仅支持单 SM89 与
`attention_backend="torch-flash"`或显式`veda-sm89`，使用已准备的原始 v0.1 I2/FL 工件。
同一实例保留普通首/尾帧条件图，并为1–3张有序参考图片使用真实官方 Ref 条件图，
共享条件前缀、文本编码器和 VAE，以内存包交给原生引擎。Ref仍为五秒，
不接受视频参考或未经验证的hybrid Sol组合。
结果以 `request_mode` 和 `stages.model_variant` 区分实际合同。
官方权重通过清单/大小/头部与每请求文件标识检查，不反复哈希大文件；仓库不分发合并权重。

原生原型已完成I2、FL及三参考完整视频。公共Python接口又在同一RTX 4090 48 GB实例中
连续执行Ref→I2→Ref：960×544、五秒、四次前向、BF16/dense/block-ring，
每请求86.542/72.074/76.631秒，单次初始化104.094秒另列；均交付120帧/24fps并完整AV解码。
两次Ref使用同一来源族的不同seed，是组件寿命检查，不是三个独立场景或配对速度比较。
抽样中角色和场景保持，但起始停留、接触细节仍有瑕疵；声音语义与更广画质资格未外推。
不能据此宣称网站已部署或人物、接触及动态瑕疵已修复。
完整代码示例见[英文接口说明](../../guide/complete-pipeline#hybrid-reference-model-in-0-6-8-opt-in)。

Vflash 0.5.1 支持纯文字生成，也支持提示词加一至三张有序参考图，输出五秒 MP4。0.4.0 支持官方 Base16 I2VA、L2VA 和 FL2VA 请求，分别输入一张明确的首帧、一张明确的尾帧，或同时输入首尾帧。连续请求使用 Python 接口，单次生成也可以使用容器命令行。完整链路支持单张 RTX 4090 48 GB 上的 T2VA Base4 和 Ref2VA Turbo4；Ref4 也支持双张 RTX 3080 20 GB。Base16 关键帧配置支持单张 RTX 4090 48 GB、可选的双张匹配 RTX 4090 48 GB 协作，或协作的双张 RTX 3080 20 GB。

单张 4090 的 Ref4 还支持[短视频参考](#reference-video)，同一实例可以交替处理图片和视频，无需切换权重。

## 各阶段的职责

Vflash 负责原生去噪、模型生命周期、本地图像读取和 MP4 输出。文本和图像编码采用固定版本的 Diffusers 与 Transformers；Turbo 配置还使用 PEFT 适配器。音视频解码采用 H3 官方 VAE。它们是明确列出的依赖，不会被描述成新实现的原生内核。运行时不依赖 LightX2V 或业务服务。

Turbo 请求使用四次去噪计算，并保留五秒合同；Base16 关键帧请求使用 16 次，接受五至十秒的整数时长并采用原生 24 fps 时钟。对应的模型帧/交付帧为：5 秒 124/120、6 秒 158/144、7 秒 175/168、8 秒 192/192、9 秒 226/216、10 秒 243/240。宽高必须是 32 的整数倍，画布总像素不超过 1,048,576，宽高比在 1:4 至 4:1 之间。视频参考请求仍保留独立的 475,136 像素输出上限。这些是 API 天花板；服务端仍只能声明已在实际硬件上通过资格验证的时长与画布组合。提示词原样传入。Ref2VA 的一至三张图片按照传入顺序编号为 `<Picture 1>`、`<Picture 2>`、`<Picture 3>`，请在提示词中说明每张图的主体和用途。这些是视觉参考，不代表视频中的帧位置，也不是严格的关键帧约束。I2VA 输入第零帧锚点，L2VA 只输入尾帧锚点；这两种单图请求都可将图片称为 `<Picture 1>`。FL2VA 同时输入首尾锚点，提示词按时间顺序将其称为 `<Picture 1>` 和 `<Picture 2>`。I2VA/FL2VA 的五秒和十秒路径已有有界硬件实证。一个十秒、736 × 992 的 SM89 L2VA 案例现已有完成性、媒体完整性、端点和延迟对照的有界证据；更广泛的 L2VA 质量、其他画布与中间时长仍需在目标硬件上资格化。

准备资产前先选择[固定模型配置](../reference/pipeline-profiles)。Ref2VA 使用 `transformer_ref` 与 Ref4 v0.1；T2VA 使用 `transformer` 与 Base4 v1.0；Base16 关键帧请求使用官方 `transformer`，无 LoRA 执行 16 次计算。同一硬件上的成对 Base16 I2VA/FL2VA 配置具有完全相同的模型工件和调度，因此任一已准备的关键帧 pipeline 都能串行处理 I2VA、L2VA 和 FL2VA，无需重启 profile 或执行冷初始化。条件信息仍按实际请求模式生成；结果以 `profile_id` 保留已准备身份，并以 `request_mode` 记录实际输入类型。每次请求的阶段驻留仍遵循所选内存策略。其他配置会在执行前拒绝模式切换。

### 0.5.0的完整二进制百万像素预算

0.5.0（包含`53d6687`）将Base16 pipeline及原生条件合同上限统一为**1,048,576像素**：
1024×1024方形不再缩成992×992。现有0.4.0标签仍保留1,032,192像素上限。宽高仍为32的倍数；
总面积不要求某条边必须为1024，也不扩大独立的视频参考合同。

一对协作的RTX 3080 20GB完成了一次五秒1024×1024 I2VA：交付120帧，完整解码通过。
同一latent的正常和`silent-v1`交付具有完全相同的压缩视频流，静音轨解码采样全零。
这只是有界容量和媒体合同筛选，不是语义质量或端到端加速结论。其他时长/硬件组合仍须保留各自的
服务资格边界，不能用API校验通过代替实卡资格。

## 精确的条件编码前缀

0.5.1将Qwen3-VL语言层从64层保留为51层。固定H3适配器只使用`hidden_states[50]`；
多保留一层是为了取得原始中间态，而非最终归一化后的状态。前50层、视觉编码器、权重及BF16
运算均不变，在安装卸载钩子之前移除未使用尾层。此优化独立于Sol，不改变去噪计算。

仍先加载原官方检查点，因此不减少下载体积，也不承诺冷启动倍率。
[目标硬件条件编码对照](../reference/benchmarks#encoder-prefix)将阶段收益与整片耗时分开；
无需新增选项或下载另一套权重。

## 安装和模型资产

0.5.0的单SM89官方Base16默认近似Sol，其他配置和双卡保持dense。用`--attention-backend torch-flash`
或`H3Pipeline(..., attention_backend="torch-flash")`保留之前的dense行为。
详见[选择策略、依赖和质量边界](../reference/sol-engine-alignment)，不承诺成片相同。

从发布版源码安装完整链路依赖，并在 `PATH` 中提供 `ffmpeg` 与 `ffprobe`：

```bash
python -m pip install '.[pipeline]'
# 单SM89 Base16默认所需；需要Git和网络，不下载模型。
python -m vflash.install_sol
```

依赖固定到经过核查的适配器版本，其中 Diffusers 的提交为 `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`。安装包不会自动下载模型权重。

本地资产配置明确指定六个路径：

| 字段 | 内容 |
| --- | --- |
| `model_directory` | `MiniMaxAI/MiniMax-H3` 在 `42ed227ee7df40d41602854ae760620d6eb651fe` 版本的官方 Diffusers 组件，包含所选模式的 `transformer_ref` 或 `transformer` |
| `adapter_path` | Turbo 配置使用[运行资产](../reference/runtime-assets)中的固定 BF16 LoRA；Base16 I2VA 和 FL2VA 配置均填 `null` |
| `decoder_directory` | 同一官方 H3 版本的 `FL2VA` 目录，包含 `video_vae` 和 `audio_vae` |
| `artifact` | 所选配置的完整 BF16 原生资产；只有 Turbo 配置包含运行时 LoRA 残差 |
| `schedule_overlay` | 对应的 training-Euler 调度：Turbo 为四次计算，Base16 I2VA/FL2VA 为 16 次且视频/音频 shift 为 12/3 |
| `auxiliary_tensor` | 与上述资产匹配的原生输入、输出权重 |

后三项是准备好的原生资产，不能直接填入任意官方 checkpoint。可以按[官方权重编译流程](./compile-weights)创建；使用已有资产前，请核对[资产合同](../reference/runtime-assets)。目前尚未发布可直接下载的原生模型包。

先把资产放入最终的只读快照，再执行准备步骤。该步骤会按照内置的官方文件清单或原生资产清单，逐一核验实际字节，并检查来源、LoRA 和调度。这是一次性的磁盘操作。准备记录绑定到当前文件系统；后续启动和请求仅检查文件身份、大小及时间戳，不会反复计算模型权重哈希。移动或修改资产后需要重新准备。

```bash
vflash prepare-pipeline \
  --assets pipeline-assets.json --receipt prepared-assets.json
```

## 使用命令行生成

将 H3 提示词写入 `prompt.txt`，然后按期望的顺序传入参考图片：

```bash
vflash generate \
  --prepared-assets prepared-assets.json \
  --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

Ref2VA 的 `--reference` 可以出现一至三次。T2VA 在准备时指定 `--profile t2va-turbo4-exact-sm89`，生成时不传图像参数。一个已准备的 SM89 或 SM86 Base16 关键帧配置可只传 `--first-frame first-frame.png` 生成 I2VA，只传 `--last-frame last-frame.png` 生成 L2VA，或同时传两者生成 FL2VA；`--duration` 接受 5–10 的整数秒，省略时仍为五秒。显式的 `i2va-*` 与 `fl2va-*` profile ID 继续用于稳定的准备和来源记录；L2VA 复用 FL2VA 权重，不增加重复 profile。关键帧不可与 `--reference` 混用；SM86 配置须传入 `--peer-gpu 1 --strategy sequence-head`，0.4.0 也允许 SM89 Base16 关键帧配置用同样参数选择可选的双卡执行。Python 中，L2VA 使用 `VideoRequest(last_frame=Path("last-frame.png"), ...)`，FL2VA 使用 `VideoRequest(first_frame=Path("first-frame.png"), last_frame=Path("last-frame.png"), ...)`。

默认交付仍保留官方 VAE 重建后的关键帧。设置 `keyframe_delivery_profile="exact-v1"` 或传入 `--keyframe-delivery-profile exact-v1` 后，解码阶段会恢复每个已提供的时间端点。图像先按照官方条件编码的拉伸几何，用 LANCZOS 调整到目标画布；端点在 H.264 量化前与输入一致，再用固定四帧线性过渡衔接模型运动。结果会记录实际策略和被修改的帧序号。这是一项显式交付策略，并不代表模型能在整个视频中保留同等精细度。

进度以 JSON 行写入 stderr，stdout 输出最终结果。每次命令独立加载并释放模型；要在多次请求间实现跨模式常驻，应复用同一个 Python `H3Pipeline` 实例。预装环境及完整挂载示例见 [Docker 生成](./docker#pipeline)。

### 无音频生成（0.6.32）

Python 请求可设置 `VideoRequest(..., generate_audio=False)`。此单卡选项在去噪前移除生成音频行，
跳过音频投影、调度更新、音频 latent 导出及波形解码，交付不含音轨的 MP4。
须保留 `audio_delivery_profile="unchanged"`，不能同时使用增益归一化或静音音轨交付；不支持音频条件输入。

移除联合音频上下文可能改变相同种子下的视频。这是一种独立生成模式，不保证像素相同或大幅提速。
完整实测覆盖一张 RTX PRO 6000 Blackwell Server 上的实验 FP8 hybrid：同进程有声/无声切换、
576×864 五秒人物片，以及1248×832、1728×1152、2048×2048十秒无音轨片。
输出已完整解码与播放；局部连续人脸裁剪不构成通用质量资格。
其他架构、默认 BF16 主干及多卡执行尚未取得此选项的实卡资格。
0.6.32 包含此 Python 选项，尚未增加对应 CLI 参数。

## 一个明确拥有资源的实例

源码 main 新增 `audio_delivery_profile="silent-v1"`（CLI：`--audio-delivery-profile silent-v1`），
按原时长和采样率交付数字零的立体声 AAC 音轨，并跳过不需要的波形解码。联合音视频去噪、视频 latent、
视频解码与端点交付均不改变。引擎不解析提示词来猜测静音；应用必须根据明确的全片静音意图选择此项，
不能把未指定声音或不要配乐当作静音。默认仍是 `unchanged`。

这是确定性交付合同，不代表模型音频理解提升，也不是去噪加速；现有 0.4.0 tag 不包含此能力。

在独立进程中创建实例，且此前不能由其他代码初始化 CUDA。显卡由调用方明确选择。下例要求该进程只能看到一张兼容显卡：

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = discover_nvidia_devices()
if len(devices) != 1:
    raise RuntimeError("make one SM89 GPU visible to this process")
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))
with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    result = pipeline.generate(
        VideoRequest(
            prompt=Path("prompt.txt").read_text(encoding="utf-8"),
            references=(Path("subject.png"), Path("setting.png")),
            seed=1234,
        ),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

原有的单图参数 `reference=Path(...)` 仍可使用；它与 `references=(...)` 二选一。

使用双张 RTX 3080 20 GB 时，先准备 [SM86 资产](../reference/pipeline-profiles#sm86)，让进程仅看到这两张卡，并向 `H3Pipeline` 传入 `peer_device=devices[1], strategy="sequence-head"`。编码和解码使用 `devices[0]`，原生去噪使用双卡。单 SM86 或 `tensor` 完整链路会在模型加载前拒绝。命令行对应选项为 `--gpu 0 --peer-gpu 1 --strategy sequence-head`。

0.4.0 也允许 SM89 Base16 关键帧配置在恰好可见两张匹配 RTX 4090 48 GB 时使用同一组 Python 与命令行参数。这里只接受 `sequence-head`；两张卡共享同一 SM89 工件，并各自使用 block ring。一个固定的十秒、736 × 992 L2VA 请求把成功 `generate` 耗时从 774.153 秒降到 484.232 秒，输出 MP4 字节一致，但总共消耗 968.464 GPU 秒，而单卡为 774.153 GPU 秒。这只是有界的单请求低延迟结果，不是吞吐默认值；同时有两个待执行请求时，应使用两个独立单卡 worker。

0.4.0 的构造函数只在 CPU 上检查配置；首次 `generate` 先完整读取、校验素材，再加载模型。服务可在接单前调用 `pipeline.prepare()` 预加载模型。重复调用会复用已有阶段，输入错误也不会销毁健康、已加载的实例。预加载不会执行条件编码或编译每种输入形状；首次使用仍可能产生算子准备成本。

`trust_local_code=True` 允许加载已核验本地快照中的官方解码器 Python 文件。请先阅读[模型与代码许可证](../reference/license)。

同一实例可以连续处理多个请求。原生权重采用 block ring，为轮流使用显卡的编码器与 VAE 留出空间；CPU 模型副本由实例持有以便复用，因此也需要足够的主机内存。进度回调同步执行，抛出异常可取消请求；不要在回调内部调用 `close`。任务队列、账号、存储及多进程调度由应用负责。

当前 main 还提供默认关闭的精确复用：同一创作请求中串行执行的 I2VA、L2VA 或 FL2VA 候选，可共享不受 seed 影响的文本和关键帧编码。可信调度器为同一次已受理请求的兄弟候选创建一个不透明 `ConditioningReuseScope`，并在每次 `generate` 时以 `conditioning_reuse_scope` 传入。不能从公开提示词或媒体内容推导scope，不能让终端用户传入，也不能跨owner或请求复用；不传scope时完全保持普通无缓存路径。

缓存只保存一个CPU副本，最多128 MiB；一小时无活动后过期，每次命中会刷新期限。scope或输入变化、无scope调用、捕获失败、以及`close()`都会立即清空。缓存仅包含Qwen3-VL文本输出、token tag和干净关键帧VAE latent；初始噪声、加噪后的参考latent、packed状态、scheduler状态、DiT/attention/FFN激活、解码latent和媒体都不会复用。可从`result.stages["encoding"]["capture_diagnostics"]["conditioning_reuse"]`读取`hit`、`miss`或`capacity-bypass`，以及编码器调用次数和耗时。

在一个双SM86 sequence-head实例上，预热后的5秒、512 × 512完整A/B/A把两个串行候选的总耗时从控制中位`352.393`秒降到`340.403`秒（`3.402%`），首候选没有变慢，两个候选的H.264和AAC流均跨三臂一致。这只是有界的SM86证据，不代表SM89或线上服务已经获得同样收益。有独立显卡时仍应一张卡一个worker并行执行；为了命中缓存而把本可并行的候选串行化，会损害首结果时延和全fleet吞吐。

### 显式内存条件交接 {#memory-conditioning}

`H3Pipeline(..., conditioning_transport="memory")` 为单设备、同进程流水线启用经过校验的一次性条件交接。张量保留原存储、只消费一次，去掉中间条件张量的写入与重读，不跨请求缓存提示词或张量。typed sibling reuse scope 的既有所有权和过期规则保持；hybrid Ref2VA 已有独立内存参考图，此选项不改变该图。

默认仍为 `conditioning_transport="file"`。独立原生调用、服务任务和跨进程传输保留内容绑定的持久条件包。内存选项拒绝双设备流水线，也拒绝没有返回一次性 typed owner 的编码器。实际路径、张量字节和完成耗时见 `result.stages["encoding"]`。此前硬件 A/B/A 保持输出字节一致但未获总耗时收益；去掉文件 I/O 不会消除模型切换与主机内存成本，评估时应区分冷初始化和暖请求。

后续同一H100 SXM六组I2VA实测采用原生／文本FP8、Veda四十层与BF16 CuTe十层、阶段化文本、预准备BF16 VAE及CPU帧装配。十二片均完整解码与播放，六对解码视频平面和音频PCM一致。私有集成源码为`video-gen` `bf708735`，汇总依据`a70b3907`；这不新增公开量化模型API。

| 画布 / 秒 | 文件整片秒 | 内存整片秒 | 文件finish秒 | 内存finish秒 |
| --- | ---: | ---: | ---: | ---: |
| 512² / 5 | 59.856 | 16.277 | 0.267 | 0.001 |
| 768² / 10 | 45.686 | 37.389 | 0.970 | 0.008 |
| 1024² / 15 | 117.724 | 110.484 | 2.537 | 0.006 |
| 1536 × 1344 / 10 | 180.765 | 166.866 | 3.141 | 0.004 |
| 2048² / 5 | 205.243 | 192.746 | 4.472 | 0.005 |
| 2048² / 15 | 908.845 | 886.411 | 9.758 | 0.015 |

直接省去的finish合计**21.107秒**。初始化加六例1565.830→1427.246秒，观察减少8.85%；后臂共享文件／编译暖缓存，因此不能全归因于交接。另一worker在内存前两例后共享只读模型存储。最大条件所有者5.392GB，两臂对应去噪峰值最高65.756GB，容器累计生命周期峰值93.305GB；累计峰值不能证明逐请求内存相同或最小RAM要求。媒体一致支持此次纯交接改动，不新增语义质量或其他设备／输入模式资格。

集成验证使用了 240 GiB 主机内存上限。这是已测预算，不是测得的最低要求；去噪器单独运行时的 64 GiB 建议不包含这些编码器和解码器。

`VideoResult.elapsed_seconds` 覆盖成功 `generate` 从输入校验到参考素材清理的完整耗时，包括首次模型加载。`stages.initialization_seconds` 是本次调用中的加载耗时，显式预加载后为零；`stages.request_elapsed_seconds` 仅扣除这部分加载，便于比较请求。`stages.session_initialization_seconds` 保留实例最初的加载成本，也可读取 `pipeline.initialization_seconds`。`stages.input_preparation` 包含资产检查和素材加载，其中 `reference_loading_seconds` 已计入准备耗时。编码和媒体阶段还分别记录 `weight_resume_seconds`、`suspend_seconds`，以及 `capture_call_seconds` 或 `decode_call_seconds`；编码阶段也会记录 `conditioning_transport` 和捕获的 `conditioning_tensor_bytes`。外层阶段同时包含同步回调和协调开销。这些明细存在包含关系，不能全部视作独立耗时相加。

`stages.encoding.capture_diagnostics` 会进一步拆分官方 conditioning 调用、捕获钩子等待、metadata 处理和持久化 bundle 收尾。收尾记录区分 tensor 写入与 seal/reload，`process_deltas` 则报告对应区间的页故障、块 I/O 和上下文切换。钩子等待已包含在 `official_pipeline_seconds` 中，全部字段又都包含在 `capture_call_seconds` 中；它们只用于归因，不能作为独立阶段相加。进程计数也不是整机资源统计。

输出路径不能已经存在。编码、媒体核验以及 GPU 阶段清理成功后才发布视频。执行失败会关闭此实例并移除临时文件。`close()` 在等待 CUDA 完成后释放持有的模型与 hook，不会重置其他调用方的 CUDA 上下文。模型关闭后，CUDA 库仍可能保留进程级工作缓冲；需要释放整个 CUDA 上下文时，应退出该独立进程。

在只读容器中运行时，需要为 Triton 提供可写、且允许加载编译后共享库的缓存目录。带有 `noexec` 的临时文件系统不能作为该缓存目录。

## 使用参考视频 {#reference-video}

安装完整链路依赖或使用带版本号的 pipeline 镜像，使用与图片生成相同的 `ref2va-turbo4-exact-sm89` 准备资产和单张 RTX 4090 48 GB。这是依据参考生成新视频，不是逐帧精确编辑。

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file video-prompt.txt \
  --reference-video reference.mp4 --gpu 0 --seed 1234 \
  --output variation.mp4 --trust-local-code
```

Python 使用 `VideoRequest(prompt=..., reference_video=Path("reference.mp4"))`。提示词以 **`<Video 1>`** 指代视频，保留其中的空格。同一 Ref4 实例可先接收图片、再接收视频、再回到图片，不切换模型或 LoRA。此前生成的本地 MP4 也可以作为下一次的 `reference_video`；引擎不会自动载入对话历史或改写提示词。

输入限一段完整的 **2–5 秒 MP4、MOV 或 WebM**，最大 **20 MiB**，源画面最多 **475,136 像素**（928×512 的面积）。源帧率可以是分数，必须可读取、大于零且不超过 240 fps。只接受一个视频流、方形像素和无歧义的直角旋转。已有源音轨会被丢弃，不参与新音轨生成；暂不支持图视频混合、音频输入、SM86 或 Ref8 视频条件。输出仍为五秒、24 fps，沿用上文的 32 对齐与画布面积限制。

CPU 解码器先固定文件副本，检查完整视频并重采样到 24 fps，保留片尾不足一帧的时间区间。它向官方视频 setup 传递源尺寸 RGB，再由官方缩放一次。视频参考尺寸独立于输出画布，**不使用图片的只缩小 `match` 策略**。入口也限制官方归一化后的画布：长边最多 1376、面积最多 1376×768、时序参考行最多 33,024。因此，即使源像素合格，过于狭长的比例仍可能被拒绝。预算限制不代表每种比例都已通过质量验证。临时文件和解码 RGB 只存活于当前请求。

视频条件比静态图片更耗资源：官方 VAE 编码完整时序块，文本编码器从视频中采样。请根据输入准备、条件编码、去噪和媒体交付的实际耗时规划服务；这里不承诺视频输入延迟。原生接口和支持范围见[视频条件包](../reference/runtime-assets#video-conditioning)。

## 如何理解正确性验证

条件张量与最终 latent 的实现一致性，应与画面质量分别评估。动作、外观、指令与声音都需要对照用户原始要求；张量一致不能让质量不合格的结果通过。

媒体验证分别检查编码前的画面与音频、所选的五至十五秒交付时钟、帧数和声道。H.264 和 AAC 均为有损编码，整段 MP4 的哈希不能用于判断去噪器或音频解码器的数值等价性。

发布检查分别覆盖 14 个条件张量、最终 FP32 音视频 latent、解码画面及可播放成片。官方音频 VAE 在重复请求间可能产生微小浮点差异，早于 PCM 或 AAC 编码；因此不承诺音频逐位复现。音频有限值、声道、帧数、时钟及完整解码仍须通过检查。具体案例与适用范围见[发布验收](../reference/releases)。

显式[Veda后端](./veda)也接受单SM89 hybrid；近似执行合同和证据单独记录。

## 十五秒与九张图像（0.6.12） {#extended-hybrid}

显式原始v0.1 **单SM89 hybrid＋Veda**接受5–15整数秒的图像条件请求及最多九张有序图像参考。普通Ref4/Base16、SM86、双卡和视频参考合同不继承该扩展。H100不属于本版支持或实测硬件。

限制必须组合判断：宽高32对齐、最长边2048，且`宽×高×模型帧数`不得超过`2048²×124`。扩展多图Ref限五秒和960×544总像素。十五秒关键帧或单图参考使用362内部模型帧，精确交付24fps的360帧及每声道480000音频采样；内部padding不延长用户指定时长。不合格组合在模型激活前拒绝。

固定官方条件适配器接受VAE对齐后的时间边界，不修改其逻辑时长上限。九张图按传入顺序保留`<Picture 1>`至`<Picture 9>`标签，属于视觉参考而非九个时间线关键帧。更多输入增加条件处理量，不能保证所有细节都遵循。完整目标硬件案例和视觉限制见[性能记录](../reference/performance#extended-hybrid)。


## 解码硬削波前的音频余量 {#pre-clamp-audio-headroom}

0.6.33新增适用于独占原始H3音频VAE组件的显式上下文，一次只包裹完整双声道解码：

```python
from vflash.media.audio_headroom import audio_decoder_headroom

observation = {}
with audio_decoder_headroom(audio_component, observation):
    waveform = decode_official_h3_audio_latents(
        audio_component, latents, device=device, torch_module=torch
    )
```

固定0.98采样峰值上限对双声道使用相同增益，不放大小音量。解码或诊断回调失败也会
移除钩子。可选 `on_unclamped(tensor)` 接收原始 `[2, 1, samples]` 波形；诊断读写
及其开销由调用方负责。此上下文内不得让另一请求共享组件。普通解码、采样器和完整
流水线默认配置保持，不新增CLI质量选择器。

这避免原解码器在交付处理前破坏越界峰值；已有 `web-v1` 后置响度处理无法恢复已被
削平的样本。两者都不能修复模型生成的音色、人声或音画对应。采样峰值上限也不等于
真峰值或有损编码后的上限保证。实测来自一张RTX PRO 6000 Blackwell Server的两条
补采双声道音轨：一条完全保持，另一条约衰减0.31 dB即消除了新增削波；不由此宣称
补采感知质量全面合格，也不扩大显卡资格。


## 显式原模型人脸低噪重绘

0.6.34新增可选的原生低噪重绘基础接口 `VideoRepaint` 与
`compile_refinement_tables` / `load_tables`。保留Turbo残差，按实际低噪sigma
编译原FL/Ref调制表，保护已给定视频行并冻结音频潜变量；不改变普通生成默认值。
用法见英文页的完整Python示例：四步、denoise=0.035，需要同一官方VAE编码得到的
原裁剪视频packed rows，而非重新生成的独立人脸图。

调用方负责检测、无歧义连续跟踪、端点匹配、保留原片及局部合成。检测置信度不是身份
一致性的证明；合成复制原音轨也不代表口型绝对不变。本接口不是自动换脸服务。
完整视频证据限定PRO 6000 Server驻留hybrid主干的512裁剪、5/10秒素材。一次真实
同进程串联为：有声生成32.31秒、重新检测/准备2.91秒、静音重绘22.71秒、全尺寸合成
0.99秒，初始化与交付另计。已选单镜头人物有用户认可的改善；多人、遮挡和切镜头不能
据此宣称普遍修复。

两槽BF16流水路径同时替换主存调制表、设备槽及预校验复制映射，并在成功/失败后同步
恢复原状态。RTX 3080上100次真实CUDA复制验证通过；这不是SM89完整H3成片资格，
也不表示正式服务已升级。流水路径目前仅允许人脸重绘，多GPU重绘未支持。

跟踪/裁剪/重绘/回贴思路参考MIT许可的
[ComfyUI H3 FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine)，
没有附带身份识别或第三方人脸恢复权重。


### 可选的裁剪编码执行方式

0.6.35提供 `vflash.media.face_encoder.face_encoder(vae, "batch2-fp16-norm")` 上下文。
在其中对完整裁剪调用固定官方 `encode_vae_condition`，保持官方归一化和FP32后验。
仅支持batch=1、320/384/512方形、120–362模型帧、原17帧块/丢弃3个token的几何。
时间块在batch维独立编码，不混合时间；原模型FP32参数不变，但FP16运算和仿射副本属于近似。
VAE必须由当前调用独占，成功或异常退出均恢复编码和逐帧归一化方法。默认 `original` 不改执行。
检测、最终画布合成及原声保留由调用方负责；此接口不替换解码器或减少帧数。
较小内部裁剪需要最终视频的细节检查，不能把它的尺寸当成交付分辨率。
实际收益与质量边界见[版本记录](../reference/releases.md)。


## 显式Veda密集边界宽度（0.6.36） {#veda-dense-edges}

`H3Pipeline(..., veda_dense_edge_layers=3)` 或 `vflash generate --veda-dense-edge-layers 3`
选择六个密集层（0–2、47–49）和44个Veda稀疏层。默认仍是前后各五层（40稀疏/10密集）。
该选项要求Veda注意力，改变近似推理，不承诺数值等价；每个会话独立持有分配，实际调用计数及元数据一致。

RTX PRO 6000 Server同卡、四步hybrid FP8/FROST控制实验中，1/2/4MP十秒完整生成分别少
6.60%/9.07%/12.29%；另一个1MP十秒A/B/A对照少4.58%。有限多模式研究也已完成，
不扩大公开流水线原有模式或容量准入。抽样上下文和连续局部帧未见新增明显缺陷，但完整感知、
声音及跨设备资格仍开放。显式改回5即可回滚；暖态生成时钟不包含模型启动和发布。

上述耗时来自会话独立接口发布前的受控实验分层；精确0.6.36现已完成四模式GPU重放，
视频像素与对应早期44/6结果一致，不据此宣称更广感知质量或音频等价。


## 原生重绘请求（0.8.0）

当前源码在常驻`NativeEngineSession.generate`中增加`VideoRepaintRequest`，接收独立匹配的
两步或四步低噪声调制表，以及官方VAE产生的打包视频行。原四步I2VA条件包照常校验，
会话基础overlay保持，临时输出头/调制绑定在每次执行后恢复。必须指定`generate_audio=False`；该接口暂不支持多卡或
去噪profile。检测、裁剪/关键帧一致性和原音轨合成仍由应用负责。

```python
from vflash.native.h3_refinement import VideoRepaintRequest
from vflash.native.h3_refinement_schedule import load_tables

_, schedule, tables = load_tables("face-modulation-two-step", mode="face")
result = session.generate(
    conditioning_bundle, output_latents, generate_audio=False,
    video_repaint=VideoRepaintRequest(schedule, tuple(tables), encoded_rows),
)
```

两步表使用`compile_refinement_tables(..., steps=2, denoise=0.035, mode="face")`编译，不能截取
不相关的Turbo调制表。结果NFE、进度及Veda调用覆盖按实际执行次数记录。重绘结果的
`execution_schedule`明确标注`video-low-noise-repaint`，从真实低噪声sigma开始并声明音频冻结；
它不是普通条件调度工件，`refinement`另记实际执行。

新接口已在一张RTX PRO6000 Blackwell Server、显式44/6下完成两个五秒人脸场景；其中一组
旧/原生两步对照的全部120帧解码像素一致。两步相对四步的编码/重绘/媒体/合成区间减少17.5–18.6%，
不含复用的修复输入及公共初始化，不据此宣称普遍感知等价或完整请求耗时达标。

使用已明确验证的分配，例如`veda_dense_edge_layers=3`（44层稀疏/6层密集）。零密集边界不受支持：
全量稀疏探索结果存在明显感知质量退化，未采用；默认五层密集边界仍可选择。


## 直接RGB交付与重复计算消除（0.9.0）

`H3Pipeline.generate_rgb(request, Path("repaint.npy"))`是显式Python纯视频接口，
保持相同条件、去噪与原始VAE，解码后仅量化一次uint8，返回形状
`[model_frames, height, width, 3]`的NumPy数组。五秒请求有120个交付帧和124个模型帧，
元数据分别声明。普通`generate`继续交付MP4。RGB接口要求`generate_audio=False`和
decoded关键帧，不会静默丢弃请求的音频或精确端点交付。

可选`required_frames=(0, 1, ...)`必须为排序、唯一的模型帧索引，供后续合成器使用。
保留原始时间组装与重叠融合，仅跳过对这些帧没有贡献的VAE块。未选帧写零，
**不能将部分数组当作完整视频**。元数据声明有效索引和实际解码块数；原音轨和完整画面合成
仍由调用方负责。数组在映射写入前实际预留磁盘空间，失败不发布部分输出，不覆盖现有文件。

显式`batch2-fp16-norm-reuse`编码器在已验证的124帧时间线上仅复用完全相同的17帧输入，
保持batch-two形状、FP16计算/归一化以及采样前的完整后验时间线。其它受支持时长沿用普通
batch-two计算。这不意味着原有FP16编码器与FP32等价。

算法原型已在同一张RTX PRO6000上完成连续人物与分段人物两个五秒场景；重复编码复用与
选择性解码的最终RGB帧和原音轨均保持。收益取决于实际重复输入和未使用的输出区间，
不作通用速度保证。此显式公开API已完成同卡控制/候选共六次完整请求，三对结果各120帧RGB及原音轨PCM保持。
原生两步与44/6固定；普通MP4默认保持，接入服务的实际部署单独验收。
