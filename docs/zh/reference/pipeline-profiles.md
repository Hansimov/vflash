# 完整模型配置

每份准备记录固定模型、LoRA 与调度方式。当前源码支持以下完整链路：

| 配置 | 硬件 | 输入 | Transformer | LoRA | 视频/音频 shift |
| --- | --- | --- | --- | --- | --- |
| `ref2va-turbo4-exact-sm89` | 单张 RTX 4090 48 GB | 提示词加 1–3 张有序图片，或一段 2–5 秒视频 | `transformer_ref` | Ref Turbo4 v0.1，alpha 8 / rank 128 | 12 / 3 |
| `ref2va-turbo4-exact-sm86` | 双张 RTX 3080 20 GB，`sequence-head` | 提示词加 1–3 张有序图片 | `transformer_ref` | Ref Turbo4 v0.1，alpha 8 / rank 128 | 12 / 3 |
| `t2va-turbo4-exact-sm89` | 单张 RTX 4090 48 GB | 纯文字提示词 | `transformer` | Base Turbo4 v1.0，alpha 128 / rank 128 | 6 / 3 |
| `i2va-base16-bf16-sm89`（预览） | 单张 RTX 4090 48 GB | 提示词加一张明确的首帧 | `transformer` | 无 | 12 / 3 |
| `i2va-base16-bf16-sm86`（预览） | 双张 RTX 3080 20 GB，`sequence-head` | 提示词加一张明确的首帧 | `transformer` | 无 | 12 / 3 |
| `fl2va-base16-bf16-sm89`（预览） | 单张 RTX 4090 48 GB | 提示词加明确的首帧和尾帧 | `transformer` | 无 | 12 / 3 |
| `fl2va-base16-bf16-sm86`（预览） | 双张 RTX 3080 20 GB，`sequence-head` | 提示词加明确的首帧和尾帧 | `transformer` | 无 | 12 / 3 |

已发布的 Turbo 配置使用四次计算、BF16 权重和独立的 LoRA 残差，并保留五秒合同；预览 I2VA 和 FL2VA 配置使用官方 Base Transformer，以 BF16、无 LoRA执行 16 次计算，支持五秒（`124 → 120`帧）或十秒（`243 → 240`帧），均为24 fps。默认仍为 SM89 Ref4。同一硬件目标上的成对 Base16 I2VA/FL2VA 配置具有完全相同的模型工件、调度和官方 FL2VA 条件工作流，因此一个已准备的 Base16 关键帧 pipeline 可串行接受 I2VA 与 FL2VA 请求，无需重启 profile 或执行冷初始化；每个请求仍生成与实际模式一致的条件 schema 和来源信息，已准备的 profile ID 继续作为 pipeline 与结果身份。每次请求的阶段驻留仍遵循完整 pipeline 选择的内存策略。Turbo、Ref2VA 和 T2VA 配置仍只接受各自模式。原生单 SM86 与 Turbo8 接口具有不同的[验证范围](../guide/profiles)。

## 准备预览版 Base16 I2VA

使用固定版本的官方 Base Transformer，省略 `--adapter` 来准备和编译权重，再创建通常的六字段完整链路记录。资产 JSON 中的 `adapter_path` 为 `null`，其余字段填写官方模型与解码器目录以及三项编译输出。

```bash
python -m vflash.compiler prepare \
  --profile i2va-base16-bf16-sm89 \
  --transformer models/minimax-h3/transformer \
  --receipt base16-i2va-weights.json
python -m vflash.compiler compile \
  --receipt base16-i2va-weights.json --output models/base16-i2va-native --gpu 0
vflash prepare-pipeline \
  --profile i2va-base16-bf16-sm89 \
  --assets base16-i2va-assets.json --receipt base16-i2va-pipeline.json
vflash generate \
  --prepared-assets base16-i2va-pipeline.json --prompt-file prompt.txt \
  --first-frame first-frame.png --duration 10 --gpu 0 --seed 1234 \
  --output video.mp4 --trust-local-code
```

`--first-frame` 表示第零帧锚点，与 Ref2VA 的 `--reference` 输入相互独立；Python 接口为 `VideoRequest(first_frame=Path(...))`。提示词可以将该图标记为 `<Picture 1>`，超出这一张已提供图片的标签会被拒绝。内部调用官方编码器的单帧 FL2VA 条件路径，对外请求类型仍明确为 I2VA。双 SM86 准备阶段使用 `i2va-base16-bf16-sm86`，生成时增加 `--peer-gpu 1 --strategy sequence-head`。一条 928 × 512、120 帧请求在不计冷启动时耗时 281.2 秒；其中一张卡触发软件热降频，因此该结果只证明可行性和持续产能，不作为无热降频延迟或质量结论。

真正的双锚 FL2VA 请求既可以复用同一硬件上已经准备好的 Base16 I2VA 关键帧 pipeline，也可以编译并准备显式的 `fl2va-base16-bf16-sm89` 或 `fl2va-base16-bf16-sm86` 身份，然后同时传入首尾帧：

```bash
vflash generate \
  --prepared-assets base16-fl2va-pipeline.json --prompt-file prompt.txt \
  --first-frame first-frame.png --last-frame last-frame.png \
  --duration 10 --gpu 0 --seed 1234 --output video.mp4 --trust-local-code
```

Python 接口为 `VideoRequest(first_frame=Path(...), last_frame=Path(...), duration_seconds=10)`。官方 FL2VA workflow 会分别收到 `image` 和 `last_image`；提示词中的 `<Picture 1>` 与 `<Picture 2>` 按时间顺序表示首帧与尾帧。当前窄合同会拒绝只有尾帧而没有首帧的请求。复用成对配置不会改变已加载工件或已准备的 profile ID，只会按当前请求选择对应的 I2VA/FL2VA 条件合同。一条 SM89 pipeline 已在 640×352 下完成同进程十秒 I2VA→FL2VA：两条输出均解码为240帧和十秒双声道音频，第二次请求没有重复初始化。该有界实证不代表十秒928×512或SM86双卡已经资格化。SM86 FL2VA 请求与 I2VA 一样要求双卡 `sequence-head`。

## 准备双 3080 生成 {#sm86}

使用[编译流程](../guide/compile-weights)中的官方 Ref4 文件，但创建两份准备记录时都指定 `ref2va-turbo4-exact-sm86`。编译只需一张 SM86 显卡；生成的时间和调制张量绑定该架构，不能复用已编译的 SM89 资产。

```bash
python -m vflash.compiler prepare \
  --profile ref2va-turbo4-exact-sm86 \
  --transformer models/minimax-h3/transformer_ref \
  --adapter models/adapters/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --receipt ref4-sm86-weights.json
python -m vflash.compiler compile \
  --receipt ref4-sm86-weights.json --output models/ref4-sm86-native --gpu 0
vflash prepare-pipeline \
  --profile ref2va-turbo4-exact-sm86 \
  --assets ref4-sm86-assets.json --receipt ref4-sm86-pipeline.json
vflash generate \
  --prepared-assets ref4-sm86-pipeline.json --prompt-file prompt.txt \
  --reference subject.png --reference setting.png --reference style.png \
  --gpu 0 --peer-gpu 1 --strategy sequence-head \
  --output video.mp4 --seed 1234 --trust-local-code
```

`ref4-sm86-assets.json` 的六个字段应填写新生成的 SM86 权重、调度和辅助张量路径。编码器、解码器和原始 LoRA 可以与 SM89 共享同一份只读文件。编码和解码使用主卡，两卡共同执行原生去噪；`i2va-base16-bf16-sm86` 和 `fl2va-base16-bf16-sm86` 遵循相同的拓扑要求。完整链路会在加载模型前拒绝单 SM86 或 `tensor` 策略；原生 latent 接口继续保留两种并行策略。

## 准备文生视频

本版新增的 `t2va-turbo4-exact-sm86` 为双 3080 的 `sequence-head` 路径固定了
同一 Base4 v1.0 原始模型与 LoRA。SM86 编译和实际安装的原生会话已在应用持有的
编码/core/媒体链路完成五秒 928 × 512、十秒 640 × 352 两次请求，均为 24 fps。
这是原生集成边界；新配置的独立 `H3Pipeline` 包装尚未重跑，公开时长合同仍是五秒。
[发布说明](./releases#v0-3-2)保留证据局限。必须生成独立的 SM86 准备记录与编译表，
不能修改 SM89 或 Ref 资产标签来替代。T2VA 不开放单卡、`tensor` 或八步执行。

使用新原生 SM86 配置时，在下方编译准备命令中选择 `t2va-turbo4-exact-sm86`，并写到独立的 SM86 输出目录。准备匹配条件后可执行：

```bash
vflash denoise t2va-turbo4-exact-sm86 \
  --artifact models/base4-sm86-native/artifact \
  --schedule-overlay models/base4-sm86-native/schedule \
  --auxiliary-tensor models/base4-sm86-native/auxiliary.safetensors \
  --bundle inputs/t2-conditioning \
  --gpu 0 --peer-gpu 1 --strategy sequence-head \
  --output-latents output/latents.safetensors
```

应用提供匹配条件包并解码输出 latent；该命令不直接写 MP4。下方 SM89 示例继续使用完整包装接口。

按[官方权重下载流程](../guide/compile-weights)选择 `t2va-turbo4-exact-sm89`，然后准备 Base 模型和明确固定的 LoRA：

```bash
python -m vflash.compiler prepare \
  --profile t2va-turbo4-exact-sm89 \
  --transformer models/minimax-h3/transformer \
  --adapter models/adapters/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --receipt base4-weights.json
python -m vflash.compiler compile \
  --receipt base4-weights.json --output models/base4-native --gpu 0
vflash prepare-pipeline \
  --profile t2va-turbo4-exact-sm89 \
  --assets base4-assets.json --receipt base4-pipeline.json
vflash generate \
  --prepared-assets base4-pipeline.json --prompt-file prompt.txt \
  --output video.mp4 --gpu 0 --seed 1234 --trust-local-code
```

`base4-assets.json` 的六个字段沿用[完整链路资产格式](../guide/complete-pipeline)。填写 Base LoRA，以及新编译的 Base 权重、调度和辅助张量路径。官方解码器及通用编码器可以共享同一份只读文件。

T2VA 不传 `--reference`；对应的 Python 请求为 `VideoRequest(prompt=..., seed=...)`。Ref2VA 支持一至三张图片并保留顺序。未绑定的 `<Picture N>` 标签和模式不匹配都会在执行前拒绝。

Base LoRA 的上游文件名包含 `fl2v`，本版验证的是其 T2VA 用法，不是首尾帧生成。更新的 Base4 LoRA 不能直接替换这个固定 v1.0 配置。版本、哈希及许可证链接见[运行资产](./runtime-assets)。

## 资源生命周期

文件放到最终位置后，进行一次准备和哈希校验；启动读取本地记录，不再完整重读模型。0.3.2 可通过 `H3Pipeline.prepare()` 显式预加载，否则首次请求在 CPU 输入检查完成后加载模型，并保留以便复用；各阶段使用的显卡由上述配置决定。预加载不执行条件编码或准备所有输入形状。资产准备、模型加载、首次请求和重复请求应分别计时。取消正在执行的请求会关闭实例；继续生成前需要新建实例。

[参考视频输入](../guide/complete-pipeline#reference-video)复用单 SM89 Ref4 资产，已通过同一实例连续处理图片、视频、图片的完整验证。SM86 和 Turbo8 的视频参考仍不在支持范围内。
