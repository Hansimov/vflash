# 完整模型配置

每份准备记录固定模型、LoRA 与调度方式。Vflash 0.3.1 支持以下完整链路：

| 配置 | 硬件 | 输入 | Transformer | LoRA | 视频/音频 shift |
| --- | --- | --- | --- | --- | --- |
| `ref2va-turbo4-exact-sm89` | 单张 RTX 4090 48 GB | 提示词加 1–3 张有序图片，或一段 2–5 秒视频 | `transformer_ref` | Ref Turbo4 v0.1，alpha 8 / rank 128 | 12 / 3 |
| `ref2va-turbo4-exact-sm86` | 双张 RTX 3080 20 GB，`sequence-head` | 提示词加 1–3 张有序图片 | `transformer_ref` | Ref Turbo4 v0.1，alpha 8 / rank 128 | 12 / 3 |
| `t2va-turbo4-exact-sm89` | 单张 RTX 4090 48 GB | 纯文字提示词 | `transformer` | Base Turbo4 v1.0，alpha 128 / rank 128 | 6 / 3 |

所有配置都使用四次计算、BF16 权重和独立的 LoRA 残差，输出五秒、24 fps 视频。默认为 SM89 Ref4。实例运行中不会切换 Base 与 Ref 权重；应用需要两种模式时，应分别准备资产和常驻实例。原生单 SM86 与 Turbo8 接口具有不同的[验证范围](../guide/profiles)。

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

`ref4-sm86-assets.json` 的六个字段应填写新生成的 SM86 权重、调度和辅助张量路径。编码器、解码器和原始 LoRA 可以与 SM89 共享同一份只读文件。编码和解码使用主卡，两卡共同执行原生去噪。完整链路会在加载模型前拒绝单 SM86 或 `tensor` 策略；原生 latent 接口继续保留两种并行策略。

## 准备文生视频

下一版本的 `t2va-turbo4-exact-sm86` 为双 3080 的 `sequence-head` 路径固定了
同一 Base4 v1.0 原始模型与 LoRA。SM86 编译和实际安装的原生会话已在应用持有的
编码/core/媒体链路完成五秒 928 × 512、十秒 640 × 352 两次请求，均为 24 fps。
这是原生集成边界；新配置的独立 `H3Pipeline` 包装尚未重跑，公开时长合同仍是五秒。
[发布说明](./releases#v0-3-2)保留证据局限。必须生成独立的 SM86 准备记录与编译表，
不能修改 SM89 或 Ref 资产标签来替代。T2VA 不开放单卡、`tensor` 或八步执行。

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

文件放到最终位置后，进行一次准备和哈希校验；启动读取本地记录，不再完整重读模型。0.3.1 可通过 `H3Pipeline.prepare()` 显式预加载，否则首次请求在 CPU 输入检查完成后加载模型，并保留以便复用；各阶段使用的显卡由上述配置决定。预加载不执行条件编码或准备所有输入形状。资产准备、模型加载、首次请求和重复请求应分别计时。取消正在执行的请求会关闭实例；继续生成前需要新建实例。

[参考视频输入](../guide/complete-pipeline#reference-video)复用单 SM89 Ref4 资产，已通过同一实例连续处理图片、视频、图片的完整验证。SM86 和 Turbo8 的视频参考仍不在支持范围内。
