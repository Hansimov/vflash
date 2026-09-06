# 完整模型配置

每份准备记录固定模型、LoRA 与调度方式。Vflash 0.2.0 在单张 RTX 4090 48 GB 上支持以下完整链路：

| 配置 | 输入 | Transformer | LoRA | 视频/音频 shift |
| --- | --- | --- | --- | --- |
| `ref2va-turbo4-exact-sm89` | 提示词加 1–3 张有序图片 | `transformer_ref` | Ref Turbo4 v0.1，alpha 8 / rank 128 | 12 / 3 |
| `t2va-turbo4-exact-sm89` | 纯文字提示词 | `transformer` | Base Turbo4 v1.0，alpha 128 / rank 128 | 6 / 3 |

两者都使用四次计算、BF16 权重和独立的 LoRA 残差。默认配置为 Ref4。实例运行中不会切换 Base 与 Ref 权重；应用需要两种模式时，应分别准备资产和常驻实例。原生接口的支持范围见[配置与硬件](../guide/profiles)；本版尚不支持 SM86 的完整生成链路。

## 准备文生视频

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

文件放到最终位置后，进行一次准备和哈希校验；启动读取本地记录，不再完整重读模型。常驻实例在接收请求之前初始化，编码、去噪和解码依次使用同一张显卡，CPU 模型副本留待复用。准备、初始化、首次请求和热请求应分别计时。取消正在执行的请求会关闭实例；继续生成前需要新建实例。
