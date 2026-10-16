# 开始使用

Vflash 把一段提示词和可选图片生成为带音轨的完整 MP4。大多数用户使用**完整流水线**
（`vflash generate`、`H3Pipeline` 或 Docker 流水线镜像）。已自行构建条件输入的应用可以使用更底层的
**原生去噪器**，它返回 latent 而不是视频（[见下文](#native-latents)）。

Vflash 不会隐式下载或编译任何内容：你选择与显卡匹配的配置、一次性准备其资源；超出该配置契约的请求会被直接拒绝，
而不是悄悄更换模型、设备或精度。

## 1. 安装 {#install}

需要 Linux、Python 3.11+、支持 CUDA 13.0 的 NVIDIA 驱动，以及 `PATH` 中的 `ffmpeg`/`ffprobe`。
基础安装是不含 PyTorch 的轻量 CLI；按所用接口安装对应 extra。

```bash
git clone --branch v0.15.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e '.[pipeline]'     # 完整 MP4 流水线（已包含 GPU 依赖）
```

只用原生去噪器时安装 `.[gpu]`，原生 HTTP 服务安装 `.[server]`。偏好容器时，[Docker 镜像](./docker)
固定了相同环境（PyTorch 2.11 / CUDA 13.0）。

## 2. 检查显卡 {#check-your-gpu}

```bash
vflash doctor      # 当前进程可见的显卡
vflash profiles    # 本版本提供的配置
```

选择与硬件对应的配置，用 `plan` 在不加载权重的情况下检查是否匹配：

| 显卡 | 常用配置 | 命令 |
| --- | --- | --- |
| RTX 4090 48 GB | `i2va-turbo4-v01-544-exact-sm89`（关键帧、混合模型、原生高清） | `vflash plan i2va-turbo4-v01-544-exact-sm89 --gpu 0` |
| RTX PRO 6000 96 GB | `i2va-turbo4-v01-544-exact-sm120`（混合 I2VA，预览） | `vflash plan i2va-turbo4-v01-544-exact-sm120 --gpu 0` |
| H100 80 GB | `i2va-turbo4-v01-544-exact-sm90`（混合 I2VA，预览） | `vflash plan i2va-turbo4-v01-544-exact-sm90 --gpu 0` |
| 两张 RTX 3080 20 GB | `ref2va-turbo4-exact-sm86` | `vflash plan ref2va-turbo4-exact-sm86 --gpu 0 --peer-gpu 1 --strategy sequence-head` |

4090 上还提供纯文本（`t2va-turbo4-exact-sm89`）、参考图（`ref2va-turbo4-exact-sm89`）和 Base16
关键帧配置。表中显存是硬性要求：24 GB 的 4090 与 10/12 GB 的 3080 不在支持范围内。主机内存、常驻方式
与各显卡的实测范围见[选择硬件与配置](./profiles)。

## 3. 准备模型资源 {#prepare}

每个配置需要固定版本的官方 MiniMax H3 组件、其固定版本的 Turbo LoRA（如有），以及三个编译后的原生文件
（artifact、调度和辅助张量）。用[官方权重编译器](./compile-weights)下载并编译，在 JSON 中填写六个路径
（[字段说明](./complete-pipeline#installation-and-assets)），然后一次性绑定：

```bash
vflash prepare-pipeline --assets pipeline-assets.json --receipt prepared-assets.json
```

凭据记录每个文件的哈希。之后启动只检查身份与时间戳，因此请把资源放在最终的只读位置；移动后需要重新生成凭据。

## 4. 生成第一个视频 {#first-video}

把提示词写入 UTF-8 文件后执行一次请求。以首帧图片（I2VA）为例：

```bash
vflash generate --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --first-frame first.png --width 928 --height 512 --duration 5 --seed 1234 \
  --gpu 0 --output video.mp4 --trust-local-code
```

- 纯文本：准备 `t2va-turbo4-exact-sm89`，省略图片参数。
- 参考图：重复 `--reference`（提示词中按顺序称为 `<Picture 1>`、`<Picture 2>` …）。
- 尾帧或首尾帧：`--last-frame`，或同时使用 `--first-frame` 与 `--last-frame`。

进度以 JSON 行写到 stderr，结果写到 stdout。`--trust-local-code` 允许从已校验的本地快照加载官方解码器代码，
使用前请阅读[许可证](../reference/license)。每次 CLI 调用都会加载并释放模型，因此首次运行包含模型加载时间。

十五秒请求、最多九张参考图以及最高 2720×1536 的原生高清需要通过 Python 使用混合模型，
做法见[生成视频](./complete-pipeline#extended-hybrid)。

## 5. 保持模型常驻 {#reuse}

需要处理多个请求时，保持一个进程运行：

- **Python：** 一个 [`H3Pipeline`](./python) 串行处理请求并复用已加载的权重。
- **Docker / HTTP：** [原生服务](./docker)排队处理任务并常驻模型。

请把首个（冷）请求与重复（暖）请求分开测量，见[速度、显存与质量](../reference/performance)。

## 原生 latent（进阶） {#native-latents}

`vflash denoise` 和原生 HTTP 服务接收编译好的条件 bundle，输出视频与音频 **latent**（`.safetensors`），
而不是 MP4。仅当你的应用自行负责条件构建与解码时使用：

```bash
vflash denoise ref2va-turbo4-exact-sm89 --gpu 0 \
  --artifact /path/to/artifact --schedule-overlay /path/to/schedule \
  --auxiliary-tensor /path/to/auxiliary.safetensors \
  --bundle /path/to/bundle --output-latents result.safetensors
```

两张 RTX 3080 通过 `--peer-gpu 1 --strategy sequence-head` 协作处理一个请求（选择 peer 时默认使用该策略，
另一种是 `tensor`）。结果不保证与单卡逐位一致。四类输入的定义见[运行资源](../reference/runtime-assets)。

## 检查失败时 {#troubleshooting}

| 现象 | 检查 |
| --- | --- |
| 未列出显卡 | `nvidia-smi`；Docker 中确认设备已暴露给容器。 |
| 配置与显卡不匹配 | 按 `doctor` 报告的计算能力和显存选择配置；改配置名不会转换资源。 |
| 资源被拒绝 | 对最终、未改动的文件重新执行 `prepare-pipeline`。 |
| 请求在加载前被拒绝 | 画布、时长或输入数量超出[模式与限制](./complete-pipeline#modes-and-limits)。 |
| 主机内存不足 | 块环与 SM86 worker 把权重放在内存中：每个 worker 预留至少 64 GiB 可用内存。 |

更多情况见[故障排查](./troubleshooting)。
