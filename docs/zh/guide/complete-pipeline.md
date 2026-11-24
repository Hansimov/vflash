# 从文字或图像生成视频

固定配置 CLI 读取已经校验的资源，输出可播放 MP4。先确定输入方式，再编译对应的主干权重与采样时序。

## 选择输入 {#modes-and-limits}

| 模式 | 输入 | 作用 |
| --- | --- | --- |
| T2VA | 提示词 | 从头生成镜头 |
| I2VA / L2VA | 首帧或尾帧 | 固定一个时间端点 |
| FL2VA | 按时间排列的首尾帧 | 同时约束开场与结尾 |
| Ref2VA | 有序参考图 | 引导整段画面中的外观 |
| 视频参考 | 受支持的短片 | 提供运动或场景条件 |

普通 Ref4 配方接收一至三张外观图；首尾帧模式可共享关键帧权重。显卡分配前应检查宽高是否对齐 32 像素，并核对该模式下像素数与帧数的联合限制。

## 准备运行时 {#installation-and-assets}

在本版源码目录安装 pipeline 扩展，再按[权重编译步骤](./compile-weights)准备资源。校验文件、模型快照与解码器均使用准备时的路径。

```bash
python -m pip install '.[pipeline]'
python -m vflash.install_sol   # only for the single-SM89 Base16 Sol default; no model downloads
```

```bash
vflash prepare-pipeline --assets pipeline-assets.json --receipt prepared-assets.json
```

## 让首帧动起来 {#first-frame}

使用关键帧配置的校验文件，将期望的动作和声音写入 UTF-8 提示文件。示例输出五秒横屏视频。

```bash
vflash generate --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --first-frame first.png --width 928 --height 512 --duration 5 --seed 1234 \
  --gpu 0 --output video.mp4 --trust-local-code
```

## 组合外观参考 {#generate-from-the-command-line}

使用 Ref4 校验文件。第一张图对应 `<Picture 1>`，第二张对应 `<Picture 2>`。在提示中写明哪张提供主体、哪张提供环境。交付时检查文件可播放、主体正确、声音符合要求。

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

## 复用与扩展 {#generate-from-python}

连续出片可使用 [Python 会话](./python)，需要扩展混合配置时查阅[条件编码](./pipeline-conditioning)。若要保留已有镜头的原始运动，使用[源片编辑](./remake)。

## 深入使用

- <span id="check-a-request-before-allocating-hardware" aria-hidden="true"></span>[先校验硬件配置](./profiles#available)
- <span id="extended-hybrid" aria-hidden="true"></span>[混合配置与画幅](./pipeline-conditioning)
- <span id="reference-video" aria-hidden="true"></span>[参考视频输入](../reference/runtime-assets)
- <span id="audio-hidden-context" aria-hidden="true"></span>[音轨精修与上下文](./pipeline-refinement)
- <span id="veda-dense-edges" aria-hidden="true"></span>[稀疏与稠密注意力](./veda)
- <span id="experimental-pika-image-reference-runtime" aria-hidden="true"></span>[Pika 内容参考配方](./pika-reference-weights)
