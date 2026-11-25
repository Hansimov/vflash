# 从原始素材到可交付的编辑片段

以下步骤使用 0.29.0 的基础 Python 包和系统 FFmpeg、FFprobe。时间从源视频画面的起点算起。以一段超过四秒的 H.264 MP4 为例，截取其中三秒，在方形模型画布内保留原构图，最后导出没有黑边或补帧的画面。

## 没有现成源片时

可以先用 FFmpeg 创建一段带提示音的本地测试视频。测试图案让拉伸、补边和末帧重复容易辨认。已有素材时，直接使用自己的 H.264 文件即可。

```bash
ffmpeg -n -f lavfi -i testsrc2=size=640x360:rate=30:duration=5 \
  -f lavfi -i sine=frequency=440:duration=5 -c:v libx264 -pix_fmt yuv420p -c:a aac source.mp4
```

## 准备三秒素材

```bash
vflash prepare-remake-source source.mp4 window.mp4 \
  --width 512 --height 512 --start 1 --duration 3 --soundtrack silent > window.json
```

对于宽高比为 16:9、方形像素的源片，内容区域为 512 × 288，位于画布中的 `[0,112,512,288]`。三秒对应 72 个内容帧，最短模型窗口为 90 帧，末尾会重复最后一帧 18 次。手机视频的显示旋转和非方形像素会在缩放前计入画幅选择。模型的合法长度为 90 到 345 帧中的 `5 + 17k`。

有源音轨时可选择 `--soundtrack preserve`。工具保留所选区间内的声音及其相对于画面的延迟，补帧部分使用静音。无音轨的素材请使用 `silent`。输出路径必须是新文件。

## 从窗口提取可编辑的完整参考帧

下载与 0.29.0 wheel 配套的独立工具。下面从同一窗口一次取出第 0、30、60 帧。编号从零开始，以 24 fps 的窗口起点计时。所选时刻必须位于实际内容内。

```bash
curl -fL https://github.com/Hansimov/vflash/releases/download/v0.29.0/remake_footage.py -o remake_footage.py
python remake_footage.py frames window.mp4 pins --receipt window.json --frames 0 30 60
```

`frames.json` 保留你指定的顺序。每张 PNG 都包含完整模型画布及补边。只在这些图片上修改外观，然后用记录的帧号传入 `RemakeReference(path, frame=n)`。

## 编辑和导出

使用 `window.mp4` 作为[源片编辑](./remake)的输入。把 `window.json` 中的 `model_frames`、`width` 和 `height` 填入模型请求，指定帧的参考图按完整模型画布准备。得到完整模型输出 `edited-window.mp4` 后运行：

```bash
vflash finish-remake-window edited-window.mp4 result.mp4 --receipt window.json
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames -of json result.mp4
```

示例输出是 512 × 288、24 fps、72 帧。要先核对素材转换，也可以把第一条命令产生的 `window.mp4` 直接送入导出命令。导出时重新编码内容区域，请保存模型输出和凭据，以便需要时重新导出。多个窗口的排序与接缝由调用应用管理。

## 核对并复用完整原声

如果要保留声音，第一步须使用 `--soundtrack preserve`。0.29.0 wheel 的导出命令检查音轨是否存在，音轨也可能在画面结束前中断。配套工具还会核对音轨的起止时间。若需要使用源片同期声，传入**同一次准备**产生的窗口，工具将其音轨与编辑后的画面一起导出。

```bash
python remake_footage.py finish edited-window.mp4 checked.mp4 --receipt window.json
python remake_footage.py finish edited-window.mp4 original-sound.mp4 --receipt window.json --original-audio window.mp4
```
