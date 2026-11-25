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

## 编辑和导出

使用 `window.mp4` 作为[源片编辑](./remake)的输入。把 `window.json` 中的 `model_frames`、`width` 和 `height` 填入模型请求，指定帧的参考图按完整模型画布准备。得到完整模型输出 `edited-window.mp4` 后运行：

```bash
vflash finish-remake-window edited-window.mp4 result.mp4 --receipt window.json
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames -of json result.mp4
```

示例输出是 512 × 288、24 fps、72 帧。要先核对素材转换，也可以把第一条命令产生的 `window.mp4` 直接送入导出命令。导出时重新编码内容区域，请保存模型输出和凭据，以便需要时重新导出。多个窗口的排序与接缝由调用应用管理。
