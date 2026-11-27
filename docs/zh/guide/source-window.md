# 从原始素材到可交付的编辑片段

在已安装 Vflash 的 Python 环境中运行以下命令，确保可以调用 FFmpeg 和 FFprobe。示例从一段横屏 H.264 MP4 中截取三秒，放入方形模型画布，最后按内容区域导出。

## 没有现成源片时

可以先用 FFmpeg 创建一段带提示音的本地测试视频。测试图案让拉伸、补边和末帧重复容易辨认。已有素材时，直接使用自己的 H.264 文件即可。

```bash
ffmpeg -n -f lavfi -i testsrc2=size=640x360:rate=30:duration=5 \
  -f lavfi -i sine=frequency=440:duration=5 -c:v libx264 -pix_fmt yuv420p -c:a aac source.mp4
```

### 给成片换上配乐或解说

当音乐或解说从画面开头开始，且长度覆盖整段画面时，可以先将选定的音轨与原视频合成一个输入文件。下面的命令直接复制视频流。

```bash
ffmpeg -i performance.mp4 -i narration.wav -map 0:v:0 -map 1:a:0 \
  -c:v copy -c:a aac -shortest with-narration.mp4
python -m vflash.media.remake_footage prepare with-narration.mp4 spoken-window \
  --width 512 --height 512 --duration 3 --soundtrack preserve
```

画面编辑完成后，从同一个准备窗口取回配乐或解说，按原选段裁去模型补帧。

```bash
python -m vflash.media.remake_footage finish model-result.mp4 finished.mp4 \
  --receipt spoken-window/window.json --original-audio spoken-window/window.mp4
```

可以在开头和结尾各找一处清晰的字音或节拍核对同步。上例选取三秒，最终文件的画面和声音都应在三秒结束。中间的模型窗口可能更长。

## 准备三秒素材

```bash
python -m vflash.media.remake_footage prepare source.mp4 prepared \
  --width 512 --height 512 --start 1 --duration 3 --soundtrack preserve
```

新目录同时保存 `window.mp4` 和 `window.json`。重复使用同一目录时，两份原文件都会保留。导出参考帧或恢复原声时，工具使用保存的 SHA-256 核对原准备窗口，避免混用尺寸相同的其他片段。

对于宽高比为 16:9、方形像素的源片，内容区域为 512 × 288，位于画布中的 `[0,112,512,288]`。三秒对应 72 个内容帧，最短模型窗口为 90 帧，末尾会重复最后一帧 18 次。手机视频的显示旋转和非方形像素会在缩放前计入画幅选择。模型的合法长度为 90 到 345 帧中的 `5 + 17k`。

有源音轨时可选择 `--soundtrack preserve`。工具保留所选区间内的声音及其相对于画面的延迟，补帧部分使用静音。无音轨的素材请使用 `silent`。输出路径必须是新文件。

## 从窗口提取可编辑的完整参考帧

使用同次准备的视频和凭据，一次取出第 0、30、60 帧。编号从零开始，以 24 fps 的窗口起点计时。所选时刻必须位于实际内容内。

```bash
python -m vflash.media.remake_footage frames prepared/window.mp4 pins --receipt prepared/window.json --frames 0 30 60
```

`frames.json` 保留你指定的顺序。每张 PNG 都包含完整模型画布及补边。只在这些图片上修改外观，然后用记录的帧号传入 `RemakeReference(path, frame=n)`。

## 编辑和导出

使用 `prepared/window.mp4` 作为[源片编辑](./remake)的输入。把 `window.json` 中的 `model_frames`、`width` 和 `height` 填入模型请求，指定帧的参考图按完整模型画布准备。得到完整模型输出 `edited-window.mp4` 后运行：

```bash
vflash finish-remake-window edited-window.mp4 result.mp4 --receipt prepared/window.json
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames -of json result.mp4
```

示例输出是 512 × 288、24 fps、72 帧。要先核对素材转换，也可以把第一条命令产生的 `window.mp4` 直接送入导出命令。导出时重新编码内容区域，请保存模型输出和凭据，以便需要时重新导出。多个窗口的排序与接缝由调用应用管理。

## 核对并复用完整原声

窗口工具沿画面时钟解码音轨，检查声音是否连续覆盖到最后一个内容帧。编码后的静音也算有效样本。如果生成结果缺音，可以传入同次准备的窗口恢复同期声，凭据中的散列用于核对原件。

```bash
python -m vflash.media.remake_footage finish edited-window.mp4 checked.mp4 --receipt prepared/window.json
python -m vflash.media.remake_footage finish edited-window.mp4 original-sound.mp4 --receipt prepared/window.json --original-audio prepared/window.mp4
```

以前由 `vflash prepare-remake-source` 输出的凭据也可以用于导出；这些凭据只记录几何与时钟。需要在多个相似剪辑之间核对原件身份时，用本页的 `prepare` 命令创建配对目录。

## 将成片用于下一次编辑

第一步完成换人后，可以用交付的 MP4 继续换景。先读视频轨的帧率、帧数和时长。交付时转为源帧率可能改变末帧的结束时刻。

```bash
ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate,nb_frames,duration -of json checked.mp4
python -m vflash.media.remake_footage prepare checked.mp4 second-window --width 512 --height 512 --soundtrack preserve
```

这条命令从成片第一帧准备第二个窗口。成片无声时使用 `--soundtrack silent`。第二次执行使用 `second-window/window.mp4` 及其新凭据。从这里导出的参考帧属于第二个窗口。最后把原片与两次成片并排，在同一播放时刻核对转头方向、手部遮挡和商品位置。

### 按原片时间选帧

一条 15 秒分镜分窗口渲染时，可先在原片上记录人物入画、转身、收尾的时刻。若本窗口从原片 7.5 秒开始，原片的 8.75 秒对应窗口第 30 帧。0.31.1 起可以直接输入原片时刻：

```bash
python -m vflash.media.remake_footage prepare input.mp4 late-window \
  --start 7.5 --duration 3 --width 512 --height 512 --soundtrack silent
python -m vflash.media.remake_footage frames late-window/window.mp4 late-pins \
  --receipt late-window/window.json --source-times 8.75 7.5
```

导出清单按输入顺序记录窗口帧号、窗口秒数与原片秒数。可选时间包含选段起点，止于选段终点之前。服装、商品外形可使用独立外观照片；带时间的图片应展示该时刻编辑后的完整场景。
