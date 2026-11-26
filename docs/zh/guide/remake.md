# 编辑源视频窗口

已有镜头提供动作与节奏时，选用 Remake。场景或商品表面修改使用 `edit`，主体替换使用 `swap`。应用负责选择对象，并准备与源片时刻对应的参考图。

录屏、手机实拍或非 24 fps 素材可直接按[准备源片与导出成片](./source-window)处理。保存准备凭据，编辑完成后用它移除模型补帧和画布补边。

## 准备执行环境

调用端安装 `remake-client` 扩展。按 [Viggle 固定版本说明](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md)单独配置本机 Comfy 环境，使用 ComfyUI 修订 `355cd6b14c92c19ac3188fe528e47a3f21ef5178`、所选模式的预合并 INT8 模型及原版 VAE，并在该环境注册 Vflash 条件节点。

```bash
git clone --branch v0.31.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

```python
from vflash.pipeline.remake_conditioning import NODE_CLASS_MAPPINGS
```

## 准备源片与目标帧

输入采用本地 H.264 MP4，时间基准为 24 fps、方形像素、零起点，画幅与声明一致。先校正旋转。只有该时刻完整的已编辑画面才可固定到指定模型帧；商品目录照片适合用作不绑定时刻的外观参考。需要保留音轨时应检查交付文件的实际音频流，静音使用 `silent`。


准备独立的人像或商品照片时，使用与视频相同的准备凭据。工具按照片的显示方向等比缩放，输出网格对齐的 PNG；同名输出会保留。

```bash
python -m pip install "Pillow>=13,<14"
vflash prepare-remake-appearance portrait.jpg portrait-view --receipt prepared/window.json --box 250 40 740 650
```

传入 `RemakeReference(Path("portrait-view/appearance.png"))` 可仅提供外观；对齐特定视频时刻的完整编辑画面仍使用 `frame=n`。0.30.0 支持在同一窗口分别绑定完整场景帧与独立外观图。

### 为人像或场地选择取景范围 {#reference-view}

先按实际照片选择左、上、右、下边界，坐标在方向校正后的 0 到 1000 网格上。省略 `--box` 会使用全图。目标目录必须是新目录。将 `portrait-view/view.png` 交给图像编辑器，将 `portrait-view/appearance.png` 作为同一视频窗口的不定时外观参考。两者保留同一块原始像素。需要保留原片衣服时，框住脸和头发；选取场地时，保留足够的空间特征并查看边缘的旁人。

### 同一窗口使用场景帧与外观照片

例如为同一段商品介绍同时换人和换景，可以用第一个画面确定机位和场地，第二张照片单独提供人物外观。提示词中的图片编号按元组顺序排列。每个窗口最多四张图片。

```python
from pathlib import Path
from vflash.pipeline.remake import RemakeReference

references = (
    RemakeReference(Path("edited-opening.png"), frame=0),
    RemakeReference(Path("portrait-view/appearance.png")),
)
```

把这个元组传给下文的 `RemakeRequest(references=references, ...)`。每个指定帧号的画面都要与准备好的窗口尺寸完全一致，帧号从窗口第一帧的 0 开始。单独的人像或商品图保持自身构图。在场地照片中，现场人物也可能影响生成。若只需要布景，优先用空场照片。逐帧检查原片的动作、边缘人物及商品形状。

可选参数 `keyframe_until` 是 0 到 1 之间的 sigma 阈值，默认 0.9。较低阈值让关键帧参与更多去噪步骤。它不会指定关键帧在视频中持续多少秒。


## 在换景中保住手中物件和短时图标

换背景时，先找一帧手和商品挡住旧背景的画面，在完整模型画布上编辑，再检查指尖、手中物件及接触点。对于短时贴纸，可以分别查看出现前、仍可见的后段和消失后的源画面，每个窗口最多选取四张有用的参考图。成片要同时核对出现和消失的时刻。

先用[源帧导出工具](./source-window)得到 `pins/frames.json`。把改好的 PNG 按原文件名存入 `edited/`。清单记录了真实帧号和原始顺序，下面生成的元组可直接传给后续请求的 `references`。

```python
import json
from pathlib import Path
from vflash.pipeline.remake import RemakeReference

pins = Path("pins")
edited = Path("edited")
references = tuple(
    RemakeReference(edited / row["file"], frame=row["frame"])
    for row in json.loads((pins / "frames.json").read_text())
)
```

原版 Remake 的一次请求支持两种方式：每张参考图都绑定帧号，或整组参考图都不绑定。只有商品照片时可用未绑定的外观参考；需要绑定时，先把该外观做到对应时刻的完整画面里。[原作者的提示说明](https://huggingface.co/Viggle/Viggle-Remake/blob/main/PROMPTING.md)解释了参考图、帧号与六段提示词如何配合。


<span id="extended-canvas-preview" aria-hidden="true"></span>

## 执行与恢复

把上一页的 `prepared/window.mp4` 和 `prepared/window.json` 放入 Comfy 的输入目录。在同一画布上制作首帧的目标图 `edited-opening.png`，将六段式 H3 编辑提示词保存为当前目录的 `prompt.txt`。下面的商品编辑会从准备凭据读取模型需要的帧数、画布和音轨。

下例需要由应用提供运行时所有者，负责停止并确认进程退出。每次执行使用新的记录目录。调用方重启后，可将原请求、目录及未更改的文件传给 `recover_completed_remake` 恢复已完成结果。若结果尚不确定，先按记录中的原执行 ID 核对。

使用上一页导出的 `frame-000.png` 制作开场目标图。需要固定另一时刻时，请把该张图与清单里的零基帧号配对。

```python
import json
from pathlib import Path
from vflash.pipeline.remake import RemakeReference, RemakeRequest
from vflash.pipeline.remake_execution import LocalRemakeRuntime, execute_remake

inputs = Path('/srv/remake/input')
window = json.loads((inputs / 'window.json').read_text())
request = RemakeRequest(
    source=inputs / 'window.mp4',
    prompt=Path('prompt.txt').read_text(),  # H3 六段式编辑提示词
    mode='edit', frames=window['model_frames'],
    width=window['width'], height=window['height'],
    references=(RemakeReference(inputs / 'edited-opening.png', frame=0),),
    soundtrack=window['soundtrack'], seed=0,
)

# runtime_owner 是应用已有的进程控制器
# stop_and_confirm() 结束本次运行，并在确认退出后返回 True
result = execute_remake(
    request,
    input_directory=inputs,
    output_directory=Path('/srv/remake/output'),
    execution_directory=Path('/srv/remake/executions/window-001'),
    runtime=LocalRemakeRuntime('http://127.0.0.1:8188'),
    stop_runtime=runtime_owner.stop_and_confirm,
    timeout_seconds=600,
    cancel_check=runtime_owner.cancelled,
)
print(result['artifact'], result['media'])
```

用返回的 `artifact` 路径作为 [导出步骤](./source-window) 的第一个参数。若仍比源片多出若干帧，核对传入的是同一个窗口的 `window.json`，并对比导出文件的帧数和其中的 `content_frames`。
