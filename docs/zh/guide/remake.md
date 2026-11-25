# 编辑源视频窗口

已有镜头提供动作与节奏时，选用 Remake。场景或商品表面修改使用 `edit`，主体替换使用 `swap`。应用负责选择对象，并准备与源片时刻对应的参考图。

录屏、手机实拍或非 24 fps 素材可直接按[准备源片与导出成片](./source-window)处理。保存准备凭据，编辑完成后用它移除模型补帧和画布补边。

## 准备执行环境

调用端安装 `remake-client` 扩展。按 [Viggle 固定版本说明](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md)单独配置本机 Comfy 环境，使用 ComfyUI 修订 `355cd6b14c92c19ac3188fe528e47a3f21ef5178`、所选模式的预合并 INT8 模型及原版 VAE，并在该环境注册 Vflash 条件节点。

```bash
git clone --branch v0.29.0 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

```python
from vflash.pipeline.remake_conditioning import NODE_CLASS_MAPPINGS
```

## 准备源片与目标帧

输入采用本地 H.264 MP4，时间基准为 24 fps、方形像素、零起点，画幅与声明一致。先校正旋转。只有该时刻完整的已编辑画面才可固定到指定模型帧；商品目录照片适合用作不绑定时刻的外观参考。需要保留音轨时应检查交付文件的实际音频流，静音使用 `silent`。


<span id="extended-canvas-preview" aria-hidden="true"></span>

## 执行与恢复

把上一页的 `window.mp4` 和 `window.json` 放入 Comfy 的输入目录。在同一画布上制作首帧的目标图 `edited-opening.png`，将六段式 H3 编辑提示词保存为当前目录的 `prompt.txt`。下面的商品编辑会从准备凭据读取模型需要的帧数、画布和音轨。

下例需要由应用提供运行时所有者，负责停止并确认进程退出。每次执行使用新的记录目录。调用方重启后，可将原请求、目录及未更改的文件传给 `recover_completed_remake` 恢复已完成结果。若结果尚不确定，先按记录中的原执行 ID 核对。

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
