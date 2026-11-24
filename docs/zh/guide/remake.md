# 编辑源视频窗口

已有镜头提供动作与节奏时，选用 Remake。场景或商品表面修改使用 `edit`，主体替换使用 `swap`。应用负责选择对象，并准备与源片时刻对应的参考图。

## 准备执行环境

调用端安装 `remake-client` 扩展。按 [Viggle 固定版本说明](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md)单独配置本机 Comfy 环境，使用 ComfyUI 修订 `355cd6b14c92c19ac3188fe528e47a3f21ef5178`、所选模式的预合并 INT8 模型及原版 VAE，并在该环境注册 Vflash 条件节点。

```bash
git clone --branch v0.28.0 --depth 1 https://github.com/Hansimov/vflash.git
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

下例需要由应用提供运行时所有者，负责停止并确认进程退出。每次执行使用新的记录目录。调用方重启后，可将原请求、目录及未更改的文件传给 `recover_completed_remake` 恢复已完成结果。若结果尚不确定，先按记录中的原执行 ID 核对。

```python
from pathlib import Path
from vflash.pipeline.remake import RemakeReference, RemakeRequest
from vflash.pipeline.remake_execution import LocalRemakeRuntime, execute_remake

inputs = Path('/srv/remake/input')
request = RemakeRequest(
    source=inputs / 'source.mp4',
    prompt=Path('prompt.txt').read_text(),  # Original H3 six-section editing prompt.
    mode='swap', frames=124, width=832, height=480,
    references=(RemakeReference(inputs / 'edited-scene.png'),),
    soundtrack='preserve', seed=0,
)

# runtime_owner is the application's existing allocation/process controller.
# Its stop_and_confirm() must stop this runtime and return True after confirmation.
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
