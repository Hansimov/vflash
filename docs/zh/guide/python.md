# Python 集成

在 SM89 上用纯文字或参考图生成完整视频，可以使用 [`H3Pipeline`](./complete-pipeline)。应用已经能准备兼容条件包，并希望在多个请求间复用权重时，可以直接使用 `NativeEngineSession`。一个会话持有固定的模型配置和显卡组，**串行**处理请求。

如果需要进程隔离和 HTTP 队列，可使用 [Docker 服务](./docker)。

## 创建会话 {#session}

在[支持的环境](./getting-started#run-a-bundle)中安装 `.[gpu]`，准备好[四类运行输入](../reference/runtime-assets)，再替换以下路径。必须在进程初始化 CUDA 之前选定显卡。

```python
from pathlib import Path

from vflash.catalog import ProfileCatalog
from vflash.hardware import discover_nvidia_devices
from vflash.native.runner import NativeEngineSession
from vflash.planner import resolve_plan

# 使用 `vflash doctor` 显示的物理 GPU 索引。
devices = {device.index: device for device in discover_nvidia_devices()}
plan = resolve_plan(
    ProfileCatalog.bundled(),
    profile_id="ref2va-turbo4-exact-sm89",
    device=devices[0],
)
outputs = Path("./outputs")
outputs.mkdir(parents=True, exist_ok=True)

with NativeEngineSession(
    plan,
    artifact=Path("/path/to/artifact"),
    schedule_overlay=Path("/path/to/schedule"),
    auxiliary_tensor=Path("/path/to/auxiliary.safetensors"),
) as session:
    for name in ("example-a", "example-b"):
        result = session.generate(
            Path("/path/to/bundles") / name,
            outputs / f"{name}.safetensors",
        )
        print(result["session"])
```

输出文件包含供解码器使用的视频和音频 latent 张量，尚不是 MP4。返回值的 `session` 字段区分[初始化与请求耗时](../reference/performance#timing)。

## 选择双 3080 {#parallel}

将上面的执行计划替换为以下内容，并使用匹配的 SM86 资源：

```python
plan = resolve_plan(
    ProfileCatalog.bundled(),
    profile_id="ref2va-turbo4-exact-sm86",
    device=devices[0],
    peer_device=devices[1],
    strategy="sequence-head",
)
```

两张卡属于同一个请求。也可选择 `tensor` 策略；不指定第二张卡时则为单卡执行。增加并发 worker 前，请先核对[硬件与内存要求](./profiles#memory)。

## 获取已完成步数 {#progress}

应用需要去噪进度时，可以传入回调：

```python
def on_step(completed: int, total: int) -> None:
    print(f"Denoising: {completed}/{total}")

# 在会话仍然打开时调用：
result = session.generate(
    Path("/path/to/bundles/example-a"),
    Path("./outputs/example-a.safetensors"),
    progress_callback=on_step,
)
```

所选显卡都完成本次评估后，回调才会执行。这是**去噪步数进度**，不是完整视频生成过程的百分比。回调应保持简短；启用后每一步增加设备同步，不需要通知时可以省略。

## 为中间结果留出显存 {#memory}

4090 会话默认让权重常驻显存。构造时传入 `weight_residency="block-ring"`，可以改为从系统内存分块加载。3080 配置始终分块加载，不支持完整 BF16 权重常驻。

请使用自己的输入，同时测量内存和延迟。分块加载可以减少权重的显存占用，但需要较多系统内存。所选策略在一个会话内保持不变。

## 关闭会话 {#lifetime}

使用上面的 `with`，或在所有者停止时调用 `session.close()`。关闭过程等待选中的显卡完成工作，再关闭通信资源、释放权重。即使继续保存已关闭的 Python 对象，权重也会释放。

CUDA 上下文和分配器缓存仍由进程持有，随进程退出释放。已关闭的会话拒绝新请求。推理失败后应关闭会话，不要尝试继续执行；如果显卡完成状态或清理操作失败，应停止对应 worker 进程，不再复用其 CUDA 上下文。

不同显卡组应通过 `spawn` 启动独立子进程管理。不要并发调用同一个会话，也不要在选卡前初始化 CUDA。模型和 LoRA 在会话内固定；切换时需要匹配的新资源和新会话。
