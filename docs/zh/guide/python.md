# 复用 Python 推理会话

进程启动时加载一次已准备的资源，对同一组 GPU 逐条提交请求，最后关闭会话。这样连续生成多个镜头时可以复用主干权重。

## 生成完整 MP4 {#pipeline}

在 worker 进程内选择实际可见的设备。模型族由校验文件决定，参考图顺序需与提示中的编号一致，每个结果写入新路径。

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = {d.index: d for d in discover_nvidia_devices()}   # indices as shown by `vflash doctor`
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))

with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    pipeline.prepare()                                  # load before accepting work
    for name, prompt, first in jobs:                    # your queue
        result = pipeline.generate(
            VideoRequest(prompt=prompt, first_frame=first, width=928, height=512,
                         duration_seconds=5, seed=1234),
            Path(f"out/{name}.mp4"),
            progress=lambda e: print(e.stage, e.completed, e.total),
        )
        print(result.output_path, result.elapsed_seconds)
```

## 显存放置 {#memory}

块环模式从主机内存流式加载，常驻模式则将主干留在显卡上，还需给编码与解码预留空间。比较重复请求耗时时，固定源素材、画幅和配方。实测设备参见[硬件要求](./profiles)。

## 直接使用原生潜变量 {#native-session}

自备编码器和 VAE 的应用可以使用 NativeEngineSession。会话读取编译后的条件包、主干、采样时序和辅助张量，四者须相互匹配。需要对接多个客户端时，[HTTP 服务](../reference/http-api)提供相同类型的输出。

```python
from vflash.catalog import ProfileCatalog
from vflash.native.runner import NativeEngineSession
from vflash.planner import resolve_plan

plan = resolve_plan(ProfileCatalog.bundled(), profile_id="ref2va-turbo4-exact-sm89", device=devices[0])
with NativeEngineSession(plan, artifact=Path("artifact"), schedule_overlay=Path("schedule"),
                         auxiliary_tensor=Path("auxiliary.safetensors")) as session:
    result = session.generate(Path("bundles/a"), Path("out/a.safetensors"),
                              progress_callback=lambda done, total: print(done, total))
```
