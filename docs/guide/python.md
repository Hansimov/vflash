# Reuse a Python inference session

Load the prepared assets once when a process starts. Submit one request at a time to its GPU allocation and close the session after its final job. This avoids repeatedly loading the transformer between clips.

## A complete MP4 {#pipeline}

Use the device discovered inside the worker process. The receipt determines the model family; supply references in exactly the order named in the prompt. Choose a new output path for each result.

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

## Memory placement {#memory}

Block-ring streams blocks from host RAM. Resident placement keeps the trunk on the GPU and needs enough headroom for conditioning and decoding. Compare elapsed warm requests with the same source, canvas and recipe. The [hardware guide](./profiles) lists tested allocations.

## Use native latents {#native-session}

Applications with their own encoders and VAEs can use NativeEngineSession. It consumes a compiled bundle, artifact, schedule and auxiliary tensors. Keep those four assets matched for the entire session. For multiple clients, the [HTTP service](../reference/http-api) exposes the same result type.

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
