# Python integration

Use **`H3Pipeline`** when your application wants finished MP4s and should keep models loaded between
requests. Use **`NativeEngineSession`** only when it already builds compatible conditioning bundles and
wants latents ([below](#native-session)). Both own one fixed profile and device selection and process
requests **serially**. For process isolation and an HTTP queue, run the [Docker service](./docker).

## A long-lived pipeline {#pipeline}

Run the pipeline in a dedicated process, select the GPU before anything else initializes CUDA, and keep
one instance for the life of the worker:

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

`VideoRequest` takes `references=(…)` for Ref2VA, `first_frame=` / `last_frame=` for keyframes, or neither
for T2VA, within the [modes and limits](./complete-pipeline#modes-and-limits) of the prepared profile.
Keyframe profiles switch between I2VA, L2VA and FL2VA without reloading.

## Long, multi-reference and native-HD requests {#hybrid}

Fifteen-second requests, up to nine references and canvases above 1 MP (up to 2720×1536) need the original
LightX v0.1 keyframe profile with the **hybrid** model and Veda attention:

```python
from vflash.pipeline import HybridModel

pipeline = H3Pipeline(
    prepared, device=devices[0], trust_local_code=True,
    hybrid_model=HybridModel(Path("models/MiniMax-H3/transformer_ref")),
    attention_backend="veda-sm89",                       # SM89; use "veda-triton" on SM90/SM120
    veda_predictor=Path("models/veda-predictor.safetensors"),
    veda_dense_edge_layers=3,                            # 44 sparse / 6 dense layers
)
```

Install Veda with `python -m vflash.install_veda` and supply its predictor ([Veda attention](./veda)). Check a
request against the same admission rules without loading anything:

```python
from vflash.pipeline import validate_request
validate_request(VideoRequest("…", first_frame=Path("f.png"), width=2720, height=1536),
                 profile_id=prepared.profile_id, hybrid=True, attention_backend="veda-triton",
                 parallel_strategy="single", weight_residency="block-ring")
```

## Memory placement {#memory}

| Setting | Where the denoiser weights live | Use |
| --- | --- | --- |
| `weight_residency="block-ring"` (complete-pipeline default) | Host RAM, streamed by block | Leaves device memory for encoders and VAEs; needs large host RAM |
| `weight_residency="resident"` on one ≥75 GiB SM90 or ≥90 GiB SM120 | GPU | Faster warm requests; needs `PYTORCH_ALLOC_CONF=expandable_segments:True` set before Python starts |

Resident execution covers I2VA with one first frame for up to 15 s: on SM90 up to 1536×864 pixels, on SM120
within the native-HD budget `width × height × model_frames ≤ 2048² × 124` (0.15). The native single-SM89 and
3080 sessions have their own defaults; see [choose hardware](./profiles#memory).

## Optional audio refinement {#audio}

For hybrid I2VA with Veda 44/6 you can add six original-model audio evaluations after the video is complete:

```python
from vflash.native.h3_audio_refinement import AudioRefinementRequest
from vflash.native.h3_refinement_schedule import refinement_schedule

refine = AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.5), tuple(audio_tables))
result = pipeline.generate(request, Path("out.mp4"), audio_refinement=refine)
```

Video frames are unchanged. Requirements, the optional hidden-context reuse and its limits are in the
[generation guide](./complete-pipeline#optional-audio-only-six-step-refinement-source-main).

## Progress, results and errors {#lifetime}

- **Progress** events carry `stage`, `completed` and `total` for encoding, denoising, optional audio
  refinement and decoding. Callbacks run synchronously; raising from one cancels the request. Do not call
  `close()` from a callback.
- **Timing.** `result.elapsed_seconds` includes any model loading done by that call;
  `result.stages.request_elapsed_seconds` excludes it. Report cold and warm requests separately.
- **Failures.** Invalid inputs are rejected before loading and leave the pipeline usable. A failed execution
  retires the instance and removes temporary files: create a new pipeline (or a new worker process).
- **Outputs** are published only after encoding and probing succeed; the output path must not exist.
- **Close** with `with` or `close()`. It waits for the device, then releases the models owned by this instance.

## Native session (latents) {#native-session}

`NativeEngineSession` runs one native profile from compiled conditioning bundles and writes video and audio
latents. It suits applications that own conditioning and decoding.

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

For two RTX 3080 20 GB GPUs, resolve the plan with `profile_id="ref2va-turbo4-exact-sm86"`,
`peer_device=devices[1]` and `strategy="sequence-head"`; both devices serve the same request. A 4090 session
is resident by default and can pass `weight_residency="block-ring"`; SM86 always streams. The step callback
reports denoising evaluations only and adds a synchronization per step. Asset contracts are in
[runtime assets](../reference/runtime-assets); the optional [attention LoRA](../reference/attention-lora)
applies to an exclusively owned single-SM89 Base16 session.
