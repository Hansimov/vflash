# Remake source-video windows

The experimental 0.28.0a2 API compiles and executes the original **Viggle Remake** H3
`swap` and `edit` graphs against an already provisioned, caller-owned local Comfy runtime.
Use the explicit preview tag below; the stable 0.27.0 wheel does not include this interface.
Ordinary Pika and native execution profiles retain their existing interfaces.

## Runtime and model identity

Use [Viggle's original installation instructions](https://huggingface.co/Viggle/Viggle-Remake/blob/7e5be143702dd53b9154482de673b8519ed2d4d6/COMFYUI.md):

| Component | Pinned configuration |
| --- | --- |
| Remake code and models | [7e5be143](https://huggingface.co/Viggle/Viggle-Remake/tree/7e5be143702dd53b9154482de673b8519ed2d4d6) |
| ComfyUI | [355cd6b1](https://github.com/Comfy-Org/ComfyUI/tree/355cd6b14c92c19ac3188fe528e47a3f21ef5178) / 0.42.0 |
| Runtime tested with the original graphs | Python 3.12.15, Torch 2.9.1+cu130, comfy-kitchen 0.2.40 |
| Diffusion model | Original premerged INT8 `swap` or `edit`; their six-step sigma schedules stay separate |
| Conditioning and decoding | Qwen NVFP4, original video and audio VAEs |

The model directory contains the filenames declared by `remake_graph`. Supply the original
node pack, model paths, input directory and output directory to the runtime owner. Keep this
runtime's Python environment separate from the native `pipeline` extra. The adapter submits a
fixed graph; the owner controls process startup, GPU allocation, readiness and shutdown.
The model's license and Viggle attribution requirements apply separately from Vflash's source license.


Install the CPU client from its fixed preview tag and provide system `ffmpeg`/`ffprobe`:

```bash
git clone --branch v0.28.0a2 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install '.[remake-client]'
```

 No GPU framework is installed by that extra. For optional
`attention="veda-r2va"`, install [Veda-on-ComfyUI](https://github.com/veda-sparse/Veda-on-ComfyUI/tree/67e4eb5541b648555563144401ebaba832bd73ec)
and its [R2VA predictor](https://huggingface.co/Veda-Sparse/Minimax-H3-R2VA-Veda-Preview/tree/1598b407905b7c0af85aa0c13acdfa6f01ff2982).
Both generated and reference budgets use the predictor's trained policy. The default is `full`.
This reference-layout choice is explicit; the adapter does not inherit a Pika or T2VA predictor.
The tested sparse route changes pixels and must be judged against the intended scene and references.

## Prepared inputs

Prepare reference images as edited source-scene frames, following the [original Remake workflow](https://huggingface.co/Viggle/Viggle-Remake/blob/main/README.md). Portrait, product and style images supply attributes to the image-editing stage; an arbitrary portrait alone is not a prepared replacement scene. Keep the source composition and pose while applying the requested appearance, then optionally pin that edited frame to its source index. Vflash executes the resulting references and does not perform this image preparation.

`RemakeRequest` owns explicit paths and roles:

- A complete H.264 source window at 24 fps, with the declared canvas and `17n + 5` frames
  (90–345). Supply the actual source soundtrack when preserving it.
- One to four `RemakeReference` images. An unpinned appearance/style reference has the
  generation's short edge and dimensions divisible by 32. Prepare its fit or padding without stretching.
- A pinned reference is an actual target frame at its declared index and exactly matches
  the source canvas. Pin all references or leave all unpinned.
- Optional `RemakePrefix(take, start, frames)` refers to actual previously generated frames;
  its length is a multiple of 17. Source video and generated history have distinct roles.

The graph consumes one explicit source and ordered `<Picture N>` references. It does not infer
person identities, create target images, choose source cuts or split an entire film. Input bounds
are validation limits; quality and capacity remain specific to the selected task and hardware.
Original-graph GPU evidence covers SM120 96 GiB, 832×480, 124/192/226 frames, character/object/style
editing, two identities and a two-window 17-frame continuation. Multiple-window long films,
other canvases and deployment through this new transport remain open qualification scopes.

## Execute an owned window

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

Use a fresh private execution directory. The journal is persisted before the single POST, and
contains the original UUID, exact graph and outcome. A lost reply is observed by that same UUID;
the client never retries the POST. A prior execution directory is preserved for explicit recovery.
Concurrent use of one runtime object is rejected so its owner can serialize work.

Completion verifies the actual executed graph, the fixed SaveVideo output namespace and the
MP4's frame count, canvas and audio presence. Input previews cannot be mistaken for outputs.
Cancellation, timeout or uncertain execution invokes the original owner's stop operation;
a missing stop confirmation remains visible in the journal for resource reconciliation.
The HTTP endpoint is loopback-only and does not follow redirects or environment proxies.

## Deliver the complete source timeline

The returned artifact is the **model window**. The caller removes generated history and model-tail
padding, retains each real source frame once, restores any explicit canvas borders, and selects
the final soundtrack. `preserve` holds source audio as clean model conditioning, but the original
graph decodes it through the audio VAE. To retain original sound without that round trip, copy the
complete prepared source AAC during final assembly. `silent` omits audio from the raw output.

Keep the original display geometry and frame-rate metadata before preparing smaller 24 fps model
inputs. The caller can select a separate final canvas and frame rate after assembling the windows,
and retain its chosen source soundtrack. Frame-time resampling changes the output clock; the model's
motion sampling and generated spatial detail remain those of its working canvas.

This interface keeps source ownership, execution and final assembly separate, allowing a future
native renderer to consume the same explicit window/reference/history contract. The existing
HTTP server's native latent API and ordinary complete pipelines keep their separate scopes.

## Recover a durable completion

If the caller exits after the model result is saved but before its own job is updated,
`recover_completed_remake(request, input_directory=..., output_directory=...,
execution_directory=...)` reads the original local completion and history. It validates
that the request graph, input identities and saved MP4 still match. This performs no
network calls and creates no new model execution. The original server may already be stopped.
An incomplete, changed or older receipt without file identities requires explicit owner
reconciliation; it is never resubmitted automatically. Ordinary `execute_remake` continues
to require a fresh execution directory. Keep the original inputs, output and receipts together.

This recovery contract is available from 0.28.0a2. It does not change sampling or qualify
additional scene, duration, hardware or quality ranges.
