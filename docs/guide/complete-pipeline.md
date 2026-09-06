# Complete video pipeline preview

This development branch adds a Python pipeline from a written prompt and one reference image to an MP4. Its first target is Ref2VA Turbo4 on one SM89 GPU with 48 GiB of memory. Two sequential requests and cancellation have passed a complete GPU integration check at 928 × 512. It is not part of the published latent-only release yet. The new [official-weight compiler](./compile-weights) is available on this branch and awaits its own GPU qualification.

## What runs where

Vflash owns the native four-step denoiser, stage lifetimes, local reference loading and MP4 delivery. The text encoder and reference encoder use pinned Diffusers, Transformers and PEFT adapters. Video and audio decoding use the official H3 VAE code. These adapters are explicit dependencies; they are not described as new native kernels. No LightX2V runtime or application server is needed.

The initial request has one image, four denoising evaluations, a five-second result and the native 24 fps clock. Width and height must be multiples of 32, with at most `928 × 512` pixels and an aspect ratio between 1:4 and 4:1. The model generates 124 frames and delivery takes the first 120. The prompt is used verbatim; refer to the image as `<Picture 1>`.

## Installation and assets

Install the pipeline extra from this branch and provide `ffmpeg` and `ffprobe` on `PATH`:

```bash
python -m pip install '.[pipeline]'
```

The extra pins the adapter implementation, including Diffusers commit `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`. It does not download model weights.

Prepare a local asset configuration with six explicit paths:

| Field | Contents |
| --- | --- |
| `model_directory` | The official Diffusers component directories, including `transformer_ref`, from `MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe` |
| `adapter_path` | The pinned Ref2VA Turbo4 v0.1 BF16 adapter described in [runtime assets](../reference/runtime-assets) |
| `decoder_directory` | The `FL2VA` directory from that same official H3 revision, containing `video_vae` and `audio_vae` |
| `artifact` | A complete BF16 Ref4 native artifact with runtime LoRA residuals |
| `schedule_overlay` | Its matching four-evaluation training-Euler schedule, video shift 12 and audio shift 3 |
| `auxiliary_tensor` | Its matching native input and output tensors |

The last three are prepared native assets, not arbitrary upstream checkpoint files. Follow the [official-weight compiler recipe](./compile-weights) to create them, or check the [asset contracts](../reference/runtime-assets) before supplying an existing artifact. A ready-made model package is not currently published.

Place assets in their final read-only snapshot before preparation. The ingestion step hashes all consumed files against the bundled upstream inventory or native artifact manifest. It also verifies source, LoRA and schedule identities. This is intentionally a one-time disk operation. The resulting local receipt is bound to this filesystem: model startup and requests check file identity and timestamps without rehashing model weights. Moving or changing an asset requires a new receipt.

```python
from pathlib import Path
from vflash.pipeline import PipelineAssets, prepare_pipeline_assets

assets = PipelineAssets.from_json(Path("pipeline-assets.json"))
prepare_pipeline_assets(assets, Path("prepared-assets.json"))
```

## One owned pipeline

Run a pipeline in a dedicated process before any other code initializes CUDA. Choose the device explicitly. This example expects exactly one visible compatible GPU:

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = discover_nvidia_devices()
if len(devices) != 1:
    raise RuntimeError("make one SM89 GPU visible to this process")
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))
with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    result = pipeline.generate(
        VideoRequest(
            prompt=Path("prompt.txt").read_text(encoding="utf-8"),
            reference=Path("reference.png"),
            seed=1234,
        ),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

`trust_local_code=True` permits loading the official decoder Python files from the verified local snapshot. Review the [model and code licenses](../reference/license) first.

Reuse the same pipeline for sequential requests. Its native block ring leaves device memory available while the encoder and VAE take turns. CPU model masters remain owned for reuse, so sufficient host memory is also required. A progress callback runs synchronously and may raise to cancel; it must not call `close` from inside that callback. Application queues, accounts, storage and parallel worker scheduling stay outside this API.

The output path must not already exist. A video is published only after encoding, media probing and GPU stage cleanup have succeeded. A failed execution retires the pipeline and removes temporary files. `close()` releases owned models and hooks after a CUDA completion fence; it never resets another owner's CUDA context. CUDA libraries may retain process-level workspaces after a model closes. Exit the dedicated process when the application needs to relinquish its entire CUDA context.

When running in a read-only container, give Triton a writable cache directory that permits loading compiled shared libraries. A temporary filesystem mounted with `noexec` cannot serve as that cache.

## What a correctness result means

Compare conditioning and final latents to the fixed reference implementation separately from visual quality. Assess generated motion, appearance, instructions and sound against the original request. A matching latent tensor does not qualify a poor result.

Media checks cover decoded video and audio before encoding, the five-second delivery clock, frame count and channel layout. H.264 and AAC are lossy formats. An MP4 hash is not a numerical equivalence test for the denoiser or audio decoder.

The fixed integration check reproduced all 14 conditioning tensors and final audio/video latents exactly. Decoded video was also identical across the two requests. The official audio VAE produced small floating-point differences between the first and second request, before PCM quantization or AAC encoding. This preview therefore does not promise bitwise audio reproducibility. Both requests used the same five-second clock without audio retiming; the source of that audio variation remains under investigation.
