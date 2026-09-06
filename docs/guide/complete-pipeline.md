# Generate a video

Vflash 0.2.2 generates a five-second MP4 from text alone, or from a prompt and one to three ordered reference images. Use the Python API for repeated requests or the container CLI for a single generation. The complete pipeline supports T2VA Base4 and Ref2VA Turbo4 on one RTX 4090 48 GB; Ref4 also runs on two RTX 3080 20 GB GPUs.

## What runs where

Vflash owns the native four-step denoiser, stage lifetimes, local reference loading and MP4 delivery. The text encoder and reference encoder use pinned Diffusers, Transformers and PEFT adapters. Video and audio decoding use the official H3 VAE code. These adapters are explicit dependencies; they are not described as new native kernels. No LightX2V runtime or application server is needed.

Each request uses four denoising evaluations, a five-second result and the native 24 fps clock. Width and height must be multiples of 32, with at most `928 × 512` pixels and an aspect ratio between 1:4 and 4:1. The model generates 124 frames and delivery takes the first 120. The prompt is used verbatim. For Ref2VA, the one to three images are numbered in the order supplied: `<Picture 1>`, `<Picture 2>` and `<Picture 3>`. Describe each image’s subject and role in your prompt. These are visual references, not frame positions or guaranteed keyframes.

Choose the [fixed model profile](../reference/pipeline-profiles) before preparing assets. Ref2VA uses `transformer_ref` and Ref4 v0.1; T2VA uses `transformer` and Base4 v1.0. Each model has its own prepared assets and persistent pipeline. A request of the other mode is rejected before execution.

## Installation and assets

Install the pipeline extra from the release checkout and provide `ffmpeg` and `ffprobe` on `PATH`:

```bash
python -m pip install '.[pipeline]'
```

The extra pins the adapter implementation, including Diffusers commit `d035dcd7cc7c88e0a154609b62887d50bba9fdc2`. It does not download model weights.

Prepare a local asset configuration with six explicit paths:

| Field | Contents |
| --- | --- |
| `model_directory` | The official Diffusers components from `MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe`, with the selected `transformer_ref` or `transformer` component |
| `adapter_path` | The selected profile's pinned BF16 adapter from [runtime assets](../reference/runtime-assets) |
| `decoder_directory` | The `FL2VA` directory from that same official H3 revision, containing `video_vae` and `audio_vae` |
| `artifact` | The selected profile's complete BF16 native artifact with runtime LoRA residuals |
| `schedule_overlay` | Matching four-evaluation training-Euler schedule: video/audio shifts 12/3 for Ref4, 6/3 for T2VA |
| `auxiliary_tensor` | Its matching native input and output tensors |

The last three are prepared native assets, not arbitrary upstream checkpoint files. Follow the [official-weight compiler recipe](./compile-weights) to create them, or check the [asset contracts](../reference/runtime-assets) before supplying an existing artifact. A ready-made model package is not currently published.

Place assets in their final read-only snapshot before preparation. The ingestion step hashes all consumed files against the bundled upstream inventory or native artifact manifest. It also verifies source, LoRA and schedule identities. This is intentionally a one-time disk operation. The resulting local receipt is bound to this filesystem: model startup and requests check file identity and timestamps without rehashing model weights. Moving or changing an asset requires a new receipt.

```bash
vflash prepare-pipeline \
  --assets pipeline-assets.json --receipt prepared-assets.json
```

## Generate from the command line

Write the H3 prompt in `prompt.txt`, then list reference images in the intended order:

```bash
vflash generate \
  --prepared-assets prepared-assets.json \
  --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

Supply one, two or three `--reference` arguments for Ref2VA. For T2VA, prepare with `--profile t2va-turbo4-exact-sm89` and omit all `--reference` arguments. In Python, use `VideoRequest(prompt=..., seed=...)`. Progress is emitted as JSON lines on stderr; stdout contains the final result. Each command starts and closes its own models. For a ready-made environment and complete mounting example, see [Docker generation](./docker#pipeline).

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
            references=(Path("subject.png"), Path("setting.png")),
            seed=1234,
        ),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

The original single-image `reference=Path(...)` argument remains supported. Use it or `references=(...)`, never both.

For two RTX 3080 20 GB GPUs, prepare the [SM86 assets](../reference/pipeline-profiles#sm86) and pass `peer_device=devices[1], strategy="sequence-head"` to `H3Pipeline`, with exactly that pair visible. Encoding and decoding use `devices[0]`; native denoising uses both. Single-SM86 and `tensor` complete pipelines are rejected before model loading. The equivalent CLI options are `--gpu 0 --peer-gpu 1 --strategy sequence-head`.

`trust_local_code=True` permits loading the official decoder Python files from the verified local snapshot. Review the [model and code licenses](../reference/license) first.

Reuse the same pipeline for sequential requests. Its native block ring leaves device memory available while the encoder and VAE take turns. CPU model masters remain owned for reuse, so sufficient host memory is also required. A progress callback runs synchronously and may raise to cancel; it must not call `close` from inside that callback. Application queues, accounts, storage and parallel worker scheduling stay outside this API.

The integration checks used a 240 GiB host-memory limit. This is a tested budget, not a measured minimum. The 64 GiB recommendation for the denoiser alone does not cover these additional encoders and decoders.

`VideoResult.elapsed_seconds` covers the successful `generate` call from input validation through reference cleanup. `stages.input_preparation` includes asset checks and image loading; its `reference_loading_seconds` is nested inside that preparation duration. Model initialization is reported separately. Encoding and media stages report `weight_resume_seconds`, `suspend_seconds`, and `capture_call_seconds` or `decode_call_seconds`. Their outer durations also cover synchronous callbacks and other orchestration. These nested fields must not all be summed as independent costs.

The output path must not already exist. A video is published only after encoding, media probing and GPU stage cleanup have succeeded. A failed execution retires the pipeline and removes temporary files. `close()` releases owned models and hooks after a CUDA completion fence; it never resets another owner's CUDA context. CUDA libraries may retain process-level workspaces after a model closes. Exit the dedicated process when the application needs to relinquish its entire CUDA context.

When running in a read-only container, give Triton a writable cache directory that permits loading compiled shared libraries. A temporary filesystem mounted with `noexec` cannot serve as that cache.

## What a correctness result means

Compare conditioning and final latents to the fixed reference implementation separately from visual quality. Assess generated motion, appearance, instructions and sound against the original request. A matching latent tensor does not qualify a poor result.

Media checks cover decoded video and audio before encoding, the five-second delivery clock, frame count and channel layout. H.264 and AAC are lossy formats. An MP4 hash is not a numerical equivalence test for the denoiser or audio decoder.

Release checks separate all 14 conditioning tensors, final FP32 audio/video latents, decoded video and playable delivery. The official audio VAE can produce small floating-point differences between repeated requests before PCM or AAC encoding. Audio bitwise reproducibility is not promised; finite output, channel layout, frame count, clock and complete decoding are checked. See [release validation](../reference/releases) for the tested cases and scope.
