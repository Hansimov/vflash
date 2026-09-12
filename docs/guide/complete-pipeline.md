# Generate a video

Vflash 0.3.2 generates a five-second MP4 from text alone, or from a prompt and one to three ordered reference images. Current main also provides preview official Base16 I2VA and FL2VA pipelines with one first-frame anchor or explicit first and last anchors. Use the Python API for repeated requests or the container CLI for a single generation. The complete pipeline supports T2VA Base4 and Ref2VA Turbo4 on one RTX 4090 48 GB; Ref4 also runs on two RTX 3080 20 GB GPUs. The Base16 keyframe profiles target one RTX 4090 48 GB or a cooperating pair of RTX 3080 20 GB GPUs.

On one 4090, Ref4 also accepts [a short reference video](#reference-video). The same instance can alternate images and video without switching weights.

## What runs where

Vflash owns the native denoiser, stage lifetimes, local image loading and MP4 delivery. The text and image encoders use pinned Diffusers and Transformers code; Turbo profiles additionally use PEFT adapters. Video and audio decoding use the official H3 VAE code. These components are explicit dependencies; they are not described as new native kernels. No LightX2V runtime or application server is needed.

Turbo requests use four denoising evaluations and retain their five-second contract. The preview Base16 I2VA and FL2VA profiles use 16 evaluations and accept either five or ten seconds on the native 24 fps clock. Five seconds generates 124 model frames and delivers 120; ten seconds generates 243 and delivers 240. Width and height must be multiples of 32, with at most `928 × 512` pixels and an aspect ratio between 1:4 and 4:1; larger duration-and-canvas combinations still require capacity qualification. The prompt is used verbatim. For Ref2VA, the one to three images are numbered in the order supplied: `<Picture 1>`, `<Picture 2>` and `<Picture 3>`. Describe each image’s subject and role in your prompt. These are visual references, not frame positions or guaranteed keyframes. I2VA takes one distinct frame-zero anchor, which the prompt may call `<Picture 1>`. True FL2VA takes both a first and final anchor, exposed to the prompt as `<Picture 1>` and `<Picture 2>` in temporal order.

Choose the [fixed model profile](../reference/pipeline-profiles) before preparing assets. Ref2VA uses `transformer_ref` and Ref4 v0.1; T2VA uses `transformer` and Base4 v1.0; preview I2VA and FL2VA use the official `transformer` at 16 evaluations without an adapter. The paired Base16 I2VA and FL2VA profiles for the same hardware share the exact model artifact and schedule, so either prepared keyframe pipeline can serve both request types serially without a profile restart or cold initialization. Conditioning metadata remains specific to the actual request. The result keeps the prepared identity in `profile_id` and records the actual input type in `request_mode`. Per-request stage residency still follows the selected memory strategy. Other profiles reject mode changes before execution.

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
| `adapter_path` | The selected Turbo profile's pinned BF16 adapter from [runtime assets](../reference/runtime-assets), or `null` for Base16 I2VA and FL2VA profiles |
| `decoder_directory` | The `FL2VA` directory from that same official H3 revision, containing `video_vae` and `audio_vae` |
| `artifact` | The selected profile's complete BF16 native artifact, with runtime LoRA residuals only for Turbo profiles |
| `schedule_overlay` | Matching training-Euler schedule: four evaluations for Turbo profiles or 16 for Base16 I2VA/FL2VA; Base16 uses video/audio shifts 12/3 |
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

Supply one, two or three `--reference` arguments for Ref2VA. For T2VA, prepare with `--profile t2va-turbo4-exact-sm89` and omit all image arguments. A prepared SM89 or SM86 Base16 keyframe profile accepts either exactly one `--first-frame first-frame.png` for I2VA or both `--first-frame first-frame.png --last-frame last-frame.png` for FL2VA. Add `--duration 10` for the native ten-second contract; omitting it keeps five seconds. The explicit `i2va-*` and `fl2va-*` profile IDs remain available for stable preparation and provenance. Do not mix keyframes with `--reference`. The SM86 profiles also require `--peer-gpu 1 --strategy sequence-head`. The corresponding Python FL2VA request is `VideoRequest(prompt=..., first_frame=Path("first-frame.png"), last_frame=Path("last-frame.png"), duration_seconds=10, seed=...)`. Set `audio_delivery_profile="web-v1"`, or pass `--audio-delivery-profile web-v1`, to apply bounded post-decode gain toward -18 LUFS with a -2 dBTP peak limit; the default `unchanged` path preserves the decoded waveform. This delivery option does not improve sound identity or synchronization. Progress is emitted as JSON lines on stderr; stdout contains the final result. Each command starts and closes its own models. Use one Python `H3Pipeline` instance to realize cross-mode residency across repeated requests. For a ready-made environment and complete mounting example, see [Docker generation](./docker#pipeline).

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

In 0.3.2, construction checks the profile on the CPU. The first `generate` call reads and validates its inputs before loading any models. To preload a fixed pipeline before accepting requests, call `pipeline.prepare()` explicitly. Repeated calls reuse its stages; invalid inputs preserve a healthy loaded instance. Preloading does not run conditioning or compile every input shape: first use can still incur operator preparation costs.

For two RTX 3080 20 GB GPUs, prepare the [SM86 assets](../reference/pipeline-profiles#sm86) and pass `peer_device=devices[1], strategy="sequence-head"` to `H3Pipeline`, with exactly that pair visible. Encoding and decoding use `devices[0]`; native denoising uses both. Single-SM86 and `tensor` complete pipelines are rejected before model loading. The equivalent CLI options are `--gpu 0 --peer-gpu 1 --strategy sequence-head`.

`trust_local_code=True` permits loading the official decoder Python files from the verified local snapshot. Review the [model and code licenses](../reference/license) first.

Reuse the same pipeline for sequential requests. Its native block ring leaves device memory available while the encoder and VAE take turns. CPU model masters remain owned for reuse, so sufficient host memory is also required. A progress callback runs synchronously and may raise to cancel; it must not call `close` from inside that callback. Application queues, accounts, storage and parallel worker scheduling stay outside this API.

The integration checks used a 240 GiB host-memory limit. This is a tested budget, not a measured minimum. The 64 GiB recommendation for the denoiser alone does not cover these additional encoders and decoders.

`VideoResult.elapsed_seconds` covers the successful `generate` call from input validation through reference cleanup, including a first model load. `stages.initialization_seconds` records loading during this call (zero after explicit preloading); `stages.request_elapsed_seconds` excludes only that load. `stages.session_initialization_seconds` records the session's original loading cost, also available as `pipeline.initialization_seconds`. `stages.input_preparation` includes asset checks and reference loading; its `reference_loading_seconds` is nested inside preparation. Encoding and media stages report `weight_resume_seconds`, `suspend_seconds`, and `capture_call_seconds` or `decode_call_seconds`. Their outer durations also cover synchronous callbacks and orchestration. Do not sum nested durations as independent costs.

The output path must not already exist. A video is published only after encoding, media probing and GPU stage cleanup have succeeded. A failed execution retires the pipeline and removes temporary files. `close()` releases owned models and hooks after a CUDA completion fence; it never resets another owner's CUDA context. CUDA libraries may retain process-level workspaces after a model closes. Exit the dedicated process when the application needs to relinquish its entire CUDA context.

When running in a read-only container, give Triton a writable cache directory that permits loading compiled shared libraries. A temporary filesystem mounted with `noexec` cannot serve as that cache.

## Use a reference video {#reference-video}

Install the pipeline extra or use the versioned pipeline container. Use the same prepared `ref2va-turbo4-exact-sm89` assets as for images and one RTX 4090 48 GB. This input produces a new video guided by the source; it does not provide frame-accurate editing.

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file video-prompt.txt \
  --reference-video reference.mp4 --gpu 0 --seed 1234 \
  --output variation.mp4 --trust-local-code
```

In Python, use `VideoRequest(prompt=..., reference_video=Path("reference.mp4"))`. Refer to that clip as **`<Video 1>`**, including the space. A Ref4 session can process image requests, video requests, and then images again without swapping models or LoRAs. A previously generated local MP4 can be the next request's `reference_video`; the engine adds no conversation memory or automatic prompt rewriting.

The boundary accepts one complete **2–5 second MP4, MOV or WebM**, at most **20 MiB** and **475,136 source pixels** (928×512 area). Source frame rates may be fractional and must be readable, positive and no higher than 240 fps. Only one video stream with square pixels and an unambiguous right-angle rotation is accepted. Existing source audio is discarded; it does not condition the new soundtrack. Mixed image/video inputs, audio inputs, SM86 and Ref8 video conditioning are not supported. Output remains five seconds at 24 fps, with the ordinary 32-aligned canvas limit above.

The CPU decoder takes an immutable copy, checks the full clip and resamples it to 24 fps, retaining a final partial frame interval. It passes source-sized RGB to the official video setup, which resizes once. Video sizing is independent of the output canvas and does **not** use the downscale-only image `match` policy. Admission also bounds the resulting canvas to a 1376-pixel long edge, 1376×768 total pixels and 33,024 temporal reference rows; extreme aspect ratios may therefore be rejected even when source pixels fit. Those budgets are limits, not quality guarantees across every ratio. Temporary files and decoded RGB have request-scoped lifetimes.

This is more expensive than a still image: the official VAE consumes complete temporal chunks and the text encoder samples the clip. Use the reported input, encoding, denoising and media durations to budget a deployment; no video-input latency guarantee is made. See the [typed conditioning contract](../reference/runtime-assets#video-conditioning) for the native interface and supported boundary.

## What a correctness result means

Compare conditioning and final latents to the fixed reference implementation separately from visual quality. Assess generated motion, appearance, instructions and sound against the original request. A matching latent tensor does not qualify a poor result.

Media checks cover decoded video and audio before encoding, the selected five- or ten-second delivery clock, frame count and channel layout. H.264 and AAC are lossy formats. An MP4 hash is not a numerical equivalence test for the denoiser or audio decoder.

Release checks separate all 14 conditioning tensors, final FP32 audio/video latents, decoded video and playable delivery. The official audio VAE can produce small floating-point differences between repeated requests before PCM or AAC encoding. Audio bitwise reproducibility is not promised; finite output, channel layout, frame count, clock and complete decoding are checked. See [release validation](../reference/releases) for the tested cases and scope.
