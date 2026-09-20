# Generate a video

Vflash 0.5.0 generates a five-second MP4 from text alone, or from a prompt and one to three ordered reference images. Version 0.4.0 supports official Base16 I2VA, L2VA and FL2VA requests with one first-frame anchor, one last-frame anchor, or both anchors. Use the Python API for repeated requests or the container CLI for a single generation. The complete pipeline supports T2VA Base4 and Ref2VA Turbo4 on one RTX 4090 48 GB; Ref4 also runs on two RTX 3080 20 GB GPUs. The Base16 keyframe profiles target one RTX 4090 48 GB, an optional cooperating pair of matching RTX 4090 48 GB GPUs, or a cooperating pair of RTX 3080 20 GB GPUs.

On one 4090, Ref4 also accepts [a short reference video](#reference-video). The same instance can alternate images and video without switching weights.

## What runs where

Vflash owns the native denoiser, stage lifetimes, local image loading and MP4 delivery. The text and image encoders use pinned Diffusers and Transformers code; Turbo profiles additionally use PEFT adapters. Video and audio decoding use the official H3 VAE code. These components are explicit dependencies; they are not described as new native kernels. No LightX2V runtime or application server is needed.

Turbo requests use four denoising evaluations and retain their five-second contract. The Base16 keyframe profiles use 16 evaluations and accept integer durations from five through ten seconds on the native 24 fps clock. The corresponding model/delivery frame pairs are 5s 124/120, 6s 158/144, 7s 175/168, 8s 192/192, 9s 226/216 and 10s 243/240. Width and height must be multiples of 32, with at most 1,048,576 canvas pixels and an aspect ratio between 1:4 and 4:1. A video-reference request retains its separate 475,136-pixel output limit. These are API ceilings; a serving system must still advertise only duration-and-canvas combinations qualified on its actual hardware. The prompt is used verbatim. For Ref2VA, the one to three images are numbered in the order supplied: `<Picture 1>`, `<Picture 2>` and `<Picture 3>`. Describe each image’s subject and role in your prompt. These are visual references, not frame positions or guaranteed keyframes. I2VA takes one distinct frame-zero anchor, L2VA takes one final-frame anchor, and either may call its single image `<Picture 1>`. FL2VA takes both anchors, exposed as `<Picture 1>` and `<Picture 2>` in temporal order. Five- and ten-second I2VA/FL2VA paths have bounded hardware evidence. One ten-second, 736 × 992 SM89 L2VA case now has bounded completion, media-integrity, endpoint and latency evidence; broader L2VA quality, other canvases and intermediate durations still require target-hardware qualification before a serving system can claim delivery.

Choose the [fixed model profile](../reference/pipeline-profiles) before preparing assets. Ref2VA uses `transformer_ref` and Ref4 v0.1; T2VA uses `transformer` and Base4 v1.0; Base16 keyframe requests use the official `transformer` at 16 evaluations without an adapter. The paired Base16 I2VA and FL2VA profile identities for the same hardware share the exact model artifact and schedule, so either prepared keyframe pipeline can serve I2VA, L2VA and FL2VA serially without a profile restart or cold initialization. Conditioning metadata remains specific to the actual request. The result keeps the prepared identity in `profile_id` and records the actual input type in `request_mode`. Per-request stage residency still follows the selected memory strategy. Other profiles reject mode changes before execution.

### Full binary-megapixel budget in 0.5.0

Version 0.5.0 (from `53d6687`) raises the Base16 pipeline and native conditioning
ceiling to **1,048,576 pixels**: a 1024 × 1024 square is no longer reduced to 992 × 992.
The existing 0.4.0 tag retains its 1,032,192-pixel ceiling. Dimensions remain multiples
of 32; total area does not require either dimension to equal 1024. This does not
enlarge the separate reference-video contract.

A cooperating pair of RTX 3080 20 GB GPUs completed one five-second 1024 × 1024 I2VA
request (120 delivered frames, full decode). With the same latent, normal and
`silent-v1` delivery produced identical compressed video streams and zero-valued
decoded silent audio. This is a bounded capacity and media-contract screen, not
a semantic-quality or end-to-end speed claim. Other duration/hardware combinations
must retain their own serving qualification; API validation alone does not qualify them.

## Installation and assets

Version 0.5.0 defaults to approximate Sol on single-SM89 official Base16. Other profiles and pairs
remain dense. Use `--attention-backend torch-flash` or `H3Pipeline(..., attention_backend="torch-flash")`
for the previous dense behavior. [Selection, dependencies and quality limits](../reference/sol-engine-alignment#sol-default)
are part of this changed default, not a promise of identical media.

Install the pipeline extra from the release checkout and provide `ffmpeg` and `ffprobe` on `PATH`:

```bash
python -m pip install '.[pipeline]'
# For the single-SM89 Base16 default; needs Git and network access, no model downloads.
python -m vflash.install_sol
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

Supply one, two or three `--reference` arguments for Ref2VA. For T2VA, prepare with `--profile t2va-turbo4-exact-sm89` and omit all image arguments. A prepared SM89 or SM86 Base16 keyframe profile accepts exactly one `--first-frame first-frame.png` for I2VA, exactly one `--last-frame last-frame.png` for L2VA, or both for FL2VA. Add `--duration 10` for the native ten-second contract; omitting it keeps five seconds. The explicit `i2va-*` and `fl2va-*` profile IDs remain stable preparation and provenance identities; there is no duplicate L2VA weight profile. Do not mix keyframes with `--reference`. The SM86 profiles require `--peer-gpu 1 --strategy sequence-head`; version 0.4.0 accepts the same options for an optional two-SM89 Base16 keyframe execution. The Python forms are `VideoRequest(last_frame=Path("last-frame.png"), ...)` for L2VA and `VideoRequest(first_frame=Path("first-frame.png"), last_frame=Path("last-frame.png"), ...)` for FL2VA.

By default, delivered keyframes remain the official VAE reconstruction. Set `keyframe_delivery_profile="exact-v1"`, or pass `--keyframe-delivery-profile exact-v1`, to restore each supplied temporal endpoint after decode. The image is stretched to the requested canvas with LANCZOS, matching the official conditioning geometry; the endpoint is exact before H.264 quantization, followed by a fixed four-frame linear feather into decoded motion. The result records the applied profile and modified frame indices. This is an explicit delivery policy, not a claim that the model preserves fine detail throughout the clip.

Set `audio_delivery_profile="web-v1"`, or pass `--audio-delivery-profile web-v1`, to apply bounded post-decode gain toward -18 LUFS with a -2 dBTP peak limit; the default `unchanged` path preserves the decoded waveform. This delivery option does not improve sound identity or synchronization. Progress is emitted as JSON lines on stderr; stdout contains the final result. Each command starts and closes its own models. Use one Python `H3Pipeline` instance to realize cross-mode residency across repeated requests. For a ready-made environment and complete mounting example, see [Docker generation](./docker#pipeline).

## One owned pipeline

Source main adds `audio_delivery_profile="silent-v1"` (CLI: `--audio-delivery-profile silent-v1`)
to enforce digital silence in a stereo AAC track with the original duration and sample rate.
It skips the unused waveform decoder, not joint audio/video denoising; video latents,
video decoding and endpoint delivery stay unchanged. The engine never infers this policy
from a prompt. Applications must request it only for explicit whole-video silence, not for
unspecified audio or a request to omit music. The default remains `unchanged`.

This is a deterministic delivery contract, not improved model audio understanding or a
denoising acceleration claim. It is included in 0.5.0, not the 0.4.0 tag.

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

In 0.4.0, construction checks the profile on the CPU. The first `generate` call reads and validates its inputs before loading any models. To preload a fixed pipeline before accepting requests, call `pipeline.prepare()` explicitly. Repeated calls reuse its stages; invalid inputs preserve a healthy loaded instance. Preloading does not run conditioning or compile every input shape: first use can still incur operator preparation costs.

For two RTX 3080 20 GB GPUs, prepare the [SM86 assets](../reference/pipeline-profiles#sm86) and pass `peer_device=devices[1], strategy="sequence-head"` to `H3Pipeline`, with exactly that pair visible. Encoding and decoding use `devices[0]`; native denoising uses both. Single-SM86 and `tensor` complete pipelines are rejected before model loading. The equivalent CLI options are `--gpu 0 --peer-gpu 1 --strategy sequence-head`.

Current main also accepts those Python and CLI peer options for the SM89 Base16 keyframe profiles when exactly two matching RTX 4090 48 GB devices are visible. Only `sequence-head` is accepted; both devices use the same SM89 artifact and block-ring residency. One fixed 10-second, 736 × 992 L2VA request reduced successful `generate` time from 774.153 to 484.232 seconds and produced a byte-identical MP4, but consumed 968.464 aggregate GPU-seconds instead of 774.153. This is a bounded single-request latency result, not a throughput default: use two independent single-GPU workers when two requests are ready.

`trust_local_code=True` permits loading the official decoder Python files from the verified local snapshot. Review the [model and code licenses](../reference/license) first.

Reuse the same pipeline for sequential requests. Its native block ring leaves device memory available while the encoder and VAE take turns. CPU model masters remain owned for reuse, so sufficient host memory is also required. A progress callback runs synchronously and may raise to cancel; it must not call `close` from inside that callback. Application queues, accounts, storage and parallel worker scheduling stay outside this API.

Current main also provides opt-in, exact reuse of the seed-independent text and keyframe encodings for sequential sibling I2VA, L2VA or FL2VA candidates. A trusted scheduler creates one opaque `ConditioningReuseScope` for candidates from the same accepted request and passes it to each `generate` call as `conditioning_reuse_scope`. Never derive the scope from public prompt or media content, accept it from an end user, or reuse it for a different owner or request. Calls without a scope retain the ordinary uncached path.

The cache holds one copied CPU entry, is capped at 128 MiB, and expires after one hour of inactivity with the deadline refreshed by each hit. A scope or input change, an unscoped call, a failed capture, or `close()` clears it immediately. It contains only Qwen3-VL text outputs, token tags and clean keyframe VAE latents. Initial noise, noised reference latents, packed state, scheduler state, DiT/attention/FFN activations, decoded latents and media are never reused. Inspect `result.stages["encoding"]["capture_diagnostics"]["conditioning_reuse"]` for `hit`, `miss` or `capacity-bypass` plus encoder-call and timing counters.

On one dual-SM86 sequence-head pipeline, a warmed five-second 512 × 512 A/B/A reduced the two sequential candidates' makespan from a 352.393-second control median to 340.403 seconds (3.402%) without delaying the first candidate; both candidates' H.264 and AAC streams matched across arms. This is bounded SM86 evidence, not an SM89 or serving-system claim. When independent GPUs are available, keep one candidate per worker: serializing otherwise parallel candidates merely to obtain a cache hit loses first-result latency and fleet throughput.

Inside `H3Pipeline`, the official encoder and native stage use the same content-bound conditioning bundle contract as standalone native calls and service jobs. A validated, one-shot in-memory handoff remains available to engine integrators as an experimental primitive, but it is not the complete pipeline default: target-hardware A/B/A qualification preserved output bytes but did not improve wall latency. Do not assume that removing file I/O also removes the dominant host-memory or model-transition costs.

The integration checks used a 240 GiB host-memory limit. This is a tested budget, not a measured minimum. The 64 GiB recommendation for the denoiser alone does not cover these additional encoders and decoders.

`VideoResult.elapsed_seconds` covers the successful `generate` call from input validation through reference cleanup, including a first model load. `stages.initialization_seconds` records loading during this call (zero after explicit preloading); `stages.request_elapsed_seconds` excludes only that load. `stages.session_initialization_seconds` records the session's original loading cost, also available as `pipeline.initialization_seconds`. `stages.input_preparation` includes asset checks and reference loading; its `reference_loading_seconds` is nested inside preparation. Encoding and media stages report `weight_resume_seconds`, `suspend_seconds`, and `capture_call_seconds` or `decode_call_seconds`. Encoding also identifies `conditioning_transport` and the captured `conditioning_tensor_bytes`. Their outer durations cover synchronous callbacks and orchestration. Do not sum nested durations as independent costs.

`stages.encoding.capture_diagnostics` further separates the official conditioning call,
capture-hook waits, metadata work, and persisted-bundle finishing. The finish record separates tensor
writing from sealing/reloading, while `process_deltas` reports page faults, block I/O, and context
switches for the relevant intervals. Hook waits are nested in `official_pipeline_seconds`, and all of
these values are nested in `capture_call_seconds`; they are diagnostic attribution, not additive stage
totals. The counters describe this process only and are not host-wide resource measurements.

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

Media checks cover decoded video and audio before encoding, the selected five-through-ten-second delivery clock, frame count and channel layout. H.264 and AAC are lossy formats. An MP4 hash is not a numerical equivalence test for the denoiser or audio decoder.

Release checks separate all 14 conditioning tensors, final FP32 audio/video latents, decoded video and playable delivery. The official audio VAE can produce small floating-point differences between repeated requests before PCM or AAC encoding. Audio bitwise reproducibility is not promised; finite output, channel layout, frame count, clock and complete decoding are checked. See [release validation](../reference/releases) for the tested cases and scope.
