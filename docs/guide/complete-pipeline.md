# Generate a video {#generate-a-video}

`H3Pipeline` (Python) and `vflash generate` (CLI) produce a complete MP4 from a prompt and optional
images or a short reference video. Vflash owns the native denoiser, stage lifetimes, local media loading
and MP4 delivery; the text/image encoders use pinned Diffusers and Transformers code and the official
H3 VAEs decode video and audio. No LightX2V runtime or application server is required.

This page describes the current release. Per-version changes and their exact evidence are in the
[release notes](../reference/releases); hardware capacity is in [profiles and hardware](./profiles).

## Modes and limits {#modes-and-limits}

| Mode | Inputs | Prompt labels | Profiles |
| --- | --- | --- | --- |
| T2VA | prompt only | — | `t2va-turbo4-exact-sm89` |
| Ref2VA | 1–3 ordered images (1–9 with the extended hybrid) | `<Picture 1>` … in input order | Ref4 (`ref2va-turbo4-exact-*`) |
| I2VA | one first-frame anchor | `<Picture 1>` | Base16 or LightX v0.1 keyframe profiles |
| L2VA | one last-frame anchor | `<Picture 1>` | the same keyframe profiles |
| FL2VA | first and last anchors | `<Picture 1>`, `<Picture 2>` in time order | the same keyframe profiles |
| Reference video | one 2–5 s clip | `<Video 1>` | Ref4 on one RTX 4090 48 GB |

Reference images are visual references, not timeline positions. The keyframe profiles serve I2VA,
L2VA and FL2VA serially from one prepared pipeline without a restart; `result.request_mode` records
the actual input type and `profile_id` the prepared identity. Other profiles reject mode changes before
execution. The prompt is used verbatim; describe each image's subject and role.

**Canvas.** Width and height are multiples of 32 with an aspect ratio between 1:4 and 4:1. Ordinary
requests allow up to 1,048,576 pixels. Image-conditioned native HD allows up to 2048² pixels with a
long edge of at most 2720, for example 2720×1536 (16:9), 1536×2720 (9:16), 2368×1760 (4:3) or 2048².
A reference-video request keeps its own 475,136-pixel output limit.

**Duration.** Turbo T2VA and Ref4 are five seconds. Base16 keyframe profiles accept integer durations
of 5–10 seconds. The [extended hybrid](#extended-hybrid) accepts 5–15 seconds. Delivery is 24 fps:
for example 5 s uses 124 model / 120 delivered frames and 15 s uses 362 / 360.

**Joint budget.** Canvas and duration are not independent: `width × height × model_frames` must not exceed
`2048² × 124`, so a 4 MP canvas is a five-second request and 2048² at fifteen seconds is rejected.
Multiple-image Ref above three images is limited to five seconds and 960×544. These are software
ceilings; a serving system must advertise only combinations qualified on its actual hardware.

### Check a request before allocating hardware {#check-a-request-before-allocating-hardware}

`validate_request` runs the same admission checks as `H3Pipeline.generate` without importing Torch,
loading weights, opening media or initializing CUDA:

```python
from pathlib import Path
from vflash.pipeline import VideoRequest, validate_request

request = VideoRequest("Describe the motion.", first_frame=Path("first.png"),
                       width=2720, height=1536, duration_seconds=5)
validate_request(request, profile_id="i2va-turbo4-v01-544-exact-sm89",
                 hybrid=True, attention_backend="veda-triton",
                 parallel_strategy="single", weight_residency="block-ring")
```

Passing admission establishes software eligibility, not GPU memory, asset integrity, hardware
qualification or output quality.

## Install and prepare assets {#installation-and-assets}

Install the pipeline extra from a release checkout and provide `ffmpeg` and `ffprobe` on `PATH`:

```bash
python -m pip install '.[pipeline]'
python -m vflash.install_sol   # only for the single-SM89 Base16 Sol default; no model downloads
```

The extra pins the adapter implementation, including Diffusers commit
`d035dcd7cc7c88e0a154609b62887d50bba9fdc2`. Choose the [fixed model profile](../reference/pipeline-profiles)
first, then describe its assets with six explicit paths:

| Field | Contents |
| --- | --- |
| `model_directory` | Official Diffusers components from `MiniMaxAI/MiniMax-H3` at `42ed227ee7df40d41602854ae760620d6eb651fe`, with `transformer_ref` (Ref) or `transformer` (T2VA, keyframes) |
| `adapter_path` | The Turbo profile's pinned BF16 adapter from [runtime assets](../reference/runtime-assets), or `null` for Base16 |
| `decoder_directory` | The `FL2VA` directory of the same revision, with `video_vae` and `audio_vae` |
| `artifact` | The profile's complete BF16 native artifact |
| `schedule_overlay` | Matching training-Euler schedule (4 evaluations for Turbo, 16 for Base16 with shifts 12/3) |
| `auxiliary_tensor` | Its matching native input and output tensors |

Create the last three with the [official-weight compiler](./compile-weights) or check the
[asset contracts](../reference/runtime-assets). No ready-made model package is published. Place assets in
their final read-only location, then prepare once:

```bash
vflash prepare-pipeline --assets pipeline-assets.json --receipt prepared-assets.json
```

Preparation hashes every consumed file against the upstream inventory or artifact manifest and binds a
receipt to this filesystem; later startups check identity and timestamps without rehashing. Moving or
changing an asset requires a new receipt.

## Generate from the command line {#generate-from-the-command-line}

```bash
vflash generate \
  --prepared-assets prepared-assets.json --prompt-file prompt.txt \
  --reference subject.png --reference setting.png \
  --width 928 --height 512 --seed 1234 --gpu 0 \
  --output video.mp4 --trust-local-code
```

- Ref2VA: one to three `--reference` arguments, in prompt order.
- T2VA: prepare `t2va-turbo4-exact-sm89` and omit image arguments.
- Keyframes: `--first-frame` (I2VA), `--last-frame` (L2VA) or both (FL2VA); add `--duration N`.
  Do not mix keyframes with `--reference`.
- Paired GPUs: `--gpu 0 --peer-gpu 1 --strategy sequence-head` (required for SM86 pairs; optional for
  two matching RTX 4090 48 GB Base16 keyframe GPUs).
- Explicit Veda edge width: `--veda-dense-edge-layers 3` (see [attention](#veda-dense-edges)).

Progress is emitted as JSON lines on stderr and the result on stdout. Each command loads and closes its
own models; use one Python pipeline to keep models resident across requests. For a container, see
[Docker generation](./docker#pipeline).

## Generate from Python {#generate-from-python}

Run the pipeline in a dedicated process before anything else initializes CUDA, and choose devices
explicitly:

```python
from pathlib import Path
from vflash.hardware import discover_nvidia_devices
from vflash.pipeline import H3Pipeline, VideoRequest, load_prepared_pipeline_assets

devices = discover_nvidia_devices()
prepared = load_prepared_pipeline_assets(Path("prepared-assets.json"))
with H3Pipeline(prepared, device=devices[0], trust_local_code=True) as pipeline:
    pipeline.prepare()                       # optional preload before accepting requests
    result = pipeline.generate(
        VideoRequest(prompt=Path("prompt.txt").read_text(encoding="utf-8"),
                     references=(Path("subject.png"), Path("setting.png")), seed=1234),
        Path("video.mp4"),
        progress=lambda event: print(event.stage, event.completed, event.total),
    )
    print(result.output_path, result.elapsed_seconds)
```

Keyframe requests use `first_frame=`, `last_frame=` or both. For a cooperating pair pass
`peer_device=devices[1], strategy="sequence-head"`; encoding and decoding stay on the first device.
Two RTX 4090s on one request reduce latency but consume more GPU-seconds: when two requests are ready,
prefer two independent single-GPU workers.

**Lifetimes.** Construction checks the profile on the CPU; the first `generate` validates inputs before
loading models. Invalid inputs leave a healthy loaded instance; a failed execution retires the pipeline and
removes temporary files. The output path must not exist, and a video is published only after encoding,
probing and stage cleanup succeed. `close()` releases owned models after a CUDA completion fence and never
resets another owner's context. The native block ring leaves device memory for the encoders and VAEs,
while CPU model masters stay resident, so host memory must cover them. Progress callbacks run synchronously
and may raise to cancel, but must not call `close`. `trust_local_code=True` loads the official decoder files
from the verified local snapshot; review the [licenses](../reference/license) first.

In a read-only container give Triton a writable, executable cache directory; a `noexec` temporary
filesystem cannot serve as that cache.

## Delivery options {#delivery-options}

| Option | Effect |
| --- | --- |
| `keyframe_delivery_profile="exact-v1"` | Restores each supplied endpoint after decode (LANCZOS to the canvas, four-frame feather). Default keeps the VAE reconstruction. |
| `audio_delivery_profile="web-v1"` | Bounded gain toward −18 LUFS with a −2 dBTP limit. Default `unchanged` keeps the decoded waveform. |
| `audio_delivery_profile="silent-v1"` | Digital-silence stereo track of the original length; skips only the waveform decoder. Use only for explicit whole-video silence. |
| `generate_audio=False` | Video-only generation: removes audio rows before denoising and writes no audio stream. Can change the video for the same seed. |
| `generate_rgb(request, Path("x.npy"))` | Video-only uint8 RGB array `[model_frames, H, W, 3]`; `required_frames=` decodes only the VAE chunks those frames need. |

The CLI accepts `--keyframe-delivery-profile` and `--audio-delivery-profile`. `generate_audio=False` and
RGB delivery are Python-only and require the single-device path. The complete pipeline also applies linked
stereo headroom before the official audio decoder clamp; the reusable context is
`vflash.media.audio_headroom.audio_decoder_headroom`, a sample-peak guard that never boosts quiet audio.
None of these options improves sound identity or synchronization.

## Use a reference video {#reference-video}

```bash
vflash generate --prepared-assets prepared-assets.json --prompt-file video-prompt.txt \
  --reference-video reference.mp4 --gpu 0 --seed 1234 --output variation.mp4 --trust-local-code
```

In Python use `VideoRequest(prompt=..., reference_video=Path("reference.mp4"))` and refer to the clip as
**`<Video 1>`**. One Ref4 session alternates image and video requests without swapping weights. The input is
one complete 2–5 s MP4, MOV or WebM, at most 20 MiB and 475,136 source pixels, with one square-pixel video
stream at no more than 240 fps. Source audio is discarded. The official setup normalizes the canvas to a
1376-pixel long edge, 1376×768 pixels and 33,024 temporal reference rows, so extreme ratios may be rejected.
Output is five seconds; this guides a new video and is not frame-accurate editing. See the
[conditioning contract](../reference/runtime-assets#video-conditioning).

## Hybrid model and extended requests {#extended-hybrid}

`HybridModel(Path(".../transformer_ref"))` keeps the LightX v0.1 four-evaluation backbone and replaces block
25–49 AdaLN projections with the pinned official Ref component (about 55 MiB of tables, no second denoiser).
Keyframes use the ordinary FL graph and Ref uses a real official Ref graph sharing the prefix, text encoder
and VAEs. It runs on one SM89, SM90 or SM120 GPU (SM103 preview) with dense or explicit Veda attention;
the measured envelope of each GPU is in [choose hardware](./profiles#heterogeneous).

With Veda, the single-SM89 hybrid accepts **5–15 s** image-conditioned requests and up to **nine** ordered
references (`<Picture 1>` … `<Picture 9>`), within the [joint budget](#modes-and-limits). Standard
Ref4/Base16, SM86, paired execution and reference video do not inherit this expansion.

## Text encoder prefix {#exact-text-encoder-prefix}

Conditioning keeps 51 of the 64 Qwen3-VL decoder layers: the pinned H3 adapter consumes `hidden_states[50]`,
so the raw intermediate state is preserved while the unused tail is removed before offloading hooks are
installed. Weights, vision encoder and BF16 arithmetic are unchanged; the full checkpoint still loads first,
so this is not a smaller download or a guaranteed cold-start gain.

## Attention {#veda-dense-edges}

Single-SM89 official Base16 defaults to approximate Sol; other profiles and pairs stay dense. Select dense
explicitly with `attention_backend="torch-flash"`. [Veda](./veda) is an explicit sparse backend with five
dense layers at each edge by default (40 sparse / 10 dense); `veda_dense_edge_layers=3` selects 44 sparse /
6 dense. Each session owns its partition and reports the layers it ran. These are approximate inference
choices, not equivalence claims; Sol's [selection and limits](../reference/sol-engine-alignment#sol-default)
are documented separately.

## Optional audio refinement {#optional-audio-only-six-step-refinement-source-main}

`AudioRefinementRequest` adds six original-model audio evaluations after a four-step hybrid I2VA generation.
It freezes the completed video, keeps every temporal position and uniformly samples spatial context up to
500,000 pixels with the original rotary coordinates. Video pixels, duration and delivery clock are unchanged.

```python
from vflash.native.h3_audio_refinement import AudioRefinementRequest
from vflash.native.h3_refinement_schedule import refinement_schedule

refinement = AudioRefinementRequest(refinement_schedule(steps=6, denoise=0.5), tuple(audio_tables))
result = pipeline.generate(request, output_path, audio_refinement=refinement)
```

It requires hybrid weights, one device, Veda 44/6, one same-canvas first image and generated audio, with
either resident weights and `veda-triton` or the native two-slot BF16 `block-ring` and `veda-sm89`. Silent
delivery, video-only output, other modes and simultaneous repaint are rejected. `context_pixels=None`
selects full-context six-step control. Progress reports 4 + 6 evaluations and `audio_refinement` records the
added work; failure restores bindings, attention, video and original audio. There is no CLI or HTTP switch
and no default activation.

### Hidden-context reuse {#audio-hidden-context}

`reuse_hidden_context=True` reuses first-evaluation BF16 non-audio hidden states; all six audio evaluations
remain, later steps rebuild K/V and process only audio rows. This is an explicit approximation, never changes
video latents and defaults to false. The cache is scoped to one refinement (at most 40 GiB, with at least
12 GiB of device memory kept free) and released on success or failure.

## Face repaint building blocks {#explicit-native-face-repaint}

These are opt-in engine APIs. Detection, tracking, crop/keyframe alignment, restoration models, source-audio
preservation and compositing belong to the application.

- `VideoRepaint` / `VideoRepaintRequest`: low-noise repaint with compiled FL/Ref modulation at the actual
  sigmas, preserved conditioned rows and frozen audio. Compile tables with
  `compile_refinement_tables(..., steps=2 or 4, denoise=0.035, mode="face")`. The native request runs on a
  preloaded `NativeEngineSession` with `generate_audio=False`; single device only.
- `vflash.media.face_encoder.face_encoder(vae, profile)`: scoped crop encoding for square 320/384/512 crops
  of 120–362 model frames. `batch2-fp16-norm` and `batch2-fp16-norm-reuse` are approximate FP16 batches
  (the latter reuses exactly equal clips); `batch2-fp16-norm-reuse-untiled` removes spatial tiling for
  512-square, 124-frame crops at higher memory. `original` keeps unchanged encoding.
- `H3Pipeline.source_scene_conditioning(source)` (development candidate): reuses the source scene's Qwen
  features for related video-only repaints. This deliberately changes conditioning.

The workflow was informed by the MIT-licensed [ComfyUI H3 FaceRefine](https://github.com/Carasibana/ComfyUI-H3-FaceRefine).
No face-restoration or identity weights are bundled.

## Conditioning reuse and handoff {#memory-conditioning}

- **Sibling reuse.** A trusted scheduler passes one opaque `ConditioningReuseScope` to sequential I2VA/L2VA/FL2VA
  candidates of the same accepted request. One copied CPU entry (≤128 MiB, one-hour idle expiry) holds text
  outputs, token tags and clean keyframe latents only; noise, scheduler state, activations and media are never
  reused. Never derive a scope from public content or reuse it across owners.
- **In-memory handoff.** `conditioning_transport="memory"` passes encoder tensors to native denoising once,
  without writing a bundle, on a single device. The default `file` transport keeps the persisted contract
  for standalone native calls and services.

## Approximate FP8 loaders {#fp8-loaders}

`create_row_fp8_native_loaders` and `load_row_fp8_text_encoder` load explicitly prepared E4M3 weights with FP32
row scales through `H3Pipeline`'s trusted `native_denoiser_loader` / `text_encoder_loader` callbacks. Row
tiling bounds temporaries while keeping full-sequence attention. W8A8 is approximate, requires SM90/SM120 and
never silently switches arithmetic; BF16 remains the default, and the prepared-asset receipt and CLI do not
accept FP8 bundles.

## Timing and diagnostics {#timing-and-diagnostics}

`VideoResult.elapsed_seconds` spans the successful `generate` call including a first model load.
`stages.initialization_seconds` is loading inside this call, `stages.request_elapsed_seconds` excludes it, and
`stages.session_initialization_seconds` (also `pipeline.initialization_seconds`) is the session's original
load. Encoding and media stages report resume, suspend and call durations; `capture_diagnostics` separates
the official call, hook waits, bundle finishing and per-process resource deltas. Nested durations are
attribution, not additive totals.

## What a correctness result means {#what-a-correctness-result-means}

Compare conditioning and final latents to the reference implementation separately from visual quality, and
judge motion, appearance, instructions and sound against the original request. H.264 and AAC are lossy, so an
MP4 hash is not numerical equivalence; the official audio VAE can differ slightly between repeated requests.
Release checks cover the 14 conditioning tensors, final FP32 latents, decoded media and playable delivery.

## Qualification evidence {#qualification-evidence}

The table summarises what the current interfaces have been run on; numbers, scopes and limits for each
are in the [release notes](../reference/releases) and [performance records](../reference/performance).

| Capability | Evidence |
| --- | --- |
| SM120 resident pixel-frame budget (0.15) | RTX PRO 6000 96 GiB, FP8 resident trunk, Veda 44/6, face and six-step audio repair: 1920×1088/5 s 163 s, 2720×1536/5 s 261 s, 1920×1088/10 s 245 s, 42.1 GiB peak |
| 2720-pixel native HD (0.14) | Five 120-frame I2VA requests on RTX 4090 48 GB, BF16 hybrid Veda 44/6: 2720×1536 ×2, 1536×2720, 2368×1760, 2720×1152 |
| Extended hybrid (0.6.12) | 15 s and nine-image cases on single SM89 with Veda |
| Six-step audio (0.11–0.13) | RTX PRO 6000 FP8 (1024², 1536×1344) and RTX 4090 48 GB BF16 ring (512², 1024², 1248×832/10 s); hidden-context reuse on two 512² cases |
| Face repaint (0.6.34–0.10) | RTX PRO 6000 Server complete-video scenes, two- and four-step; crop-encoder profiles with matched final frames |
| In-memory handoff | Six H100 SXM requests with identical decoded planes and PCM |
| Sibling reuse | Dual-SM86 A/B/A with matching streams |

Evidence on one GPU class or canvas does not qualify others; serving systems keep their own qualification.
