# Generate a video {#generate-a-video}

Choose a complete-video path using [runtimes and interfaces](./runtime-selection): prepared Pika6 with Larry audio, original BF16 Pika, and fixed-profile `H3Pipeline` each have distinct inputs, assets and dependencies. The instructions below describe the fixed-profile path; they do not select Pika automatically.

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
| Reference video | one 2–5 s clip | `<Video 1>` | Single-SM89 Ref4; single-SM89/SM120 Veda hybrid |

Reference images are visual references, not timeline positions. The keyframe profiles serve I2VA,
L2VA and FL2VA serially from one prepared pipeline without a restart; `result.request_mode` records
the actual input type and `profile_id` the prepared identity. Other profiles reject mode changes before
execution. The prompt is used verbatim; describe each image's subject and role.

**Canvas.** Width and height are multiples of 32 with an aspect ratio between 1:4 and 4:1. Ordinary
requests allow up to 1,048,576 pixels. Image-conditioned native HD allows up to 2048² pixels with a
long edge of at most 2720, for example 2720×1536 (16:9), 1536×2720 (9:16), 2368×1760 (4:3) or 2048².
Reference-video output is five seconds: up to 1,048,576 pixels on the Veda hybrid, or 475,136 on Ref4.

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

Piped media delivery has a 120-second inactivity deadline and a 600-second total encoding deadline.
Completed RGB input writes and encoded-file growth refresh the inactivity clock. On expiry, the owned
encoder is reaped and temporary media is removed. Direct `encode_mp4` callers can set
`stall_timeout_seconds` and `timeout_seconds`; file input uses the total deadline.

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
**`<Video 1>`**. The Veda hybrid shares its official Ref graph on a single SM89 or SM120; one pipeline
alternates image and video requests. Supply one complete 2–5 s MP4, MOV or WebM, at most 20 MiB and
1344×768 source pixels, with one square-pixel video stream at up to 240 fps. Source audio is discarded.
Official encoding normalizes to 24 fps with the fixed temporal alignment and prefix budget: a 120-frame
input uses its first 107 frames on the official grid, producing 32 encoded temporal positions.
The output is a new five-second video guided by that reference.

Each conditioner owns a CPU reference-latent cache capped at 128 MiB. Repeated decoded identities and
geometry reuse the official VAE result; text and noise are recomputed. Closing the pipeline clears the
cache; callers own media authorization and isolation. Default `reference_video_profile="full"` retains
full spatial context. Explicit `"spatial-half"` average-pools the encoded spatial grid while retaining
all encoded temporal positions. This approximate Veda-hybrid option can change faces, detail and framing;
prefer full for identity-sensitive work. See the [conditioning contract](../reference/runtime-assets#video-conditioning).

Reference-latent reuse and compression were informed by MIT-licensed
[RefMod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod/tree/517e8c419a8584fbf69fceb487bc39d83e6a6ec8).
This implementation reuses official latents and offers explicit spatial pooling; RefMod's per-asset MSE
fitting and voice-reference work are separate methods.

### Combined appearance images and motion video (0.21)

The 0.21 hybrid path combines one video with one to three ordered images:

```python
request = VideoRequest(
    prompt="<Picture 1> supplies appearance; <Video 1> supplies action and camera timing.",
    references=(Path("appearance.png"),),
    reference_video=Path("motion.mp4"),
    width=960, height=544, duration_seconds=5,
)
```

Use `<Picture 1>` through `<Picture 3>` for image roles and `<Video 1>` for the video.
Images retain target-area, downscale-only match sizing; the video retains its official geometry.
This path defaults to full spatial reference on a single SM89/SM120 Veda hybrid. The five-second output and
1,048,576-pixel canvas bounds remain. Image ordering, identities and target canvas participate in
reference-cache identity, so reusing a guide with a new appearance recomputes the combined condition.

Initial execution evidence covers one appearance image plus a five-second guide on a 48 GiB SM89:
960×544 output completed in 280.7 seconds, with 60.2 seconds encoding and 201.3 seconds denoising;
session initialization was another 90.9 seconds. The complete handoff kept the appearance and action
sequence. Plain and detailed image-to-video prompts also achieved this simple action in 76.1/70.5 seconds
with different conditioning and cold/warm state. Choose motion references for a concrete control need;
broader scene coverage, three-image capacity and target-hardware performance remain development work.
For rough motion guides, explicit `reference_video_profile="spatial-half"` pools just the final video
latent grid. All preceding appearance-image latents and video temporal positions are preserved.
In the same initial scene, this reduced the condition prefix from 32,766 to 8,574 rows. SM89 call time
was 168.9 seconds (59.1 encoding, 90.1 denoising), with the handoff still completed. SM120 full/pooled
calls on two cached hosts took 110.3/79.4 seconds, including different cold encoding costs; denoising
was 74.0/31.8 seconds. These are scene-specific approximate-control results, with wider motion and
identity coverage still being measured. Version 0.21 adds combined inputs; earlier releases retain separate image/video request modes.

## Integration and tuning

<span id="extended-hybrid"></span><span id="exact-text-encoder-prefix"></span><span id="veda-dense-edges"></span><span id="memory-conditioning"></span><span id="fp8-loaders"></span><span id="timing-and-diagnostics"></span><span id="what-a-correctness-result-means"></span><span id="qualification-evidence"></span>
- [Conditioning, memory and hardware profiles](./pipeline-conditioning) — Use this guide when integrating the fixed-profile pipeline on a new device or measuring a change to its numerical path. Start with a playable control at the same resolution, profile and seed.

<span id="optional-audio-only-six-step-refinement-source-main"></span><span id="audio-hidden-context"></span><span id="audio-kv-overlap"></span><span id="audio-ring-weight-cache"></span><span id="explicit-native-face-repaint"></span><span id="encoded-source-soundtrack-restoration"></span><span id="longer-audio-passes-019"></span><span id="compact-prepared-tables-019"></span>
- [Soundtrack refinement and face repaint](./pipeline-refinement) — Choose an audio pass when the picture already meets the brief. Face repaint has its own masks, weights and execution limits. Record the selected options with the delivered film.

<span id="retaining-native-audiovideo-latents"></span><span id="experimental-native-av-target-prefixes"></span>
- [Continue from native audio and video latents](./native-av-prefix) — Use these Python interfaces to retain the model output before media encoding and to supply compatible history to the next request. The overlap, source cut and audio clock must agree across both windows.

<span id="experimental-pika-image-reference-runtime"></span><span id="experimental-qad-visual-weights-0260a2"></span>
- [Pika reference recipes and QAD visual weights](./pika-reference-weights) — For runtime setup, start with [Pika H3 Flex](./pika-flex). Choose one of the prepared paths below according to the references, duration, available weights and device.
