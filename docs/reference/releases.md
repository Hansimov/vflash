# Release notes

## 0.15.0

Resident I2VA on one 96 GiB SM120 GPU is admitted by the same joint
`width × height × model_frames ≤ 2048² × 124` budget as native HD, instead of a fixed
1536×864 canvas. It therefore covers 1536×864 for fifteen seconds, 1920×1088 for ten
seconds and true 4 MP canvases such as 2720×1536 for five seconds. SM90 keeps its measured
1536×864 envelope; block ring remains the default and the rollback. No numerical path,
schedule or default changes.

On one RTX PRO 6000 Blackwell (96 GiB) with the resident FP8 trunk, Veda 44/6,
two-step face repair and six-step audio refinement, complete I2VA requests finished
1920×1088/5 s in 163 s, 2720×1536/5 s in 261 s and 1920×1088/10 s in 245 s, with a
42.1 GiB device-memory peak across all three. Frames at the start, middle and end of each
video and a native-resolution 4 MP crop showed coherent content without tiling. This is
capacity and visual qualification, not a sound-quality claim.

## 0.14.0

Qualifies a 2720-pixel long edge for image-conditioned native HD. The 2048² pixel cap,
1:4–4:1 ratio range and joint `width × height × model_frames ≤ 2048² × 124` budget are
unchanged, so true 4 MP canvases such as 2720×1536 (16:9), 1536×2720 (9:16) and
2368×1760 (4:3) are five-second requests. The request contract and the first-frame
conditioning bundle share `NATIVE_HD_MAX_LONG_EDGE`; wider long edges still fail before
model activation. No numerical path, schedule or default changes.

Five complete I2VA requests on one RTX 4090 (48 GiB reported memory) with BF16 hybrid
Veda 44/6 produced coherent 120-frame videos without tiling or repeated structure:
2720×1536 city and street scenes (672.0 and 658.9 s), 1536×2720 with a person
(669.6 s), 2368×1760 (663.5 s) and 2720×1152 (462.5 s). The last three ran the
unmodified release contract. This is visual qualification, not a sound-quality or
other-GPU claim; longer 4 MP requests remain outside the joint budget.

## 0.13.0

Adds explicit request-owned hidden-context reuse for six-step audio refinement.
`AudioRefinementRequest(..., reuse_hidden_context=True)` preserves the six audio
steps and video latents, while approximating later non-audio hidden states with
the first evaluation. The default remains full recomputation. Logical-layer
ownership supports the two-slot ring and resident implementation; GPU listening
and speed evidence is limited to two 512²/5s RTX 4090 cases, with roughly 34% less
audio time. Memory bounds, failure restoration and qualification limits are
specified in the [complete pipeline guide](../guide/complete-pipeline.md#audio-hidden-context).


## 0.12.0

Adds opt-in six-step audio refinement to the native two-slot BF16 SM89 weight ring
with Veda 44/6 and approximately 0.5 MP spatial context. All host modulation tables,
device copy plans, temporary Turbo-residual removal and attention bindings restore
on success or failure. Existing resident execution and defaults stay unchanged.

Two five-second RTX4090 pairs (1024² audio-only and 512² with application face repair)
kept all 120 decoded frames. The audio passes took 55.12/33.98 seconds; initialization,
cold/warm scope and quality limits are detailed in the complete-pipeline guide.
No larger/longer SM89 audio workload, 15% overhead target or universal sound-quality
improvement is established. Existing geometry and capacity limits remain unchanged.

## 0.11.0

Adds optional six-step audio-only refinement with a roughly 0.5 MP spatial context,
original video/time positions, matched modulation tables and decode headroom.
It requires an explicit resident hybrid I2VA request on one device with Veda 44/6.
Progress separates the four main and six additional evaluations; defaults stay unchanged.
Owned FP8 native/text factories and full-context row tiling remove duplicated integration
implementations without downloading weights or changing admission.

Two independent high-resolution cases reproduced all decoded frames and audio PCM through
the new API and the public FP8 factories. Two smaller combined face/audio requests kept
all previously repaired video frames and preserved the refined soundtrack through face
composition. These are limited RTX PRO 6000 Server integration checks; private expanded
capacity, application assets and production rollout are separate. Optional source-scene
conditioning retains its explicitly experimental qualification. No model weights are bundled.


## 0.10.0

Adds the explicit `batch2-fp16-norm-reuse-untiled` face encoder profile for
512-square, 124-frame inputs. It removes overlapping spatial encoder tiles,
retains temporal clip reuse and restores model state after success or failure.
The normal encoder and MP4/RGB defaults remain unchanged.

Three integration scenes on a 48 GiB RTX 4090 showed lower complete face-pass
latency. The exact public profile also reproduced all 120 final RGB frames of
one private prototype case. This is equivalence to that prototype, not to the
previous tiled encoder. Spatial context, normalization and some visual details
change; memory increases. See the complete-pipeline guide for timing and quality
limits. No new restoration model, weights or application scheduling is bundled.
Wheel, source distribution and native/pipeline images are published. Consuming
applications opt in separately; release does not change their deployed recipe.

## Earlier releases {#earlier}

The full notes, evidence and limits of every earlier version are kept in the
[0.15.0 copy of this page](https://github.com/Hansimov/vflash/blob/v0.15.0/docs/reference/releases.md)
and in each [source tag](https://github.com/Hansimov/vflash/tags). Highlights:

| Version | Highlights |
| --- | --- |
| 0.9.0 | `H3Pipeline.generate_rgb`: uint8 RGB array output with optional selected-frame decoding |
| 0.8.0 | `VideoRepaintRequest`: two- or four-step low-noise repaint with frozen audio |
| 0.7.0 | Consolidates the 0.6 complete pipeline, repaint building blocks and attention controls |
| 0.6.24–0.6.36 | Dense backends inside Veda (SM90 `cute-flash`/`fa3-flash`, SM120 `frost-flash`), dense edge width, in-memory conditioning, CPU admission and bounded media delivery, FP8 linear primitives, buffered weight reads, loader hooks, video-only output, audio decoder headroom, face encoder and native face repaint |
| 0.6.15–0.6.23 | SM120 resident trunk (≥90 GiB), SM120 hybrid FFN/QKV fusion and buffer reuse, lighter RGB quantization, checkpoint lineage, experimental Sage path |
| 0.6.13 | SM90/SM103/SM120 profiles, partition-aware discovery, portable assets, CPU-prepared hybrid tables |
| 0.6.12 | Hybrid/Veda up to 15 s and nine references within a joint budget |
| 0.6.11 | Native HD up to 2048² for five seconds |
| 0.6.10 | Veda sparse attention for LightX v0.1 on SM89 |
| 0.6.9 | Optional FlashVSR v1.1 upscaling |
| 0.6.8 | `HybridModel`: official Ref block 25–49 modulation on the FL trunk |
| 0.6.6 | Original LightX v0.1 four-step I2VA/FL2VA profiles |
| 0.6.0–0.6.7 | VDN8 community keyframe pipeline (SelfLift, learned latent lift), sibling conditioning reuse, 544p Turbo8 keyframes |
| 0.5.x | Complete-pipeline attention adapters, spatial VAE composition, FP32 attention LoRA, restoration, exact text-encoder prefix, single-SM89 Sol default |
| 0.4.0 | Base16 keyframes and measured Sol-Engine alignment |
| 0.3.x | Reference-video generation, lower 4090 QKV memory, wide tensor offsets, dual-SM86 T2VA |
| 0.2.x | Complete text-to-video, complete video on two 3080s, lower 4090 FFN memory |
| 0.1.x | Multi-reference video, prompt and image to MP4, explicit memory placement and resource ownership |

## Availability {#availability}

The current [profile and hardware guide](../guide/profiles) and
[complete-pipeline contract](../guide/complete-pipeline) define supported input
modes, precision choices and joint capacity limits. The [image inventory](https://github.com/Hansimov/vflash/blob/main/docker/images.json)
identifies the published native and pipeline images. Historical entries above
retain the package and image availability at their own release dates; they do
not override the current support tables. Model weights are separate downloads
under their own licenses.

New hardware, modes and refinements need their own execution and complete-media
qualification. A source option does not automatically upgrade a deployed service.
