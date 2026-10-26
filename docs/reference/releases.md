# Release notes

## 0.21.0

The single-SM89/SM120 Veda hybrid accepts one video with one to three ordered appearance images.
Images precede the video in the official graph; prompts retain independent `<Picture N>` and `<Video 1>` labels.
Images use target-area match sizing, while the video retains its official CFR24 geometry.
Conditioning schema 6 binds both modalities and checks their combined prefix. Existing schemas 1–5 keep their contracts.
The reference cache includes ordered image identities and target canvas, preserving ownership when a motion guide
is reused with a different appearance.

Full spatial video remains the default. Explicit `reference_video_profile="spatial-half"` pools only the final
video latent grid, preserving every preceding image latent and all encoded video temporal positions.
The full cache entry stays available for later full-reference requests. Output remains five seconds and up to
1,048,576 canvas pixels. See [combined inputs](../guide/complete-pipeline#reference-video).

Initial evidence: one appearance image and a five-second CPU-rendered two-person prop handoff,
960×544/120 delivered frames, four denoising steps, Veda 44/6, native joint audio and web-v1 delivery:

| GPU and execution | Full reference call | Video spatial-half call | Pooled encoding / denoising / media |
| --- | ---: | ---: | ---: |
| RTX 4090 48 GiB, BF16 ring, Torch dense | 280.706 s | 168.883 s | 59.060 / 90.092 / 18.383 s |
| RTX PRO 6000 Blackwell Server, resident FP8, SageAttention2 | 110.342 s | 79.395 s | 35.187 / 31.809 / 11.410 s |

Complete calls exclude session initialization and application transport. The two SM120 runs used different
cached hosts and cold encoding states; initialization was 33.843/19.981 seconds. SM89 full initialization was
90.879 seconds and pooled initialization 94.774 seconds. The condition prefix fell from 32,766 to 8,574 rows.
SM89 full-reference denoising allocation peaked at 14.99 GB; whole-window NVML samples peaked at 17,159 MiB.
Torch 2.11/CUDA 13 with the pinned official conditioner.

Sampled full frames and contact sequences retained appearance and completed the handoff. This is one-scene
evidence for an approximate motion-control option; wider actions, styles and three-image workloads retain
their separate qualification. Plain/detailed I2VA prompts also performed this simple action, so motion
references should serve a specific control need. Existing audio-refinement defaults and qualified scopes remain.

## 0.20.0

Adds an opt-in `weight_cache_bytes` budget to `AudioRefinementRequest`. The BF16
two-slot ring keeps selected weights on device during audio refinement, preserves
the original compute-slot addresses, and restores its bindings on every exit.
The default is zero; resident trunks retain their existing behavior. The scope
reserves 20 GiB for context and overlapping decode work.

One RTX 4090 **48 GiB**, BF16, Veda 44/6 with Torch dense attention, twelve audio
steps, 0.25MP audio context, hidden-state reuse and video-decoder overlap:

| Five-second I2VA | Warm complete call | With 20 GiB budget | Saved |
| --- | ---: | ---: | ---: |
| 960×544 percussion | 102.644 s | 93.252 s | 9.392 s |
| 960×544 laughter | 103.964 s | 94.095 s | 9.870 s |
| 1024² percussion | 158.456 s | 149.528 s | 8.928 s |
| 960×544 speech | 81.360 s | 70.296 s | 11.063 s |

These complete MP4 calls include cache setup and exclude model initialization,
application queueing and transport. The first cold call is reported separately
at 110.150 s. The retained prefix is 27/50 blocks, 21.01 GB; recurring H2D traffic
falls from 466.96 GB to 214.80 GB, plus one 21.01 GB preparation copy. Peak device
allocation for the cached cases is 33.07–35.70 GiB. Torch 2.11/CUDA 13.

All 120 decoded video frames and all stereo PCM samples match each row's control.
Four further complete calls qualified the owned native implementation; its two
cached outputs also match the prototype's full decoded media. This establishes
weight-placement equivalence within the selected audio recipe. Sound acceptance
and the earlier recipe's transient limitations retain their existing scope.
Other durations, GPU capacities and memory layouts require their own qualification.


## 0.19.0

Adds matched twelve-step (`denoise=0.9`) and twenty-step (`denoise=0.95`) audio passes,
encoded-source audio conditioning and compact prepared audio modulation storage.
The engine keeps audio refinement opt-in; applications choose their deployed recipe.
Video generation remains four steps, with joint audio conditioning. Completed video latents stay fixed
through the additional audio pass. Hidden/K/V reuse and video-decoder overlap are explicit choices.

Complete MP4 calls, five-second I2VA with the same input within each row, measured on one GPU:

| Device / workload | Six-step cache + overlap | Twelve-step cache + overlap | Twenty-step cache + overlap | Twenty-step full context computation + overlap |
| --- | ---: | ---: | ---: | ---: |
| RTX 4090 **48 GiB**, BF16 two-slot ring, 960×544 percussion | 86.34 s | 102.71 s | 128.66 s | 176.00 s |
| RTX 4090 48 GiB, BF16 ring, 1024² percussion | 143.45 s | 155.19 s | 179.28 s | — |
| RTX PRO 6000 Blackwell **Server**, resident FP8, 512² claps | 12.81 s | 12.88 s | 13.33 s | 34.18 s |
| RTX PRO 6000 Blackwell Server, resident FP8, 1344×768 guitar | — | 44.09 s | 43.25 s | 66.31 s |

These are complete pipeline calls, excluding model initialization, network transfer and application
queue/first-frame preparation. Local uses hidden reuse and Torch dense attention; remote uses K/V
reuse and SageAttention2. Both use Veda 44/6 and a 0.25MP audio context, Torch 2.11/CUDA 13.
First-call initialization and kernel warmup are separate from the warm rows above.
The 44/6 partition remains an explicit application choice; the engine's existing default stays 40/10.

Without caching/overlap, 960×544 ten-second local generation measured 178.84 / 228.36 / 295.43 s
for six/twelve/twenty steps. On remote 2048² five-second generation, full serial six/twenty measured
243.79 / 270.55 s. The existing per-device cache/headroom gates continue to apply.

Twelve/twenty-step recipes change both evaluation count and re-noising strength. Human feedback on
source-conditioned speech favored twenty steps for naturalness; residual crowd transients remain a
quality limitation. Cached and fully recomputed audio preserve the picture but can differ in sound.
The measured set includes speech, percussion, laughter/claps, acoustic music and ambience. These
measurements qualify execution and latency; sound acceptance remains tied to each brief and listening.

Compact tables deduplicate identical complete rows and retain every original BF16 bit. Twenty-step
storage decreases from 611.00 MB to 400.92 MB; all 51 full reconstructed tables are bit-identical.
Legacy tables, original files and the ordinary inference layout remain supported.

## 0.18.0

Adds hybrid reference video on single SM89/SM120, an owned 128 MiB CPU VAE cache, explicit spatial-half
reference context, resident-SM120 audio K/V reuse, and owned video-decode/audio-refinement overlap.
[The complete guide](../guide/complete-pipeline#reference-video) defines input and lifetime contracts.
Engine audio refinement stays explicit; applications select their defaults and supply matching tables.

Measured complete MP4 calls, five seconds, excluding model loading and service transport:

| Workload | RTX 4090 48 GiB, BF16 ring, 44/6 Torch | RTX PRO 6000 Server 96 GiB, FP8 resident, 44/6 Sage |
| --- | --- | --- |
| 512² transient, serial → selected overlap | 65.41 → 60.69 s (hidden reuse in both) | 20.12 → 14.03 s (full audio context → K/V + overlap) |
| 1344×768 music, serial → selected overlap | 148.08 → 142.63 s | 49.91 → 42.55 s |
| 512² identity reference, full cold → spatial-half cached | 125.02 → 61.32 s | 47.10 → 20.84 s |
| 1344×768 camera reference, full cold → spatial-half cached | 356.63 → 178.04 s | 134.74 → 63.25 s |

The reference rows combine VAE-cache reuse and spatial compression; they do not isolate either gain.
A separate full-context 4090 pair reduces reference VAE work from 22.62 s to 0.0034 s with identical
120 decoded video frames. Ten matched audio-overlap pairs also preserve all 120 video frames;
nine preserve decoded PCM exactly, while the first 4090 speech pair differs by 0.001864 PCM RMS.
Speech transcripts agree. These are matched-scene observations, with independent visual/audio evidence;
spatial compression changes facial detail and composition, so full reference remains the default.
Final application-worker calls on the same PRO 6000 also completed three refined I2VA requests and one
full video reference using the owned public implementations. CPU checks cover cleanup, admission and isolation.

## 0.17.2

Piped MP4 delivery now ends an encoder that makes no progress for 120 seconds, while retaining
the 600-second total deadline. Completed RGB writes and encoded-file growth refresh the inactivity
clock. Both deadlines reap the owned process and clean temporary files; publication follows the
existing successful-media checks. Codec settings and input tensors retain their existing behavior.
CPU checks cover blocked children, actual FFmpeg output and file/pipe media equivalence.

## 0.17.1

Align resident SM120 hybrid request admission with I2VA, L2VA, FL2VA and image Ref modes.
The existing pixel-frame ceiling and multi-reference 960×544/5-second limit remain in effect;
SM90 retains its existing I2VA scope. This fixes rejection before inference and leaves model
weights, attention and denoising unchanged. CPU contract checks pass. Resident RTX PRO 6000
SM120 runs now complete FL2VA at 1440×1440/10 seconds, L2VA at 1344×768/15 seconds,
and image Ref with two and nine references at 960×544/5 seconds. I2VA also completes
at 2048×2048/5 seconds. These runs establish execution and output delivery for the listed
inputs; visual quality remains scene-dependent and is assessed against each original brief.

## 0.17.0

`veda_dense_edge_layers=4` selects dense layers 0–3 and 46–49, with Veda on the
remaining 42. Both CLI entries and complete pipelines accept 3, 4 or 5; default 5
keeps the existing 40/10 behavior. The optional audio context pass preserves the
explicit 44/6 or 42/8 partition. On SM120, Sage uses INT8 QK / FP8 PV in dense and
sparse layers, while FROST uses BF16 dense and INT8 QK / FP16 PV sparse attention.

Same physical RTX PRO 6000, resident FP8 weights, four evaluations, matched image,
prompt and seed per row; complete engine seconds, with model loading and network
delivery recorded separately:

| Five-second I2VA input | Sage 44/6 | Sage 42/8 | FROST 44/6 | FROST 42/8 |
| --- | ---: | ---: | ---: | ---: |
| 1344×768 | 46.2 | 47.0 | 47.5 | 47.0 |
| 544×960, six-step audio pass | 31.3 | 31.5 | 31.4 | 31.2 |
| 2720×1536 | 222.9 | 225.2 | 244.9 | 257.5 |

These are three input scenes and one timed request per configuration. Native crops
show changes in expression, motion phase and local detail; scene-specific quality
is judged against the reference and action brief. Additional scene coverage remains
a distinct comparison. The earlier 0.16.0 four specification pairs covered two scene
families; their flicker/sharpness statistics describe that cohort.

The 0.16.1 explicit resident-memory threshold is included. Published native and
pipeline images report 0.17.0 under their unprivileged user; image digests and
immutable dependency parents are in `docker/images.json`. SageAttention is installed
separately when that option is selected.

## 0.16.1

`VFLASH_SM120_RESIDENT_GIB` lowers the SM120 card size admitted to resident weights (default 90,
accepted 24–90). A deployment that serves only shapes whose measured peak fits a smaller card,
such as a 32 GiB RTX 5090 limited to 1 MP / 5 s requests by its scheduler, sets 30; a larger
shape then fails with CUDA OOM rather than silently falling back. SM90 stays at 75 GiB and
nothing else changes.

## 0.16.0

`veda_dense_backend="sageattention2"` on SM120 becomes one INT8 Q·K / FP8 P·V policy: besides
the dense edge layers (SageAttention 2.2.0 CUDA), Veda's sparse layers now multiply P·V in FP8.
Their walk, masks, tile selection and INT8 Q·K are unchanged; V is quantized per head and
channel to e4m3 and accumulates in FP32. A startup self-test bounds the sparse kernel against
Veda's own (relative error < 5%) and there is no silent fallback. Receipts add
`sparse_precision`. Every other backend, default and profile is unchanged.

Same RTX PRO 6000 Server, resident FP8 weights, Veda 44/6, same seeds, FROST BF16 dense with
FP16 sparse P·V as baseline:

| Request | Denoise, baseline | Denoise, 0.16 policy | Complete request |
| --- | ---: | ---: | ---: |
| 2720×1536 / 5 s | 211.1 s | 182.3 s | 276.2 → 237.8 s |
| 1344×768 / 15 s | 110.6 s | 96.8 s | 149.7 → 135.3 s |
| 1 MP dialogue / 5 s | 28.4 s | 26.2 s | 43.1 → 40.4 s |
| 0.5 MP dialogue / 5 s, 0.25MP audio pass | 22.1 s | 20.7 s | 31.1 → 29.3 s |

The baseline arm ran first and its first request includes some warm-up outside denoising.
Precision changes the sampling trajectory. Without reference, frame-to-frame change of the
fine-detail luma band stayed at 0.97–1.02 of the baseline and Laplacian sharpness at
0.96–1.01 across these four pairs. That is a bounded comparison, not an equivalence claim.

## 0.15.1

Adds a qualified 0.25MP spatial budget for the six-step audio pass's video context:
`AudioRefinementRequest(..., context_pixels=250_000)` beside the default 500,000. The pass
keeps the same uniform token subsample, original RoPE, every time step and frozen video
latents; no other numerical path, schedule or default changes.

With the release code (guard widened on 0.15.0), the audio stage of complete I2VA requests
took, at 0.25MP versus 0.5MP:

| GPU, execution | Request | 0.25MP | 0.5MP |
| --- | --- | ---: | ---: |
| RTX 4090 48 GB, BF16 block ring | eight 5 s canvases, 0.45–1.06 MP (median) | 34.1 s | 50.6 s |
| | 2720×1536 / 5 s | 42.2 s | 65.4 s |
| | 1920×1088 / 10 s | 56.0 s | 107.4 s |
| | 1344×768 / 15 s | 75.2 s | 160.1 s |
| | ≤1.4 MP dialogue / 5 s (three clips, median) | 33.6 s | 52.2 s |
| RTX PRO 6000 96 GB, resident FP8 | 2720×1536 / 5 s | 11.9 s | 19.9 s |
| | 1920×1088 / 10 s | 16.8 s | 32.4 s |
| | 1344×768 / 15 s | 23.5 s | 49.2 s |
| | 1 MP dialogue / 5 s (three clips, median) | 7.5 s | 14.4 s |

Delivered frames were identical across budgets. Whisper large-v3-turbo transcribed three
English and Mandarin dialogue clips on each GPU identically in both budgets and without
improvement. The speech-band envelope of 0.25MP against 0.5MP had zero lag in all 20 clips,
with correlation 0.97–1.00 in 19 and 0.77 for one mechanical 4 MP clip; in all but that clip it
stayed closer to 0.5MP than 0.5MP stays to the unimproved track. Audiobox Aesthetics scores
moved by at most ±0.3. This is speed and objective-metric qualification, not a listening claim.

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
