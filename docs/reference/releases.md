# Release notes

## 0.28.0a7 (preview)

Remake journals now use a private temporary file per write. A file left by an interrupted process no longer blocks subsequent atomic receipt updates. Recovery continues to identify an accepted submission by its original execution ID.


## 0.28.0a6 (preview)

Adds an eight-second option to the six-step Pika content-reference recipe with ten native
soundtrack refinement steps. The reference encoder, spatial audio attention and decoder
share a 57-latent-frame clock. One 1280×720 RTX PRO 6000 run produced 192 frames at 24 fps
and stereo audio. Small faces in wide shots can still lack detail.

## 0.28.0a5 (preview)

The six-step Pika content-reference recipe now supports 12-second output. Its sampled
video, attention layout, audio refinement and decoder follow the same duration. One
1280×720 run on a 96 GB RTX PRO 6000 produced 288 frames at 24 fps with ten soundtrack
refinement steps. On this 12-second recipe the audio stage can keep up to 12 GiB of frozen
attention keys and values in host RAM when its device cache budget is insufficient. Five seconds remains the default. Dialogue accuracy
and facial detail vary with the scene.

Remake can consume request-bound precomputed text conditioning, reuse an identical
source-image encoding for a pinned frame, and explicitly select half-size source video
conditioning for eligible large edits. See the [complete pipeline guide](../guide/complete-pipeline).

## 0.28.0a4 (preview)

Adds explicit prepared canvases up to a 1440-pixel short edge. Larger canvases use
`VFlashRemakeCanvas` to widen the original conditioning node's range and delegate its
computation unchanged; smaller canvases keep their original graph. Sampling defaults and
per-window temporal bounds are unchanged.

One NVIDIA RTX PRO 6000 executed a 1920×1088, 226-model-frame edit. This is a bounded
hardware execution observation, not a general memory, quality, throughput or service-time
guarantee. Application timelines and delivery sizes remain separately owned. See
[extended canvas setup](../guide/remake#extended-canvas-preview).

## 0.28.0a3 (preview)

Adds explicit static source-region conditioning and an optional pure FL2V v1.2 four-step
composition. Six-step defaults and durable completion recovery remain. The mask reuses the
source latent in the existing sampler, without OCR or final pixel compositing; its file identity
participates in recovery.

Complete SM120 96 GiB edit evidence covers a portrait product clip and a multi-shot 20-second
landscape clip. Protected dynamic captions survived; text/inset and hand/detail failures remain
in unprotected four-step outputs. The long clip stalled in upstream source decoding and needed
a confirmed stop followed by recovery of its last two windows. Step savings do not establish
reliable end-to-end speed. These optional experimental interfaces do not change default quality
or deployment guarantees. See [scopes and setup](../guide/remake).

## 0.28.0a2 (preview)

Adds offline recovery of a durable original Remake completion. The request graph, input identities,
executed history and output media must match; recovery performs no new model submission.

## 0.28.0a1 (preview)

Adds an explicit original Viggle Remake client for source-video editing, with separate
six-step swap/edit models and optional R2VA Veda attention. Frozen source, reference images,
target pins and generated-history frames have distinct roles. The owned loopback executor
persists one submission, reconciles a lost reply by that original identity, verifies the executed
graph and saved media, and asks the caller's process owner to confirm cancellation or timeout.
The `remake-client` extra installs Pillow only; the pinned Comfy model runtime remains separate.

Original-graph GPU evidence covers one SM120 96 GiB allocation at 832×480 with 124, 192 and
226 model frames: character/object/style edits, two identities, and 17-frame generated-history
continuation. Full and sparse graphs retain separate evidence; Veda changes pixels. The new
client reproduces eleven of those frozen graphs and has local HTTP/media recovery checks.
Deployment through the new transport, longer multi-window films and other shapes remain open
qualification scopes. This preview does not change ordinary Pika/native profiles or published
Docker images. See [Remake inputs and ownership](../guide/remake).

## 0.27.0

Promotes native four/five-second prepared Pika FL generation with a separately selected
Larry ten-forward soundtrack. The six-step visual sampler, model partitions and attention
recipe are unchanged. Four seconds use 107 model frames, 32 video latent times, 96 delivered
frames and 128,000 stereo samples. Text and first-frame inputs use the same request clock;
scoped upstream validation and audio hooks restore their owners after failures.

Real complete 96 GiB SM120 execution covers portrait four-second first-frame input,
landscape five-second first-frame input and both orientations of four-second text input at
1376×768 / 768×1376 model canvases. The original encoder supports exact 1366×768 / 768×1366
cropping with unit pixel aspect ratio. The resulting videos were fully decoded and played;
two spoken lines were reviewed as complete without evident new audio issues. Framing and
motion-completion defects remain in some scenes, so this is bounded execution and media
qualification, not universal semantic quality. Ordinary H3 profiles, QAD and Ref durations
retain their existing scopes. Published container images keep their separately documented versions.

## 0.27.0a4 (candidate)

Declares square pixels directly with `setsar=1` during the original encode. Setting a display
aspect ratio rounded 1366×768 / 768×1366 to a small non-unit pixel ratio in FFmpeg,
which strict product canvas validation correctly rejected. Both exact canvases and the
existing 720P shape now have real codec checks. Media geometry qualification remains
separate from picture and sound acceptance.

## 0.27.0a3 (candidate)

Scopes the pinned upstream audio pipeline's minimum duration to the explicit four-second
request, restores it afterward, and calls its original full input validation before sampling.
The actual 107 model frames pass unchanged; prompt, canvas, step and upper-bound checks remain.
The a2 spatial/KV fix alone still encountered the upstream five-second admission limit.
New GPU/media qualification remains pending.

The final media decoder also accepts the four-second native window (96 frames and 128,000 stereo samples); FFmpeg delivery is covered on CPU.

## 0.27.0a2 (candidate)

Fixes the inner spatial-audio frame clock for the explicit four-second Pika candidate.
The ten-forward audio pass retains all 32 video times and its full image condition; the default 37-time
five-second scope is unchanged. Rejects unqualified short zero/twelve-audio configurations before
execution. CPU regression combines the real pinning, spatial selection and KV owners instead of
replacing the audio context with a stub. New GPU and media qualification are still pending.

## 0.27.0a1 (candidate)

Adds explicit CPU admission for prepared Pika AV four/five-second text or first-frame requests,
including aligned 1376×768 and 768×1376 canvases. A four-second request uses 96 delivery frames,
107 model frames and 32 video latents; picture, native soundtrack, Veda geometry and delivery
receive the same request clock. Ordinary fixed-profile H3 admission keeps its previous envelope.
The new scope is disabled for QAD. Prepared reference modes retain their prior duration rules.

CPU tests cover actual sampling/audio call counts, clock propagation, video preservation and
owner restoration, plus request rejection before accelerator access. GPU capacity, media quality
and complete service delivery for the added scope are pending qualification. Recommended stable
installation and published Docker images remain unchanged.

## 0.25.1

Rejects a reference recipe passed to an FL-only Pika session before loading models
or optional tensor dependencies. Each prepared session now requires its own exact
frozen recipe type. Sampling, numerical settings and existing defaults are unchanged.

## 0.25.0

Adds the opt-in prepared `PikaReferenceAVRuntime` and frozen `PikaReferenceRecipe`.
Ordered image references keep the original FP32 conditioning and Ref six-forward
sampler. Optional Larry sound uses a separately loaded original FL model and
preserves the generated picture. Compact CPU masters and sequential transformer
residency make ownership and transfer stages explicit.

Reference geometry remains tiered by default. Explicit 1024/2048 short-edge
candidates retain the requested output canvas and report their policy and token
count. The runtime rejects a recipe whose budget differs from its allocation;
reference recipes have identities separate from FL-only Pika generation.

Complete GPU evidence covers five-second landscape/portrait outputs within 1 MP,
one/two image references, original-tier and explicit-1024 conditioning, native
single-tile decoding and Larry10 on complete 96 GiB SM120 allocations. This is
execution and bounded performance evidence, not universal identity/action/sound
qualification. More references, longer video, reference audio/video, and automatic
serving adoption remain outside this prepared candidate's measured scope.
Fixed-profile CLI/HTTP defaults and published container images are unchanged.

## Earlier releases {#earlier}

[Changes, compatibility and distribution history through 0.24](./releases-archive).

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
