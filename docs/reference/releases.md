# Release notes

## 0.6.23

Expose CPU-only complete-request admission before allocating an accelerator, and explicit bounded media delivery options: `video_input="pipe"` avoids a full RGB temporary file, while `video_threads` controls codec concurrency. Defaults remain file delivery and codec auto-threading. These are the previously published reusable changes; this release pins them for distributed worker images.

A CPU encoding comparison at 2048²/120 frames found explicit 16-thread encoding 8.68% faster on average and 50.54% lower peak anonymous memory than the measured auto-thread control. Full decode, audio and compression checks passed; thread settings can change lossy bitstreams. This is a CPU-stage result, not a guarantee of lower remote cold-start or whole-generation latency. Weight preparation, cache placement and rental ownership remain application responsibilities. Flash remains the default; the experimental Sage quality limitations in 0.6.22 remain unresolved.

Source and wheel are released. Private integration evidence: `video-gen` `fc9306fe`; only aggregate results are public. Existing public container tags and running workers keep their pinned versions.

## 0.6.22

**Quality follow-up, 2026-10-10:** the optional Sage path remains experimental. High-resolution temporal instability has been reported; consecutive-frame checks found detail fluctuation in both compared backends and did not isolate its cause. Decode, playback and sparse frame sampling do not establish temporal-quality qualification. The timings below remain execution measurements, not accepted-video throughput. No default or runtime change is made by this documentation correction.

Keep profiled native blocks on the ordinary path's activation lifetimes: do not retain attention temporaries or pre-gather FFN modulation across the next large allocation. CPU FP32/BF16 parity and weak-reference checks cover both paths; a seven-video RTX 5090 integration completes the previously failing profiled 1536×864/15 s request and 2048²/5 s output. This fixes diagnostic-path memory overhead, without broadening the existing canvas contract.

Add an explicit `veda_dense_backend="sageattention2"` option for single-SM120 Veda, also exposed by `generate --veda-dense-backend sageattention2`. The ten dense layers use separately installed, pinned SageAttention 2.2.0 with INT8 QK / FP8 PV; the other forty Veda layers are unchanged. Flash remains the default. Unsupported architectures, missing extensions and kernel failures raise errors instead of silently changing execution. See [installation and measured scope](../guide/veda#explicit-sm120-dense-block-acceleration).

Same-instance I2VA warm full-generation controls measured 1536×864/15 s at 257.718/223.718 s on RTX PRO 6000 Server (13.19% shorter) and 379.763/310.595 s on RTX 5090 (18.21%). PRO 2048²/5 s was 314.753/302.123 s (4.01% shorter). The 5090 also completed 2048²/5 s at 368.807 s and 26.982 GiB denoising allocation, without a matched Flash control. These times include conditioning, denoising and media but exclude initialization and delivery. The PRO's 1205.634 s initialization prevents claiming a cold-start improvement. Complete AV decode, application playback, five sampled times and native crops accompany the comparisons; changed motion, existing camera adherence defects and unassessed audio semantics remain. No general quality-equivalence or H100 Sage qualification is claimed.

Private integration evidence: `video-gen` `40745220`; only aggregate measurements are public. Product deployment is separate from this package release.

## 0.6.21

Reuse the exclusively owned SM120 hybrid FFN base projection for its activated value half, instead of allocating another full output. The gate half and adapter remain unchanged, strict BF16 boundaries are retained, and the next linear consumes a strided view. Other architectures retain the allocated-output path. The owned contract is explicit and rejects autograd inputs.

On a 32 GB, 600 W RTX 5090, a seven-video integration completed 2048²/5 s in 451.039 seconds (26.982 GiB peak denoising allocation), 1536×864/15 s in 364.943 seconds (24.143 GiB), and 1440²/10 s in 392.383 seconds. The earlier 4.28 GiB FFN output allocation failure is avoided. Same-host 1920×1088/5 s A/B/A2 produced identical complete MP4 bytes; its denoising peak stayed 16.960 GiB and no speed gain was established. This is a capacity improvement, not a general speed or image-quality claim. Separate-host 500 W and 600 W runs do not isolate this optimization's latency effect. Full media decode, multi-time visual review and application playback accompany the integration; existing camera motion and unassessed audio semantics remain. Product routing is unchanged.

Private integration evidence: `video-gen` `3e20f9f3`; only aggregate results are public. The promoted kernel also passed four real-GPU shape checks, including subsequent strided linear output without a hidden clone.

## 0.6.20

Extend the existing strict BF16 FFN/QKV small-adapter fusion to SM120 original-v0.1 hybrid weights, and release dead FFN modulation/normalization tensors before the next large allocation. A same-RTX-5090 1920×1088, five-second A/B/A2 comparison reduced denoising allocation from 20.086 to 17.949 GiB and denoising time by about 2.1%; full-generation speedup was not established because media-stage time varied. A separate same-card lifetime A/B/A2 reduced 17.949 to 16.961 GiB with identical complete MP4 bytes, without a measured speed gain.

The combined path completed 1536×864 at fifteen seconds on a 32 GB RTX 5090 (500 W): 431.601 seconds total generation, 27.916 GiB peak denoising allocation, full audio/video decode and multi-time visual review. A 2048² five-second request still failed its 4.28 GiB FFN output allocation; the following ten-second case did not run. This is bounded I2VA integration evidence, not a general four-MP or quality qualification. Existing unwanted camera movement and unassessed audio semantics remain. Use `PYTORCH_ALLOC_CONF=expandable_segments:True` for this measured block-ring path. CPU ownership/dispatch checks cover the existing and added profiles. Private integration evidence: `video-gen` `674c84fc`; only aggregates are public.

## 0.6.19

Avoid a separate FP32 copy before RGB scaling: multiplication owns its result, then rounding and clipping reuse it. Lower-precision input retains the owned conversion buffer used since 0.6.18. Inputs, quantization and codec settings stay unchanged. On eight CPUs, a decoded FP32 2048²/120-frame clip encoded in 15.145 seconds versus 17.342/17.627 seconds for controls, approximately 13.4% faster. Complete output bytes and audio/video decode matched. This is CPU encoding evidence; remote full-generation improvement remains unqualified. Private integration source: `video-gen` `fc41b4b8` (only aggregate evidence is public).

## 0.6.18

Reuse one owned FP32 block for RGB quantization, reducing temporary CPU allocations without changing the caller's decoded tensor, rounding, codec or audio processing. On an eight-CPU allocation, encoding a decoded FP16 2048², 120-frame RGB clip took 15.903 seconds versus 18.863/18.846 seconds for the before/after controls (15.7% faster). Complete MP4 bytes matched and all audio/video decoded. This is encoding-stage evidence, not a full-generation or GPU throughput claim; no thread-count or hardware-profile defaults change. A later FP32 CPU control measured 17.066 seconds versus 17.180/17.386 seconds (about 1.3%), while one remote PRO same-instance 1080-class control was slower (13.472 versus 8.427/7.519 seconds). The official VAE returns FP32; the FP16 gain must not be generalized to every pipeline or host.

## 0.6.17

Extend explicit hybrid/Veda trunk residency to single SM120 allocations with at least 90 GiB. A same-PRO-6000-Server A/B/A2 comparison completed nine videos: warm five-second latency improved 18.1% and a matched fifteen-second request 3.3% versus the faster return control, while initialization plus first output increased 18.4 seconds. Peak host RSS fell from 108.7 to 68.7 GiB. Use total batch time when selecting residency; the tested three-output batch did not amortize startup. Default block-ring behavior, I2VA limits, allocator requirement and smaller-card exclusions remain. See [scope and evidence](../guide/profiles#sm120-resident).

## 0.6.16

Add explicit exact-subset lineage for trusted consumed H3 checkpoints. `prepare_pipeline_assets(..., derivation=Path(...), verify_content_hashes=False)` and `prepare_portable_assets(..., derivation=Path(...))` bind a bounded conversion declaration, the pinned source identity, resulting size and immutable local stamps. Derived payloads retain null digests; source hashes are never presented as subset hashes. Schema-3 receipts reopen the same declared subset; ordinary official assets keep their existing contract.

A single RTX PRO 6000 Server completed ten mixed I2VA/L2VA/FL2VA/Ref2VA cases from a cache reduced from 125.084 to 100.451 GiB. Initialization was 120.737 seconds versus a previous-host 120.114 seconds; no startup or quality speedup is claimed. Host RSS remained 109.229 GiB. This qualifies the storage reduction, not arbitrary conversions, smaller host RAM or additional hardware modes. The caller must own a trusted byte-preserving preparation process and immutable storage.


## 0.6.15

Fix the opt-in resident I2VA request check to use the temporal first frame. I2VA intentionally has no Ref2VA `ordered_references`; the previous check rejected valid keyframe requests before inference. Capacity, allocator and hardware restrictions are unchanged. Regression coverage now uses the real `VideoRequest` contract.


## 0.6.13 · Heterogeneous device preview and CPU preparation {#v0-6-13}

Adds explicit SM90/SM103/SM120 profiles, partition-aware discovery, portable immutable asset views and CPU-prepared hybrid tables. Complete I2VA hybrid/Veda measured on H100 SXM and PRO Server/Workstation; other new cards and modes remain unqualified. SM89 defaults and the original Veda name remain available. See [scope and measurements](../guide/profiles#heterogeneous).


## 0.6.12 · Qualified duration and reference expansion {#v0-6-12}

Single-SM89 original-v0.1 hybrid/Veda now supports up to fifteen-second image-conditioned delivery and nine ordered images, within a joint resource bound. Fixes VAE padding at fifteen seconds and extends exact media trimming. Standard profiles, paired/SM86 execution and video references keep their earlier limits; H100 is not qualified. See [complete pipeline](../guide/complete-pipeline#extended-hybrid) and [measured cases](./performance#extended-hybrid).

## 0.6.11 · Native HD and bounded activation lifetimes {#v0-6-11}

Adds explicit native sampling up to a 2048-pixel long edge (2048² maximum), five seconds, with the original-v0.1 single-SM89 hybrid Veda pipeline. Image keyframes and image reference conditioning use the original source; this is not the unreleased official Regenerate-2K model or postprocessing. Ordinary request limits and automatic attention selection are unchanged.

Attention and FFN modulation tensors now have shorter lifetimes. One complete 2048×1152 control reduced peak allocated memory by 30.69%, with all 120 RGB frames unchanged; audio was not sample-identical. Native HD trades substantially higher time and memory for more local detail. See the performance page for measured scope.

## 0.6.10 · Explicit Veda sparse attention {#v0-6-10}

Original LightX v0.1 on one SM89, including the explicit hybrid model, can select
[Veda attention](../guide/veda). The owned predictor and per-request target layout
integrate with the existing two-slot ring; protected connections remain dense,
first/last five blocks use Torch Flash, and actual INT8 sparse execution is reported.
An explicit installer packages the pinned independent upstream core and its notices;
no ComfyUI, model downloads or default change. Dense execution is the rollback.


Version **0.6.9** adds optional complete-video FlashVSR spatial upscaling.

## 0.6.9 · Optional FlashVSR upscaling {#v0-6-9}

`upscale-video` and `vflash.flashvsr.FlashVSR` integrate an explicitly pinned RTX
adaptation of official FlashVSR v1.1 with real sparse local attention. The wrapper
preserves all frames, exact output aspect ratio, rational CFR and source audio,
and saves a separate result. It corrects sample-runner tail truncation and center
cropping; color correction failures are explicit rather than silently swallowed.

Five complete exploratory 2×/4× videos and one public-interface 2× video passed
on RTX 4090 48 GB. The public-interface 960×544/5s case took 43.19s plus 11.18s
load time. [Measurements and quality limitations](../guide/restoration) include
repainted detail and unresolved source anatomy/physics. This is optional, not a
changed H3 default or lossless restoration. Source/wheel only; external runtime
and weights are explicit dependencies, with no automatic download or deployment.

## 0.6.8 · Shared-backbone hybrid references {#v0-6-8}

Opt-in `HybridModel` replaces only official Ref block 25–49 modulation while retaining
the FL trunk, prefix, final layer and original LightX v0.1 adapter. One single-SM89/dense
Python pipeline accepts ordinary keyframes and real one-to-three-image Ref requests;
Ref uses its own source identity and in-memory conditioning, not a generated first frame.
The fixed model adds about 55.4 MiB of modulation tables rather than another backbone.

Ref → I2VA → Ref complete requests passed in one RTX 4090 48 GB session, with full media
decode and bounded cross-mode lifetime checks. Source/provenance, ownership/error cleanup
and unchanged default paths have CPU regression coverage. See
[scope, timings and quality limitations](../guide/complete-pipeline#hybrid-reference-model-in-0-6-8-opt-in).
No claim of universal quality improvement, video-reference support or hybrid Sol qualification.
Source/wheel only; no model weights, new prebuilt image or automatic downstream deployment.

## 0.6.7 · Learned latent lift 6+2 {#v0-6-7}

The complete VDN pipeline accepts `strategy="learned6+2"` with an explicit local
BF16 LBH upscaler. It replaces SelfLift-zero's all-frame VAE round trip, preserves
clean low/high conditioning and the original eight-step video/audio schedule,
and adds no NFE. Missing weights fail clearly; there is no automatic download.
The separate `selflift6+2`, `pixel8+2`, full8 and native H3 profiles remain available.

Three frozen-prefix SM89 controls showed a substantial lift-stage cost reduction
with no new major defect in the inspected video windows; some temporal diagnostics
were worse, and motion/physics and small-face defects remain. See the
[execution scope, observations and timing boundaries](../guide/community-vdn#optional-learned-latent-lift-6-2).
This is explicit, not a changed application default. Source/wheel only; no weights
or updated prebuilt images are included.

## 0.6.6 · Original LightX v0.1 keyframes {#v0-6-6}

Independent preview I2VA/FL2VA profiles pin the original 544p v0.1 four-step adapter,
12/3 schedule and rank128/alpha8 residual. A matching prepared pipeline accepts first-only,
last-only and true two-endpoint requests; it never relabels a v1.0 adapter as v0.1.
Three 960×544 five-second dense requests completed on one RTX 4090 48 GB in about
72–78 seconds excluding initialization. Explicit Sol is supported for these profiles,
but `auto` stays dense: matched Sol controls were about 20–24% slower and did not establish
a compensating sampled quality benefit. Base16 defaults are unchanged.

See [exact contracts, endpoint execution evidence and limitations](./pipeline-profiles#lightx-v01).
This is not a universal face/action repair or a new downstream default. Source/wheel only;
prebuilt images and application deployments require explicit updates. No weights are bundled.

## 0.6.5 · Video-aware SelfLift correction {#v0-6-5}

Select and normalize spatial correction independently at each latent time, rather
than across the whole video. This avoids confounding H3 temporal phase scales with
spatial errors. Three frozen-prefix SM89 controls showed much smaller periodic
soft/sharp jumps with useful detail retained. No frame averaging, extra NFE, model
or schedule change; broader motion and face limitations remain. See
[evidence and limits](../guide/community-vdn#temporal-detail-pulsing-fixed-in-0-6-5).
Source/wheel only; downstream deployments require an explicit runtime update.

## 0.6.4 · Bounded sibling-candidate conditioning reuse {#v0-6-4}

`VDNKeyframePipeline.generate` accepts an optional `ConditioningReuseScope`.
Clean raw text/keyframe features are CPU-cached within that scope; seed-dependent
sampling remains independent. Reuse is off without a scope, bounded to 256 MiB,
and cleared on scope change, execution failure or close. No serving policy changes.
One SelfLift SM89 control removed about 36.8 seconds of repeated encoding;
video RGB matched at the same seed but near-silent PCM was not bitwise equal.
See [evidence and limits](../guide/community-vdn#scoped-clean-conditioning-reuse).
Source/wheel only; prebuilt images and downstream deployments require explicit updates.

## 0.6.3 · Native 544p keyframes and bounded asset preparation {#v0-6-3}

Optional I2VA/FL2VA Turbo8-544 profiles use their own pinned LightX weights, rank/alpha
and 12/3 schedule. One prepared native pipeline accepts first-only, last-only and true
two-endpoint inputs. Three complete five-second requests passed on one RTX 4090 48 GB;
two same-GPU small-canvas controls observed 45.6–47.6% less request time than Base16/Sol.
A separate true FL2VA ten-second request also completed using the same compiled artifact.
See [measurement boundaries and remaining quality limits](./benchmarks#native544).
This does not change Base16/VDN defaults or qualify arbitrary hardware and canvases.

Python preparation/compilation APIs accept explicit `verify_content_hashes=False` for
trusted immutable local snapshots. Full tensor headers, sizes and file identities are
still checked; provenance digests are not represented as newly measured checksums.
Artifact schema 6 and overlay schema 3 permit missing content digests only when the
caller explicitly disables content verification. Strict verification rejects those
artifacts; the existing verified mode remains the default. See [assets](./runtime-assets).
Source/wheel are separate from unchanged prebuilt images; serving pins require explicit updates.

## 0.6.2 · Preview keyframe artifact loading {#v0-6-2}

The native artifact loader now accepts the pinned I2VA/FL2VA adapter identities already declared
by the four SM89 Turbo4/Turbo8 preview profiles. Previously the compiler and profile resolver
accepted them but a later loader allowlist still rejected them as unknown oracle identities.
Regression tests exercise complete artifact loading, not just source-metadata validation.

Adapter revisions, schedules, NFE, precision and hardware checks remain enforced. This is a CPU
contract correction, not new GPU media qualification or a quality/speed claim. It does not enable
the 544p adapter, PDD, tail-only Turbo, change Base16/VDN defaults or replace deployed engines.
Source and wheel are released; existing prebuilt container images are unchanged.

## 0.6.1 · SelfLift-zero progressive I2VA {#v0-6-1}

The complete VDN pipeline accepts `strategy="selflift6+2"`: six low-resolution
evaluations, an all-frame pixel/VAE consistency correction, and two evaluations
at the target resolution. Independent paper-based code requires no new trained
weights and never patches a global sampler. Stage loads and VAE costs are explicit.

The initial scope is five-second I2VA, short side at least 640, on one SM89 48GB.
Three local scene controls showed substantially fewer repeated-contour artifacts
than nearest latent lifting, with a useful detail trade-off against pure pixel
re-encoding. This does not establish universal superiority over full8, solve
small-face defects or inherit the image paper's speed claims. See the
[community guide](../guide/community-vdn) for pinned dependencies and limitations.
Default full8, optional pixel8+2 and the separate native H3 pipeline are retained.

## 0.6.0 · Complete VDN keyframe pipeline {#v0-6-0}

The optional [VDN interface](../guide/community-vdn) accepts a raw prompt and first frame,
or ordered first/last frames, and delivers MP4 through official conditioning and media components.
Choose full-canvas eight steps or pixel-conditioned 8+2 explicitly. The two-pass route encodes
the resized authoritative RGB keyframes independently, avoiding the observed discontinuities
from interpolating high-resolution keyframe latents. It does not cut, drop or blend bad frames.

On one RTX 4090 48 GB, complete fresh-encoding requests delivered 1536×640/5s in 195.0s
and 1280×704/10s in 266.5s. The latter also exercised exact endpoint and silent delivery.
These are bounded execution observations, not matched speed guarantees; residual motion,
face and endpoint-hold defects remain. CPU decoder caching does not mean all models stay on GPU.

Install the pinned upstream runtime described in the guide. This backend is separate from
Sol/Base16 and from the preview LightX4/8 contracts. SM86, L2VA and arbitrary upstream combinations
are not qualified for VDN. No weights are bundled or downloaded implicitly. Source origins,
model licenses and hardware limits remain explicit. The established Base16 interface is preserved.

## 0.5.6 · Complete-pipeline attention adapter {#v0-5-6}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.5.6) · [Usage and limits](./attention-lora)

The complete Python/CLI pipeline now accepts the same explicit FP32 DiT-only attention
adapter as the native context. A prepared pipeline loads it once, reuses it across serial
requests, records its scope/rank/scale in each result, and detaches it before releasing the
native core, including on failure. All three CLI adapter options must be supplied together.

This is a single-SM89 official Base16 interface. It does not merge base weights, download
adapters, enable an adapter by default, or add HTTP adapter configuration. TokenRefiner,
SM86 and cooperating-pair application remain outside the supported adapter scope.
The VAE overlap correction, exact text-encoder prefix and Sol selection policy are retained.
Package and runtime versions are checked together to prevent stale reported versions.

Three serial 672×384, six-second, 24 fps complete requests passed on one RTX 4090 48 GB,
using Base16/Sol and an explicit rank-8, scale −1 adapter. They reused one owner, delivered
144 frames each and closed cleanly. A cached candidate and its same-seed uncached control
produced identical decoded RGB and PCM; the hit avoided the conditioning encoder call.
This bounded cache control took 107.1 versus 115.8 seconds, not a general speed guarantee.
CPU failure-path coverage and minimal-install CI also pass. Existing native quality evidence
remains scoped; residual face/motion defects are not claimed solved.
Source and wheel are released; prebuilt registry images remain 0.3.2.

## 0.5.5 · H3 spatial VAE composition {#v0-5-5}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.5.5)

The owned media decoder composes horizontal tile strips before vertical crossfades,
preserving the diagonal contribution at overlap intersections. Model weights, decoded
tile inputs, local position coordinates, precision, temporal chunks and audio are unchanged.
The correction is instance-local and applies only to the single-device media decoder;
the denoiser may still use its existing multi-device profiles.

Frozen-latent RTX 4090 48 GB checks cover 5/6-second landscape clips and a 10-second
864×864 clip, including optional-adapter output. They show small pixel changes, not a
general cure for face distortion or high-motion ghosting. Decode/assembly cost of the
large clip remained about 56–58 seconds; this is not an end-to-end speed claim.
The overlap defect has also been reported in
[ComfyUI's independent correction](https://github.com/Comfy-Org/ComfyUI/pull/16436).
Sequential triple-overlap weights are not normalized all-contributor overlap-add.
The public adapter additionally reproduced every qualified raw RGB pixel of a complete
five-second clip on SM89. This release has no new SM86 GPU measurement or independent
audio determinism claim. No new weights, precision profile or decoder option is needed.
Source and wheel are released; prebuilt registry images remain 0.3.2. Serving pins do not change.

## 0.5.4 · Optional FP32 DiT attention LoRA {#v0-5-4}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.5.4) · [Interface and limits](./attention-lora)

The low-level `apply_dit_attention_lora` context attaches caller-provided FP32 PEFT attention
weights to an exclusively owned single-SM89 Base16 runtime. It validates the complete layout,
applies only the 50 DiT blocks, binds logical layers correctly across ring slots, and restores
methods on exit. Explicit signed scale includes alpha/rank; do not negate an update twice.
Base weights, conditioning, scheduler and generation/restoration defaults remain unchanged.
There is no new CLI, pipeline or HTTP loading option, automatic download or bundled model.

A same-input 672×384, six-second, 24 fps RTX 4090 48 GB control completed both videos and
preserved the old implementation's final AV latents exactly, separately for Base and reverse.
This is migration evidence, not proof of universal quality, speedup or full-adapter equivalence.
TokenRefiner is intentionally not applied; other architectures, multi-GPU and step schedules
are outside this interface. Face/action artifacts can remain. Model licenses are separate.
Source and wheel are released; prebuilt registry images remain 0.3.2. Serving pins do not change.

## 0.5.3 · Explicit restoration acceleration {#v0-5-3}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.5.3) · [Measured scope and limits](../guide/restoration)

Optional `resident` placement avoids repeated model transfers. Optional SM89
`sage-int8-fp16` attention uses a separately installed, ABI-compatible SageAttention 2.
CLI/Python report both selections. Defaults remain native attention and CPU offload;
H3 generation, ten-step restoration, motion segments and original audio are unchanged.
Residency preserved exact decoded RGB on a 17-frame window and two complete normal
clips. Sage is approximate: one full ten-second run observed 25.4% less restoration
compute, with differing total timing boundaries. Selected-frame review found no
additional obvious structural failures, but existing blur/repainting remained.
These are bounded exploratory results, not universal quality or service-speed guarantees.
Source/wheel are released; prebuilt registry images remain 0.3.2.

## 0.5.2 · Optional video enhancement {#v0-5-2}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.5.2) · [Usage and evidence](../guide/restoration)

`vflash restore-video` and `vflash.restoration.restore_video` use a caller-supplied,
trusted STCDiT-tiny runtime and local weights. The owned loader, complete motion
segmentation, step progress/cancellation and atomic output delivery keep the
original video. Original audio packets are copied; enhanced H.264 video is lossy.
The supported input boundary is 24 fps, up to ten seconds and one mebipixel on a
32-aligned canvas. This is not a guarantee of restoration quality or all-GPU capacity.

Restoration is off by default and may soften or repaint details. Existing H3
Base16, Sol/dense policy, conditioning, weights and delivery remain unchanged.
Install `[pipeline,restoration]` and explicitly trust the supplied runtime; no
third-party source, weights, product accounts or service scheduling are vendored.
Source/wheel are published separately from prebuilt registry images, which remain 0.3.2.

## 0.5.1 · Exact H3 text-encoder prefix {#v0-5-1}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.5.1) · [Measured scope](./benchmarks#encoder-prefix)

The pinned H3 conditioner reads `hidden_states[50]`. Retaining 51 of its 64 decoder layers
preserves that raw intermediate state and avoids executing the unused tail. Keeping only 50
would incorrectly substitute the final normalized state. The vision encoder, retained weights,
precision, denoiser, scheduler, Sol policy and audio/video delivery are unchanged.

First-frame and last-frame A/B/A2 checks preserved all 14 conditioning tensors on both SM86
and SM89. A complete SM89 eight-second A/B/A2 preserved every delivered RGB pixel and decoded
PCM16 sample. Local warm capture improved by about 18–19%, but the complete-video difference
was small: denoising still dominates. This is not an 18% whole-video speedup or a repair for
existing generation defects. Removing the tail releases references to 6.34B BF16 parameters;
it is not a measured RSS reduction or a smaller model download.

No new model files or configuration option are required. Source and wheel are released;
prebuilt registry images remain 0.3.2. Existing serving deployments keep their explicit pins.
The 0.5.0 approximate single-SM89 Sol default and its quality limits remain unchanged.

The default test suite now hides CUDA devices; hardware tests require explicit allocation and
opt-in. A stream-order race in the profiling test's artificial zero-filled slots was removed
by matching the runtime's uninitialized slots. No production ring arithmetic changed.

## 0.5.0 · Integrated single-SM89 Sol default {#v0-5-0}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.5.0) · [Sol evidence and limits](./sol-engine-alignment)

`auto` now selects approximate `sol-sm89` for official Base16 on one SM89 GPU. SM86, cooperating
pairs and Turbo profiles retain dense `torch-flash`. Python complete/native sessions, `generate`,
`denoise`, `plan` and the native HTTP service use the same selection rule. Explicit `torch-flash`
preserves dense behavior. Sol uses serial block-ring residency and reports `exact=false`, actual
operator calls and protected modality rows; it does not silently fall back on dependency failure.

`profiles` and `/v1/profiles` now name the model's dense policy `baseline_attention`; it is not
the executed selection. Use `plan` or `/readyz`'s `attention_selection` for resolved configuration,
and completed-job runtime metadata for actual execution. Update clients of those inspection fields.

Standard `runtime` and `pipeline` Docker builds include fixed Sol 0.5.0 dependencies and the reviewed
four-call CUTLASS stream ABI patch. Python users install GPU/pipeline extras, then run
`python -m vflash.install_sol`. The former separate `pipeline-sol-sm89` target is removed.
Model weights are unchanged and remain separate licensed inputs; no product configuration is packaged.

The unchanged Sol kernel reuses earlier SM89 hardware evidence: one fixed ten-second 736 × 992
Base16 A/B/A reduced local complete-request time by 19.584% and denoising by 21.078%, with 0.070%
control drift and profiling enabled. Each Sol request made 800 real operator calls. This is not a
new 0.5.0 benchmark, a universal speedup or a same-quality claim. Motion and audio-content quality,
unprofiled accepted throughput and first-use compilation remain separate limitations.

This release also includes the post-0.4.0 1,048,576-pixel validation/conditioning budget and explicit
`silent-v1` media delivery. They do not expand every GPU's measured capacity. Source archives and
a wheel are published; the historical 0.3.2 registry images are not relabeled as 0.5.0. Build current
Docker targets from this tag. Existing deployments keep their pinned version until explicitly upgraded.

## 0.4.0 · Base16 keyframes and measured Sol-Engine alignment {#v0-4-0}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.4.0) · [Sol-Engine alignment](./sol-engine-alignment)

One prepared Base16 pipeline can now process I2VA, L2VA and FL2VA requests without reloading the official transformer. It accepts integer durations from five through ten seconds at 24 fps and canvases up to 1,032,192 pixels, with dimensions divisible by 32 and aspect ratios from 1:4 through 4:1. Conditioning stays in memory by default; persisted bundles remain available for explicit replay. Exact endpoint delivery and bounded audio delivery are part of the owned media stage. Turbo and reference-video profiles retain their separate five-second contracts.

Matching RTX 4090 48 GB devices can opt into `sequence-head` with block streaming for a single Base16 request. On one fixed ten-second, 736 × 992 L2VA workload, the paired path took 484.232 seconds versus 774.153 and 773.515 seconds in bracketing single-device controls. All three MP4 files were byte-identical. The 37.4% latency reduction costs 25.2% more aggregate device time than two independent workers, so the pair is only a latency choice when its peer would otherwise be idle. Other SM89 profiles and `tensor` remain rejected.

The cooperative path replaces materialized QKV and returned-attention layout chains with exact Triton destination-major copies. At a representative 55,413-token shape, the QKV copy was 2.429× faster on SM86 and 1.952× faster on SM89; returned-head merge was 2.027× and 1.932× faster respectively. QKV and head-merge operation peaks fell by 1,136.4 and 378.8 MiB. Every compared element matched. Final unprofiled complete A/B/A requests improved by 1.503% on dual SM86 and 0.472% on dual SM89, with identical compressed video streams. These are useful exact memory-layout gains, not a large complete-video speedup; the workloads and media boundary are recorded in the [performance guide](./performance#direct-relayout).

The implementation was compared with NVIDIA Sol-Engine at pinned revision [`ca26dbd`](https://github.com/NVlabs/Sana/tree/ca26dbd2b7034cc90a64c093d715d99b0bfa5b7f). Vflash adopts only the exact relayout mechanism for its SM86/SM89 contract. AdaLN precomputation and strict fused operations already existed in the engine. SOL/BSA attention, cross-step caches, INT8/FP8 communication, SM100-only MXFP8 compute and the four-step FastH3 adapter do not enter the exact defaults. Upstream B300 or 50-step results are not presented as Vflash speedups.

The release also adds opt-in denoising phase attribution, live-conditioning capture metrics, fail-closed ring-copy validation and explicit keyframe-mode provenance. Its evidence is bounded to the listed target GPUs and workloads; it does not establish every duration/canvas combination or broad semantic and audio quality. Model weights remain separate licensed inputs. Version 0.4.0 is released as source and a wheel; the published 0.3.2 container images remain the latest prebuilt images until a separately qualified image inventory is published.

## 0.3.2 · wide tensor offsets and dual-SM86 T2VA {#v0-3-2}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.3.2) · [Images and package identities](https://github.com/Hansimov/vflash/blob/v0.3.2/docker/images.json)

Large token/stride products can exceed a signed 32-bit element offset. Eight fused kernels now select 64-bit address arithmetic when the actual shape and stride require it, casting before multiplication. Small inputs keep their 32-bit path; BF16 rounding, LoRA arithmetic and schedules are unchanged. This fixes an invalid-memory-access cause, not a new quality or speed feature.

Fifteen small/wide target-GPU operator checks matched bit-for-bit, including wide comparisons against safe chunks of the original arithmetic. Saved 243-frame conditioning then completed all eight Ref evaluations on one RTX 4090 48 GB, producing finite FP32 video/audio latents and fully decoded 1344 × 768 media: 240 frames at 24 fps, with stereo 32 kHz audio. Resource owners closed and the device was released. This was a native/core-and-media check using saved conditioning, not a newly encoded `H3Pipeline` request or a full-trajectory bitwise comparison. The application integration revision was `6310e023fc31caa4b70bf9a5c64a07c062a4c343`.

The new `t2va-turbo4-exact-sm86` native profile uses two RTX 3080 20 GB GPUs, `sequence-head` and block streaming. Its pinned Base4 v1.0 adapter uses four evaluations and video/audio shifts 6/3. The SM86 compiler passed 52 same-architecture official timestep/modulation comparisons and produced a complete 50-block artifact. SM89 or Ref artifacts cannot be relabeled for this profile.

The installed public native session completed application-owned encoding→core→media requests at 928 × 512 for five seconds and 640 × 352 for ten seconds, both 24 fps. Each returned 14 finite conditions, FP32 audio/video outputs, a fully decoded MP4 and confirmed close. Twelve sampled frames per output supported the fixed example's visible object/action requirements; audio semantics and unsampled motion remain unjudged. Thermal throttling occurred, so these observations do not establish clean timing or broad quality.

The new dual-SM86 T2 `H3Pipeline` wrapper was not rerun on GPU and its public temporal contract remains five seconds. The long Ref check also does not add a ten-second wrapper API. Single-SM86 T2VA, `tensor` T2VA, T2VA Turbo8, SM86 video references and arbitrary long/high-resolution combinations remain unqualified. Existing five-second pipeline evidence is retained at its original scope.

Both images retain all 0.3.1 dependency layers and entrypoints. Their installed 59-file package, CLI, service construction, writable cache and pinned adapter/media imports passed CPU checks. The fused module matches the target-tested implementation exactly. The wheel and reproducible small-layer recipe are [release attachments](https://github.com/Hansimov/vflash/releases/tag/v0.3.2); model weights remain separate licensed inputs.

## 0.3.1 · lower QKV memory on 4090 {#v0-3-1}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.3.1)

The strict Ref4 path on RTX 4090 uses fewer temporary tensors when combining its LoRA projections. It selects the optimization automatically and keeps existing model assets and API calls compatible. Other profiles retain their existing implementation.

A complete 50-layer, four-step native request preserved the final FP32 video and audio latents exactly. The kernel also passed a same-device operator comparison. These checks cover numerical behavior and local temporary memory; this update does not establish a new prompt-to-MP4 latency figure.

## 0.3.0 · generate from a reference video {#v0-3-0}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.3.0)

`H3Pipeline` and `vflash generate --reference-video` now accept a local 2–5 second clip. On one SM89 48 GB GPU, the existing Ref4 model can process images, then video, then images again without reloading. Source audio is discarded; the result contains newly generated audio. Output remains five seconds at 24 fps. Mixed image/video inputs, video references on SM86, and Ref8 video references are outside this release. [Input limits and examples](../guide/complete-pipeline#reference-video).

Construction and input validation run on the CPU before the first model load. Services can call `prepare()` to preload a fixed pipeline before accepting requests. This loads model stages, not every shape-specific operator. Repeated calls preserve ownership; invalid input leaves a healthy loaded instance usable. Timing separates input preparation, model loading, encoding, denoising and media delivery.

An installed wheel completed one fixed image/video/image sequence. The video request used a five-second 928 × 512 reference: all 14 conditions and both final FP32 latents matched independent controls, and all 120 delivered RGB frames matched. The final image matched the first image's conditions, latents and decoded video. A subsequent first-evaluation cancellation published no partial video, removed temporary files, and released the three model owners. GPU resources were relinquished after process exit. Existing text and dual-SM86 image computations remain unchanged.

Video references add substantial encoding and denoising work. The representative request recorded about 64 seconds encoding, 191 seconds native execution and 19 seconds media delivery, including diagnostic observers; model preload was measured separately. These single observations are not a latency promise or a speed comparison. Original-brief review supports a useful guided variation, with some timing and pose details partial. It does not qualify precise editing or general prompt adherence. Repeated official audio decoding differed slightly before encoding; audio bitwise reproducibility and subjective audio quality are not claimed.

The containers provide writable Jiterator and Inductor caches, including for an arbitrary UID with an empty writable cache mount. [Image identities and installation checks](https://github.com/Hansimov/vflash/blob/v0.3.0/docker/images.json) bind the final package to the validated implementation.

## 0.2.2 · lower FFN memory on 4090 {#v0-2-2}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.2.2)

SM89 Ref4 now combines the FFN LoRA merge and SiLU activation in one kernel. It preserves the original BF16 rounding after adapter scaling, addition and activation, while removing a large intermediate tensor. The default strict Ref4 path selects it automatically; model assets and generation commands stay compatible. SM86, Base4 T2VA and Ref8 retain their previous kernels.

A same-device native comparison measured a median **43.033 → 42.569 seconds** over three warm runs per implementation, a **0.464-second (1.08%)** reduction. Peak denoising allocation fell by 597 MB on that fixed workload. See [the workload and measurement boundary](./benchmarks#sm89-ffn). These are native-engine results, not an end-to-end speed claim.

The same kernel then passed an original/fused/original sequence in a persistent complete pipeline, using three ordered references at 928 × 512. All 14 conditioning tensors, final FP32 video/audio latents, 124 raw video frames and 120 delivered RGB frames matched exactly. Each MP4 decoded fully with five-second stereo audio. Cancellation after the first denoising evaluation retired the pipeline, removed partial output and released owned storage. The three-reference denoising peak fell by 517 MiB; the observed maximum across pipeline stages stayed unchanged because another stage dominated. Repeated official audio decoding can still vary slightly; this is not a bitwise audio or broad generation-quality guarantee.

The public implementation uses a direct method call with no experiment hooks. Installed-package and CPU dispatch checks bind it to the qualified kernel; the other native and pipeline computations are unchanged. The [versioned image inventory](https://github.com/Hansimov/vflash/blob/v0.2.2/docker/images.json) records source, wheel and immutable container identities.

## 0.2.1 · complete video on two 3080s {#v0-2-1}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.2.1)

The complete Python/container pipeline now supports Ref4 on two RTX 3080 20 GB GPUs. Compile `ref2va-turbo4-exact-sm86` assets on an SM86 GPU, then select both devices and `sequence-head` for generation. One to three ordered images produce a five-second, 24 fps MP4. Encoding and decoding use the primary; native denoising uses both. The [profile recipe](./pipeline-profiles#sm86) gives the commands. Single-SM86 and `tensor` complete pipelines are rejected before model loading; their native latent interfaces are unchanged.

An actual installed container completed a representative three-reference request at 928 × 512. All 14 conditioning tensors matched an independent official capture on the same primary SM86 GPU. Final FP32 audio/video latents matched a standalone native session on the same pair. All 124 raw video frames and raw audio were finite, and the delivered MP4 decoded fully with 120 frames and five-second stereo audio. Cancelling a second request after one denoising evaluation retired the pipeline and removed temporary output; both devices released owned resources.

The native implementation is byte-identical to 0.2.0. This release changes profile admission and the complete-pipeline strategy guard; the SM86 compiler arithmetic also passed 52 comparisons against same-architecture official modules. Together these checks qualify the wrapper, numerical binding and resource lifetime for the measured workload. They are not a new official full-DiT comparison or a broad instruction-quality or speed claim. The run included thermal throttling; its timings are not an isolated performance result. Host RSS peaked at 114.67 GiB under a 240 GiB budget, with device-wide peaks of 11.05/6.85 GiB. Leave deployment headroom beyond this measured case.

SM89 Ref4 and T2VA retain their existing qualification and strict original audio delivery. The release also corrects older documentation that described the public package as latent-only. Model weights remain separate licensed downloads. Published image digests and package identity are recorded in [the versioned inventory](https://github.com/Hansimov/vflash/blob/v0.2.1/docker/images.json).

## 0.2.0 · complete text-to-video {#v0-2-0}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.2.0)

The Python and container pipelines now generate a five-second, 24 fps MP4 without reference images. Select `t2va-turbo4-exact-sm89`, prepare the fixed Base4 v1.0 assets and omit `--reference`. Ref4 keeps its one-to-three-image interface. Each mode owns separate prepared assets and a persistent pipeline; mismatched requests are rejected before execution. The [model-profile guide](./pipeline-profiles) shows the complete T2VA commands.

The official-weight compiler now creates Base4 assets as well as Ref4 assets. On SM89, all 50 compiled Base4 blocks, four schedule tensors and nine auxiliary tensors matched independently prepared BF16 assets exactly. The actual installed container then completed two requests for one fixed 928 × 512 case: all 14 conditioning tensors and final FP32 video/audio latents matched independent controls. Every repeated raw video frame matched; both MP4s decoded fully with 120 frames and five-second stereo audio. Cancelling another request after its first denoising evaluation removed temporary output and retired the pipeline.

These checks establish implementation and resource lifetime for this workload, not broad instruction or audio quality. The Ref4 path retains its 0.1.0 multi-reference qualification. Small repeat differences in the official audio VAE remain possible. Both complete profiles use BF16, pinned LoRAs and strict original audio delivery; this release does not include W8, dynamic LoRA or first/last-frame generation. Complete SM86 generation is not yet qualified and is rejected; its native interfaces remain available.

The images also provide a writable Inductor cache for arbitrary user IDs. Final image checks cover installed package identity, supported profile contracts and CPU startup; the cache-directory and support-list changes preserve the GPU-qualified numerical implementation.

Published Linux AMD64 images: [`hansimov/vflash:0.2.0-pipeline`](https://hub.docker.com/r/hansimov/vflash/tags) for complete generation and `hansimov/vflash:0.2.0` for the native HTTP service. Both support anonymous pulls. The [immutable digest inventory](https://github.com/Hansimov/vflash/blob/v0.2.0/docker/images.json) records their source and qualification. Model weights are separate downloads under their own licenses.

## 0.1.0 · multi-reference video generation {#v0-1-0}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0)

Use `vflash generate` or `H3Pipeline` to produce a five-second, 24 fps MP4 from a prompt and one to three images on an RTX 4090 48 GB. Images keep their supplied order and `<Picture N>` labels. The original single-image Python argument remains compatible. `vflash prepare-pipeline` verifies model assets once before use. The [container guide](../guide/docker#pipeline) includes complete preparation and generation commands.

The actual installed pipeline image passed three sequential complete GPU requests with one, two and three references at 928 × 512. All 14 conditioning tensors and final FP32 video/audio latents matched their fixed controls. The first case also matched every raw FP32 video frame; the second matched all 120 delivered RGB frames. All three MP4s decoded fully with 120 frames and a five-second stereo audio track. Cancelling a three-reference request after its first denoising evaluation retired the pipeline, removed temporary output and released its resources.

These are implementation and lifecycle checks, not a broad instruction-quality or speed guarantee. The denoiser, LoRA arithmetic and official media decoding are unchanged from a7. Raw audio may vary slightly across repeated official VAE calls; audio bitwise reproducibility is not promised. References do not impose strict keyframe timing.

The complete image targets Ref4 on SM89. Ref8, T2VA Turbo4 and single/dual SM86 profiles retain their native bundle-to-latent interfaces. W8, dynamic LoRA and FL2VA are outside this release. Model weights remain separate downloads under their own licenses.

Published Linux AMD64 images: [`hansimov/vflash:0.1.0-pipeline`](https://hub.docker.com/r/hansimov/vflash/tags) for complete generation and `hansimov/vflash:0.1.0` for the native HTTP service. Both support anonymous pulls. The [immutable digest inventory](https://github.com/Hansimov/vflash/blob/v0.1.0/docker/images.json) identifies the exact images.

## 0.1.0a7 · prompt and image to MP4 {#a7}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a7)

A prompt and one image can now produce a five-second, 24 fps MP4 on one RTX 4090 48 GB. The Python pipeline owns the native denoiser and stage lifetimes, with explicit pinned official text, reference and VAE adapters. It supports sequential reuse, progress and cancellation. [Generate a video](../guide/complete-pipeline).

The [Ref4 compiler](../guide/compile-weights) builds native assets from fixed official H3 weights and Turbo4 v0.1, without a captured request or another project. Base weights and LoRA identities remain separate. All 1,250 layer tensors, four schedule tensors and nine auxiliary tensors matched the qualified BF16 assets exactly.

Complete checks reproduced conditioning and native latents exactly, with matching decoded video. Small variations in the official audio VAE remain under investigation; this is not a bitwise audio or broad quality guarantee. Total request timing now includes input preparation and cleanup, while weight transfers and stage calls are reported separately.

The complete pipeline and compiler initially cover Ref4 on SM89. T2VA Turbo4, Ref8 and SM86 keep their existing bundle-to-latent interfaces. Dynamic LoRA, FL2VA and W8 are not released. The Docker HTTP service also remains a latent interface. Model weights are downloaded separately under their own licenses.

## 0.1.0a6 · text-to-video {#t2va}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a6)

SM89 now supports T2VA Turbo4 with Base4 v1.0 assets. The engine checks the task, adapter identity and conditioning before execution, so Ref assets cannot accidentally process a text-only request. Applications can keep separate Base and Ref sessions to avoid loading different weights for each request.

One decoded-output case and repeated session calls passed implementation checks. Read the [validation scope](../guide/profiles#t2va) for the workload and remaining quality limits. The Ref profiles retain their existing weights and arithmetic. Newer Base4 adapters, T2VA Turbo8 and T2VA on SM86 are not part of this release.

Install this source tag using [the quick start](../guide/getting-started). The [Docker guide](../guide/docker) builds `vflash:0.1.0a6` locally; that version had no public prebuilt image.

## 0.1.0a5 · loading and resource ownership {#a5}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a5)

- Read groups of tensors from each immutable file into their final owned CPU storage. This avoids repeated header parsing and intermediate copies during loading.
- Release session-owned weights and pinned storage when the session closes, including when the caller retains the closed Python object.
- Report failed device or communication cleanup as a failure. Close a failed session and stop its worker if cleanup cannot be confirmed.

The model weights, Turbo adapters, denoising schedule and BF16 arithmetic are unchanged. These changes improve loading and resource lifetime; they are not a new cross-workload speed or quality claim.

Install from this tag for a fixed source release:

```bash
git clone --branch v0.1.0a5 --depth 1 https://github.com/Hansimov/vflash.git
cd vflash
python -m pip install -e .
```

The [Docker guide](../guide/docker) builds the tagged source locally. That version was distributed from source without a public prebuilt image. Keep using assets that match your profile; this update does not convert weights or conditioning bundles.

## 0.1.0a4 · lower host allocation {#a4}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a4)

Block streaming packs tensors into shared segmented pinned storage to reduce unused allocation space. Failed loads release their partially built resources. See the [4090 measurement](./benchmarks#sm89-host-memory) for its scope and numerical comparison.

## 0.1.0a3 · explicit memory placement {#a3}

[Source tag](https://github.com/Hansimov/vflash/tree/v0.1.0a3)

Added explicit block streaming on a 4090 and completed-step callbacks for integrations. The [a3 dual-3080 measurements](./benchmarks#sm86-parallel) retain their original version and workload instead of being relabeled as current-release results.

## Availability {#availability}

The package provides the listed **BF16 Ref2VA Turbo4/Turbo8, SM89 T2VA Turbo4 and dual-SM86 T2VA Turbo4 denoisers**, plus the **Ref4 Python pipeline on SM89 or dual SM86, T2VA on SM89, and Base16 I2VA/L2VA/FL2VA on SM89 or dual SM86**. Matching SM89 devices have an opt-in cooperative Base16 latency path. The official-weight compiler creates Ref4 and Base4 assets for SM86/SM89; Base16 keyframe profiles use their fixed official transformer assets. The latest prebuilt containers remain 0.3.2 and therefore retain that release's narrower interface. Model weights are not bundled.

New modes, adapters and hardware require their own installation, numerical and decoded-output checks before entering the support table.
