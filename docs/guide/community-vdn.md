# VDN8 community sampling adapter

`vflash.adapters.vdn_h3.VDNEngineSession` is an **opt-in latent backend**, with
an explicit `VDNKeyframePipeline` complete adapter. Neither replaces the default
`H3Pipeline`. They run the trained VDN hybrid model,
not the original H3 weights with an interchangeable attention switch.

The current configuration is one SM89 GPU with 48 GB VRAM, rowwise FP8, eight
trained steps, 50 resident DiT blocks and streamed TokenRefiner. Local complete
I2VA and true FL2VA experiments cover 5–10 seconds. This is not a qualification
for every geometry, other GPU architectures, Ref2VA, or a production default.

## Scoped clean-conditioning reuse

For sibling candidates executed serially by the **same pipeline**, pass the same
caller-owned `ConditioningReuseScope` to `generate`. Use a new opaque scope for
each accepted request and owner; never accept the scope directly from an untrusted
client. Omitting the scope disables reuse and clears retained inputs.

```python
from vflash.pipeline import ConditioningReuseScope

scope = ConditioningReuseScope("opaque-accepted-request")
first = pipeline.generate(first_request, Path("first.mp4"), conditioning_reuse_scope=scope)
second = pipeline.generate(second_request, Path("second.mp4"), conditioning_reuse_scope=scope)
```

Only raw encoder features and clean VAE anchors are retained in CPU memory, not
TokenRefiner output, DiT states or noise. A different seed keeps independent
sampling; changed prompt, ordered canonical RGB pixels, geometry, model path or
sampling plan misses the cache. Assets must remain immutable. The cache is bounded
to two entries and 256 MiB; one-hour idle expiry is checked on access, and scope
change, generation failure and `close()` clear it. It is not a cross-process cache
and does not accelerate the first candidate on another GPU.

A local single RTX 4090 48 GB, SelfLift 6+2, 1536×640/5-second three-video control
reduced conditioning from 36.954 s to 0.128/0.114 s, retaining 25.3 MiB. Same-seed
decoded RGB matched across every frame; the next seed changed the video. The
near-silent audio was **not bitwise equal** (maximum 6 PCM16 units, RMS difference
0.581), so this is not a complete audio/video equivalence guarantee. Total times
231.334/147.019/150.101 s also include cold/warm decoder and compute differences;
only the roughly 36.8 s removed encoding is attributed to reuse. Rich-audio quality
and other strategies have not received the same GPU control. The reusable core
was promoted from integration source `83512f70`; no private media are distributed.

## Explicit dependencies and assets

Use an isolated environment with the exact revisions below. No code or model
download happens inside the adapter; do not replace another application's
Diffusers installation in place.

- [FreeVideo Apache-2.0 runtime](https://github.com/FlashML-org/FreeVideo/tree/5878b005ecf7f65c8271d1fae0a1069cbc713245).
- [VDN](https://github.com/OpenVDN/vdn-minimax-h3) revision `30b6b380c2482f3519469350810c2955d8847fd9`
  and its patched Diffusers tree `37068ab7331d8b28f4cf718dba7015c742a306d2`.
- [VDN H3 weights](https://huggingface.co/OpenVDN/vdn-minimax-h3), under the
  MiniMax community model license, separate from the runtime's Apache license.
- The FreeVideo SM89 rowwise pack, with local `rowwise/cache`, `config/h3-base`
  and `config/stage-dmd-step-250` directories. Acquire and verify the immutable
  source manifest before constructing a session.

Expose `freevideo_engine` on Python's import path and set `FREEVIDEO_VDN_ROOT` to
the explicitly installed VDN checkout. Local experiments used Python 3.11 and
Torch 2.11; upstream's declared Python 3.12/Torch 2.13 environment is distinct.
No ComfyUI/GPL integration is copied into Vflash.

## Caller-owned conditioning and delivery

```python
from pathlib import Path
from vflash.adapters.vdn_h3 import VDNEngineSession

engine = VDNEngineSession(
    Path("models/vdn"), task="i2va",
    canvas={"width": 1088, "height": 448, "frames": 124},
)
try:
    video, audio, report = engine.sample(Path("clean-input.pt"), seed=17)
finally:
    engine.close()
```

The trusted local conditioning file uses the upstream format: `prompt_embeds`
(raw official encoder layer-50 output), `text_token_tags`, `keyframe_anchors`
(`first`, or ordered `first,last`), and clean `condition_latents`. Never reuse
another adapter's refined text or already-noised keyframes. FL2VA text encoding
also sees both reference images. The caller provides official VAE/media stages,
requested 24 fps duration, audio policy and final file publication.

## Optional eight plus two

Create the upstream two-pass plan and use `first_pass_reference(rgb, plan)` for
each target-sized reference **before separately VAE-encoding the smaller
canvas**. Supply that conditioning file as `first_pass_conditioning` and an
explicit `upscale(video, width, height) -> (video, report)` callable to `sample`.
The same session performs eight small-canvas steps, latent upscaling/alignment
crop and the trained final two steps. It restores target geometry even on error,
uses restart seed +1, and retains the upstream first-pass audio policy.

Missing small-canvas conditioning is an error, not a fallback to interpolating
high-resolution keyframe latents. That interpolation caused a visible opening
appearance discontinuity in two local cases; RGB reduction followed by VAE
encoding improved those windows. No output frames are pasted over or removed.
The local upscaler was an explicit BF16 variant with additional channel
normalization, **not** upstream's FP16 file. Reports do not falsely inherit an
upstream checkpoint digest for a caller-supplied model.

Local single-SM89 1280×704/10-second FL2VA measurements were 363.397 seconds for
full-canvas eight steps and 240.029 seconds for RGB-conditioned eight plus two,
including keyframe encoding, load, sampling and decode, but excluding cached
text encoding and resource handoff. Runs were not cold/warm matched. Motion
trajectories differ; the observed improvement is not universal quality parity
or a formal speed guarantee. Keep full-canvas eight as an explicit comparison.

Calls are serial and non-reentrant; close after use. Keep inference tensors
writable for asynchronous staging (`no_grad`, not outer `inference_mode`).
The adapter performs no account scheduling, quota changes or automatic retries.

The RGB conditioning and thread-safe staging mechanisms were exercised by the
private integration at source revision `92bd02c7`; only generic code and aggregate
measurements are included here. The packaged serial adapter completed a separate
1536×640/5-second two-pass GPU run; its video and audio latent tensors matched the
earlier standalone execution exactly. That is a migration check, not a requirement
that improved algorithms match a baseline. No private media are distributed.

## Complete prompt and keyframe input

The complete adapter consumes `VideoRequest` and returns `VideoResult`; it does
not label the new model as Base16. Fresh prompt/keyframe encoding through MP4
completed on single SM89: 1536×640/5s in 195.0s and true FL2VA 1280×704/10s in
266.5s. The latter exercised explicit exact endpoints and silent delivery.
These bounded trials do not qualify every input or imply all-model residency.

```python
from pathlib import Path
from vflash.pipeline import VideoRequest
from vflash.pipeline.vdn import VDNAssets, VDNKeyframePipeline

pipeline = VDNKeyframePipeline(
    VDNAssets(
        official_model=Path("models/official-h3"),
        weights=Path("models/vdn"),
        decoder=Path("models/official-decoder"),
        upscaler_checkpoint=Path("models/upscaler-bf16.safetensors"),
    ),
    strategy="pixel8+2",  # explicit; full8 remains the default
    trust_local_code=True,
)
try:
    result = pipeline.generate(
        VideoRequest(
            prompt="An uninterrupted shot of a bird taking flight.",
            first_frame=Path("first.png"),
            width=1280, height=704, duration_seconds=5,
        ),
        Path("output.mp4"),
    )
finally:
    pipeline.close()
```

Both keyframes are center-cover resized to the same target geometry before text
and VAE encoding. They remain ordered for FL2VA. `silent-v1` and explicit
`exact-v1` delivery use the existing media layer; default `decoded` does not paste
over endpoints. Output is published only after complete MP4 encoding, without
overwriting another file. Temporary conditioning is removed.

Without a reuse scope the adapter reloads text/keyframe encoding per request;
sampling is always independent. It retains the CPU media decoder between serial
calls. Explicit scope reuse is described above; neither is full-model residency.
Do not use cached-text timings as uncached request latency.
Product routing, capabilities, configuration and rollout remain separate work.

## Optional SelfLift-zero 6+2

`VDNKeyframePipeline(..., strategy="selflift6+2")` selects an independently
implemented [SelfLift-zero](https://arxiv.org/abs/2609.02036) transition. It runs
six original-schedule evaluations at half spatial resolution, corrects the clean
prediction using an all-frame pixel/VAE round trip, and continues the original
video/audio clocks for two full-resolution evaluations. The correction selects
the 60% largest channel-mean spatial disagreements **independently at each latent
time**, at unit strength. This is **eight**
NFE, not a completed eight-step low-resolution clip plus two restarted steps.

The initial supported scope is five-second I2VA with short side at least 640 on
one SM89 48GB. FL2VA, tail-only input, smaller canvases and longer durations are
rejected by this strategy; full8 and pixel8+2 keep their separate contracts.
No learned upscaler is required. VAE decoding/encoding is additional real work;
the report includes it and both stage loads. Engines are closed before the VAE
round trip and recreated for the suffix. No module-global sampler is patched.

Three local animation/live-action/action cases showed substantial removal of
the repeating contours caused by direct nearest latent lifting. Compared with
correcting the entire latent from pixels, selective correction retained more
face/fabric detail in the inspected live-action crops. This is not proof that
every scene improves over full8: motion differs, small faces remain imperfect,
and this is an H3 adaptation, not one of the paper's evaluated image models.
Keep it explicit and compare complete videos. No unlicensed community node code,
private media or trained restoration weights are included.

### Temporal detail pulsing fixed in 0.6.5

The earlier image-style global time/space threshold selected very different
correction fractions at H3's temporal latent phases. Local video controls exposed
periodic sharp/soft transitions that sparse contact sheets had missed. Version
0.6.5 uses per-time spatial quantiles (and per-time strength normalization), without
averaging adjacent frames, adding NFE, or changing model weights and sampling clocks.

Three five-second SM89 cases reused the same six-step prefix and pixel/VAE anchor,
then regenerated only the final two steps. The 17-frame phase span of log edge
energy fell from 1.175 to 0.119 (live action), 1.172 to 0.123 (action), and 0.288
to 0.051 (animation). This is a diagnostic, not a perceptual quality score. Native
consecutive crops showed reduced soft/sharp switching with retained fabric,
background and character detail. Global time-consensus masks also reduced pulsing;
per-time selection is retained because it preserves spatial adaptation to movement.
The reusable implementation was derived from integration source `7855b6b5`.

Motion/physics and small-face defects remain, and this does not guarantee the
absence of every kind of flicker. The original sparse-frame quality observations
above are superseded for temporal stability. Full8 remains a separate option.

A later 640×640 complete request executed successfully but showed new transient
speckles on the face and clothing. A 640-pixel short side is therefore an execution
limit, **not** sufficient quality qualification for automatic selection. The clean
controls above used 992×992 and 1536×640; retain full8 for smaller unqualified
canvases. Do not use successful decoding as evidence of visual quality.

## Optional learned latent lift 6+2

`strategy="learned6+2"` keeps the same six-low/two-high eight-step video/audio
clock, but replaces the zero variant's pixel/VAE correction with the explicit
local BF16 [LBH H3 latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler).
Set `VDNAssets.upscaler_checkpoint`; missing weights are an error, not a download
or fallback. Channel normalization, alignment crop and the final two original
steps are retained. The report distinguishes `rho=0`, `pixel_vae_roundtrip=False`
and the upscaler load/compute timings. This is not the ten-step `pixel8+2` path
and does not combine the VDN model with a LightX adapter.

The execution boundary is the same five-second I2VA, minimum short side 640,
single SM89 48 GB contract as `selflift6+2`. Keep selection explicit: successful
execution is not automatic quality qualification for all canvases.

Three local five-second controls reused frozen six-step estimates, evolved audio
and clean target conditioning. Loading and learned lifting took 0.832–1.606 s,
versus 39.758–42.212 s for the earlier full-video VAE round trip. These are stage
costs, **not uncached end-to-end speedups**. All three complete outputs decoded.
Sparse full-video views and native consecutive face crops retained the subjects
and fine detail without the old strong soft/sharp pulsing. Some temporal edge
diagnostics worsened in action/animation; unchanged motion/physics and small-face
defects remain, and audio semantics were not assessed. The supported net benefit
is removed VAE work with no new major defect observed in this bounded screen,
not a universal fidelity guarantee. Integration source: `9a235f0b`.
