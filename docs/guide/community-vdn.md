# VDN8 community sampling adapter

`vflash.adapters.vdn_h3.VDNEngineSession` is an **opt-in latent backend**, not a
replacement for the default `H3Pipeline`. It runs the trained VDN hybrid model,
not the original H3 weights with an interchangeable attention switch.

The current configuration is one SM89 GPU with 48 GB VRAM, rowwise FP8, eight
trained steps, 50 resident DiT blocks and streamed TokenRefiner. Local complete
I2VA and true FL2VA experiments cover 5–10 seconds. This is not a qualification
for every geometry, other GPU architectures, Ref2VA, or a production default.

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
measurements are included here. The serial adapter has focused CPU lifecycle
tests; its new packaged invocation is verified separately from that earlier
standalone experiment. No private references or generated media are distributed.
