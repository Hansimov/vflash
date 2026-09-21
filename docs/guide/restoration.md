# Optional video restoration

Restoration is separate from H3 generation and is **off by default**. Keep the
original result and save an enhanced version separately. It is not lossless:
STCDiT can reduce broken textures while softening or repainting details.

## Local STCDiT-tiny runtime

The complete file-to-file entry point keeps the original and copies its audio
packets without re-encoding. The enhanced H.264 picture is separately encoded:

```bash
vflash restore-video --source-video original.mp4 --output enhanced.mp4 \
  --runtime-code /models/STCDiT --weights /models/stcdit-tiny \
  --caption-file observation.txt --trust-local-code
```

The Python equivalent is `vflash.restoration.restore_video`. It decodes the local
source, derives complete motion segments, invokes the model and atomically saves
a new MP4. Existing output paths are refused. Optional `on_progress` callbacks
can cancel by raising; the original remains usable. Model/GPU allocation belongs
to the caller; this command does not borrow a serving device automatically.

Install `vflash[pipeline,restoration]`. In an isolated process with one allocated
CUDA device, provide these local artifacts; no model is downloaded automatically:

- Author runtime: [STCDiT](https://github.com/JyChen9811/STCDiT), revision
  `7c4be6e2774b1bdf51658d1e495a0a3a3ace3772`.
- Base: [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B), revision
  `37ec512624d61f7aa208f7ea8140a131f93afc9a`, including the diffusion safetensor,
  UMT5 weights/tokenizer and Wan VAE.
- Adapter: [author STCDiT weights](https://modelscope.cn/models/junyangchen/STCDiT_ckpt),
  revision `3bc4dfb4e720bcc39d7313e62fd77dca4cb17a25`, file `tiny_8k.bin`.

The base and adapter model cards declare Apache 2.0. The runtime's package metadata
declares Apache, but its checkout has no complete root license/NOTICE. Preserve
upstream attribution and review those artifacts separately; Vflash does not vendor
or relicense them. `trust_local_code=True` explicitly authorizes execution of the
provided runtime. File sizes are inventory checks, not content authentication.

```python
from pathlib import Path
from vflash.adapters.stcdit_runtime import LocalStcditTiny

# frames: chronological RGB PIL images, on the original 24 fps clock.
# segments: explicit half-open motion segments covering every input frame.
with LocalStcditTiny(
    source=Path("/models/STCDiT"),
    weights=Path("/models/stcdit-tiny"),
    trust_local_code=True,
) as restorer:
    result = restorer.restore(
        frames, caption=observation_caption, segments=segments, seed=42
    )
    # Save result.frames to a NEW destination; retain source audio and timebase.
```

The recipe fixes BF16, ten evaluations, CFG 1, shift 5, native PyTorch attention,
CPU model offload and same-size output. H3's Sol/dense selection is unaffected.
The adapter rejects missing/overlapping segments, changed frame counts and changed
canvases. It copies input images before calling the backend. It does not detect
bad frames, write account data, charge credits, modify audio or approve quality.
Up to 240 frames and 32-aligned canvases up to 1,048,576 pixels are input limits,
**not a full-length hardware or quality qualification**.

Applications with an already loaded compatible pipeline may instead use
`StcditTinyRestorer(pipeline)`. That adapter does not own the supplied model's CUDA
lifetime; the application must provide inference mode and serial access.

## Evidence boundary

Two 17-frame, 864-square, same-size exploratory intervals on one RTX 4090 48 GB
each needed approximately 127–130 seconds including RGB output saving, excluding
model loading and captioning. Device samples reached about 11.55 GiB occupied.
These are two intervals from one video, not independent-scene validation or an
SM86 benchmark. Broken texture was reduced, but fast-action softness and detail
changes remained. Do not extrapolate this into a universal repair guarantee.

The adapter's frozen-input 17-frame recheck matched the direct runtime's output
RGB exactly. This proves the adapter did not alter that result, not that the
baseline is fully repaired. Full-length runtime, other geometries, boundary
playback and production integration remain separately qualified surfaces.
