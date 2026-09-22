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
CPU model offload and same-size output by default. H3's Sol/dense selection is unaffected.
The adapter rejects missing/overlapping segments, changed frame counts and changed
canvases. It copies input images before calling the backend. It does not detect
bad frames, write account data, charge credits, modify audio or approve quality.
Up to 240 frames and 32-aligned canvases up to 1,048,576 pixels are input limits,
**not a full-length hardware or quality qualification**.

Applications with an already loaded compatible pipeline may instead use
`StcditTinyRestorer(pipeline)`. That adapter does not own the supplied model's CUDA
lifetime; the application must provide inference mode and serial access.

### Explicit high-memory residency

`--memory-policy resident` (Python: `memory_policy="resident"`) retains the VAE,
text encoder and DiT on the allocated GPU until the runtime closes. It changes
placement, not the ten-step trajectory, attention, segments or precision. It
uses more VRAM; there is no automatic fallback or claim of 24 GB qualification.
The default remains `offload`.

An exploratory matched 17-frame, 864-square run on one RTX 4090 48 GB at a 350 W
limit took 107.76 seconds versus 124.87 seconds with offload. All 17 output RGB
frames matched exactly; peak allocated tensor memory was 18.23 GiB. This is one
short-window comparison, not a repeated latency benchmark or a full-video ratio.
Two additional complete five-second, 928 × 512 normal-motion clips retained
exact decoded RGB equality over 120/120 frames each, with unchanged source audio
and clocks. This establishes placement parity for those cases, not universal
quality or a matched full-length speed ratio.

### Explicit approximate SM89 attention

`--attention-backend sage-int8-fp16` (Python: `attention_backend="sage-int8-fp16"`)
uses SageAttention 2 INT8 Q/K quantization and FP16 P/V computation, including
key smoothing. This is approximate, SM89-only, and independent of H3 Sol.
The default stays `torch`. Only the isolated STCDiT provider modules are changed;
global PyTorch attention and other model instances are not patched. Out-of-range
FP16 values fail explicitly rather than silently changing the requested backend.

Install an ABI-compatible [SageAttention 2](https://github.com/thu-ml/SageAttention)
separately; the tested source revision is
`d9704247a5139ab4c03bf7fc6b35cc0e2cbb5ea4`. It is not vendored or downloaded by
Vflash. Do not substitute Blackwell-only SageAttention 3 on Ada GPUs.
Combine with `--memory-policy resident` only when the allocated VRAM permits it.

A matched exploratory 17-frame 864-square run on one RTX 4090 48 GB at 350 W
took 103.33 seconds versus 124.87 seconds with native attention and CPU offload
(17.2% less restoration time). Ten evaluations, seed, caption and motion segments
were unchanged. A complete ten-second, 864-square, 240-frame run at 450 W with
Sage plus residency took 813.45 seconds for restoration and 851.63 seconds for
the file-to-file pipeline. The earlier native/offload restoration stage took
1,090.70 seconds: an observed 25.4% compute-time reduction, **not** a matched
end-to-end service claim. The runs were not repeated; the full candidate shared
host resources with another GPU experiment, and the older total used a different
decode/RGB-saving boundary. Peak allocated tensor memory was 20.40 GiB.

All 240 frames decoded and retained the source clock and identical decoded audio.
Non-blind original/native/candidate review at 76 selected frames found no obvious
additional structural failure; baseline fast-motion blur and repainted detail
remained. RGB is not identical. This is limited opt-in evidence, not a universal
same-quality guarantee or resolution of the source model's damaged frames.

## Evidence boundary

Two 17-frame, 864-square, same-size exploratory intervals on one RTX 4090 48 GB
each needed approximately 127–130 seconds including RGB output saving, excluding
model loading and captioning. Device samples reached about 11.55 GiB occupied.
These are two intervals from one video, not independent-scene validation or an
SM86 benchmark. Broken texture was reduced, but fast-action softness and detail
changes remained. Do not extrapolate this into a universal repair guarantee.

The adapter's frozen-input 17-frame recheck matched the direct runtime's output
RGB exactly. This proves the adapter did not alter that result, not that the
baseline is fully repaired.

A subsequent complete 240-frame, 864 × 864, 24 fps run on one RTX 4090 48 GB
(450 W limit, BF16, ten evaluations, CFG 1, shift 5, native attention, CPU offload)
completed model restoration in **1,090.70 seconds**. Model loading took 9.00 seconds;
total loading/restoration/RGB-saving/MP4 delivery took **1,177.38 seconds**. Input
decoding, captioning and service queueing are excluded from this latter measurement.
The 101 motion segments covered all frames. Device occupancy samples peaked at
13,059 MiB. The resulting MP4 decoded completely and retained identical decoded
audio samples on the original ten-second clock. This is one full-video measurement,
not independent-scene or SM86 qualification, and replaces a linear interval extrapolation.

Labeled offline review of normal moments, four damaged intervals and cut boundaries
found reduced broken texture but persistent blur during fast movement and repainted
fine detail. Clear face close-ups remained recognizable; that is not identity or
all-frame quality certification. This supports an opt-in comparison, not automatic
replacement. Real-time playback quality, other geometries and service integration
remain separate qualification surfaces.
