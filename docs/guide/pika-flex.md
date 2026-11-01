# Pika H3 Flex reference integration

`PikaFlexRuntime` adapts the original [Pika release](https://github.com/Pika-Labs/Pika-H3-Flex)
without changing the hybrid or segmented pipeline defaults. This initial adapter preserves the
upstream staged BF16 implementation. The staged path has completed 5-second 480p text and first-frame generation on a
96 GiB Blackwell PRO 6000. Other modes, larger canvases and serving integration remain
unqualified; CPU contract checks do not establish GPU output quality or service availability.

Use a separate Python 3.12 environment with the upstream pinned requirements and its two
Diffusers patches. The existing `vflash[pipeline]` stack pins different dependencies: do not
install both stacks into the same environment. Install vflash itself without its pipeline extra.
The source checkout is pinned to `4bf8ea508891a432106e70f675d287c27f02306d`; its 22 Python/YAML
files are checked against the packaged inventory before import. No code or models are downloaded
by this adapter. Explicit local-code trust is required.

```python
from pathlib import Path
from vflash.pipeline.pika_flex import PikaFlexRequest
from vflash.pipeline.pika_flex_runtime import PikaFlexRuntime

runtime = PikaFlexRuntime(
    source_root=Path("Pika-H3-Flex"),
    base_path=Path("models/MiniMax-H3"),
    adapters_root=Path("models/Pika-H3-Flex"),
    trust_local_code=True,
)
try:
    result = runtime.generate(
        PikaFlexRequest(prompt="integrated_multimodal_description: [Shot 1] ...\n"
                               "overall_soundscape: ...\nnon_diegetic_music: N/A"),
        Path("outputs/take.mp4"),
    )
finally:
    runtime.close()
```

Assets use MiniMax H3 revision `42ed227ee7df40d41602854ae760620d6eb651fe` and Pika adapter
revision `0ddf55ad7067ca1e76812253bd7b4600df87f759`. Each available adapter directory is named
`fl2va` or `ref2va` and contains its original `manifest.json` and `adapter.safetensors`.
FL handles text, first, last and both-frame input; Ref handles ordered content references.
The upstream executor verifies the publisher's adapter hash before loading. Its metadata records
local base overrides; the application must retain the base download/conversion provenance.

The release always generates video and audio jointly, with six model evaluations. It uses
knots `[1000,750,500,250,100,40,0]`, shifts 12/3 and fresh noise after the first five clean
predictions. A six-step Euler schedule is a different algorithm. No Larry audio pass is added.
FL uses BF16 conditioning and Ref uses FP32 conditioning. The adapters have rank and alpha 128.

The local release exposes 480p/768p buckets, eleven aspect buckets, integer 5–15-second
requests and 24 fps. Output retains the model's aligned `17n+5` frame clock, so a five-second
request emits 124 frames. Application trimming/cropping and display geometry must be recorded
separately. This adapter does not claim native 4K, arbitrary canvases, separate audio stems or
seamless extension. A failed request poisons this runtime; tear down its dedicated process
before retrying. Existing output files are never overwritten.

Pika code is Apache-2.0. The adapters are Model Derivatives under the MiniMax H3 Community
License Agreement. Keep the upstream LICENSE, NOTICE, modification notices and model licenses
with any distributed assets. The integration contains release identities, not model weights.

## Opt-in component reuse

`reuse_components=True` retains one original BF16 transformer partition and one original FP32
VAE pair. Before text encoding or decoding, the transformer moves to CPU; text weights load
onto their target device without materializing a second full CPU model. Changing FL/Ref releases
the previous trunk. Adapter injection/loading runs once for a retained trunk; changing the
adapter file or configuration rejects reuse. Sampling, conditioning precision and native audio
remain the original six-step computation.

This option is a lifecycle candidate: CPU tests cover reuse, partition changes, exact small-tensor
values, failure cleanup and restored hooks. Full-weight GPU memory, quality and latency validation
is still pending. It cannot be combined with `cpu_offload=True`. Use a dedicated process; all
`PikaFlexRuntime` calls in that interpreter share a nonblocking owner lock. `close()` releases
retained components. Results include `component_reuse` and load/reuse counters in `component_stats`.

The reusable activation adapter also accepts explicit `kind="pika"` with the original release
manifest. It validates all 312 rank-128 pairs, retains the original parameter names, uses
alpha/rank=1 and casts stored FP32 adapter tensors to BF16 as the released PEFT loader does.
It never applies LightX's fallback alpha. Partial installation failures remove all hooks.
This is a loader contract, not qualification of a quantized six-step generation pipeline.
