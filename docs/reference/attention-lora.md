# Optional FP32 attention LoRA

This experimental **low-level** interface applies a reversible DiT-only attention adapter to a
serial, single-SM89, official H3 Base16 runtime. It is off by default. It does not change prepared
assets, merge the base weights, select a model automatically, or add a pipeline/HTTP parameter.

## Use an existing native session

Create a Base16 session using the [Python integration guide](../guide/python), selecting
`i2va-base16-bf16-sm89`, matching assets, and block-ring residency for Sol. Then:

```python
from pathlib import Path
from safetensors.torch import load_file
from vflash.native.h3_attention_lora import apply_dit_attention_lora

state = load_file("attention-adapter.safetensors", device="cpu")
# Inside an exclusively owned, active Base16 NativeEngineSession:
with apply_dit_attention_lora(session.runtime, state, rank=8, scale=-1.0) as calls:
    result = session.generate(
        Path("bundles/example"), Path("outputs/adapter-latents.safetensors")
    )
print(calls)
# Outside the context, the same session uses its unchanged base weights.
```

Output remains AV latents; use the matching official media decoder for MP4 delivery. Preserve the
checkpoint identity, rank, effective scale, DiT-only scope and engine revision alongside results.
Do not label the output as an adapter-free baseline merely because its base artifact is unchanged.

## Format and arithmetic

The input is the complete 208-tensor CPU FP32 PEFT attention state: 50 DiT blocks plus two
TokenRefiner blocks, with QKV and output A/B tensors. TokenRefiner is validated but **not applied**.
This is not equivalent to applying the complete adapter in another framework.

The source QKV B rows are interleaved per head and are reordered to whole Q/K/V blocks. The two
adapter GEMMs and addition use FP32, then return to the base activation dtype. Adapter weights are
not rounded into BF16 base weights. Row chunks bound temporary memory; ring slots receive the
correct logical block's adapter on the compute path, not in asynchronous prefetch callbacks.

`scale` must be explicitly supplied in [-1, 1] and includes any alpha/rank multiplier.
For alpha=rank, -1 subtracts the original update once. Do not both negate B and pass -1.
All shape, precision and finite-value checks run before attaching the adapter.
SM86, multi-GPU, other step schedules, existing compiled adapters and nested contexts are rejected.

A published experimental example is the
[reverse attention adapter](https://huggingface.co/rockstarengine/vflash).
Its model card and MiniMax model license apply separately from this package's source license.
No weights or private reference media are bundled.

## Lifetime and evidence boundary

Use one exclusively owned runtime/process, serial requests, and an unmodified state during the
context. Exit restores original methods, including after exceptions. As with any failed CUDA
request, close the runtime instead of assuming its device state is reusable.

Focused checks cover FP32 arithmetic, QKV layout, ring traversal, repeated invocations and
exception cleanup. The mechanism has prior single-SM89 complete-video evidence at small and
large canvases; the public extraction is undergoing a same-input migration check. None of this
guarantees every face, action or soundtrack improves. Some existing artifacts remain, and the
generated composition can change. Assess against the input brief, not pixel similarity to Base.
The interface does not enable an adapter in production or change Base16/Sol defaults.

