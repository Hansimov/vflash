# Complete pipeline profiles

This development branch extends the complete pipeline beyond the Ref4 SM89 profile in the published [0.1.0 release](./releases). The new T2VA and SM86 paths are undergoing container qualification; this page describes their configuration contract, not additional released support.

Each prepared pipeline has one fixed model profile. The profile binds the Transformer checkpoint, distilled adapter, scheduler and GPU architecture together. A running pipeline does not switch between Base and Ref weights.

| Profile | Transformer component | Adapter | Video/audio shifts | Hardware |
| --- | --- | --- | --- | --- |
| `ref2va-turbo4-exact-sm89` | `transformer_ref` | Ref Turbo4 v0.1 | 12 / 3 | SM89 48 GB |
| `ref2va-turbo4-exact-sm86` | `transformer_ref` | Ref Turbo4 v0.1 | 12 / 3 | SM86 20 GB, explicit cooperating pair supported by the native plan |
| `t2va-turbo4-exact-sm89` | `transformer` | Base Turbo4 v1.0 | 6 / 3 | SM89 48 GB |

All three use four evaluations and separate BF16 adapter residuals. Base4 v1.0 uses rank 128 and alpha 128; Ref4 v0.1 uses rank 128 and alpha 8. A newer adapter with a similar filename does not satisfy these fixed identities. Refer to the pinned [asset inventory](./runtime-assets) and upstream model terms before obtaining weights.

## Prepare and compile

The compiler selects its contract when raw weights are prepared:

```sh
python -m vflash.compiler prepare \
  --profile t2va-turbo4-exact-sm89 \
  --transformer /models/MiniMax-H3/transformer \
  --adapter /models/base4-v1.0.safetensors \
  --receipt /models/base4-weights.json
python -m vflash.compiler compile \
  --receipt /models/base4-weights.json --output /models/base4-compiled --gpu 0
vflash prepare-pipeline \
  --profile t2va-turbo4-exact-sm89 \
  --assets /models/base4-assets.json --receipt /models/base4-pipeline.json
```

The six asset paths retain their [existing schema](../guide/complete-pipeline). The explicit profile belongs to the receipt, rather than being inferred from a path or a filename. A compiler uses an exclusively assigned GPU of the profile's architecture; it does not acquire a deployment's device lease itself. Preparation verifies official bytes once, and startup checks local file identity without rehashing the large weights.

The time-embedding MLP and AdaLN projections preserve each evaluation's original row count. Ref4 has `(2, 3, 3, 3)` distinct timestep rows; T2VA has `(1, 2, 2, 2)`. Padding occurs after each learned projection. The original Ref4 SM89 model identity, asset receipt and Python compiler entrypoints remain compatible.

## Requests and ownership

A `VideoRequest` without images is T2VA. One to three ordered images select Ref2VA. Passing a request to a pipeline of the other mode fails before any encoding or denoising stage starts. `<Picture N>` labels must refer to supplied images.

```python
request = VideoRequest(prompt="A structured H3 text-to-video description.", seed=17)
```

The CLI follows the same rule: omit `--reference` for T2VA. For a prepared SM86 profile, `--gpu 0 --peer-gpu 1` selects an explicit pair, using the native `sequence-head` strategy by default. The Python equivalents are `device=...`, `peer_device=...` and optional `strategy=...`. The text encoder and VAE run serially on the primary device; the native denoiser owns both devices. Session shutdown retires the stages before releasing the native GPU group. Queueing and resource borrowing remain application responsibilities.
