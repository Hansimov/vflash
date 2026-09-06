# Complete model profiles

A prepared pipeline uses one fixed model, adapter and scheduler. Vflash 0.2.0 supports these complete pipelines on one RTX 4090 48 GB:

| Profile | Input | Transformer | Adapter | Video/audio shifts |
| --- | --- | --- | --- | --- |
| `ref2va-turbo4-exact-sm89` | Prompt and 1–3 ordered images | `transformer_ref` | Ref Turbo4 v0.1, alpha 8 / rank 128 | 12 / 3 |
| `t2va-turbo4-exact-sm89` | Prompt without images | `transformer` | Base Turbo4 v1.0, alpha 128 / rank 128 | 6 / 3 |

Both use four evaluations, BF16 weights and separate adapter residuals. The default is Ref4. A running pipeline does not switch between Base and Ref weights. Prepare separate assets and sessions when an application needs both modes. See [hardware and validation scope](../guide/profiles) for the native interfaces; complete SM86 generation is not supported in this release.

## Prepare T2VA

Follow the [official-weight download recipe](../guide/compile-weights), selecting `t2va-turbo4-exact-sm89`. Then prepare the Base checkpoint and its exact pinned adapter:

```bash
python -m vflash.compiler prepare \
  --profile t2va-turbo4-exact-sm89 \
  --transformer models/minimax-h3/transformer \
  --adapter models/adapters/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors \
  --receipt base4-weights.json
python -m vflash.compiler compile \
  --receipt base4-weights.json --output models/base4-native --gpu 0
vflash prepare-pipeline \
  --profile t2va-turbo4-exact-sm89 \
  --assets base4-assets.json --receipt base4-pipeline.json
vflash generate \
  --prepared-assets base4-pipeline.json --prompt-file prompt.txt \
  --output video.mp4 --gpu 0 --seed 1234 --trust-local-code
```

The six fields in `base4-assets.json` follow the [complete pipeline asset schema](../guide/complete-pipeline). Use the Base adapter and the newly compiled Base artifact, schedule and auxiliary paths. The official decoder directory and common encoders can be shared as immutable files.

Omit `--reference` for T2VA. Its Python request is `VideoRequest(prompt=..., seed=...)`. Ref2VA accepts one to three references and preserves their order. Unbound `<Picture N>` labels and mode mismatches are rejected before execution.

The Base adapter filename includes `fl2v`; this release qualifies it for T2VA, not first/last-frame generation. Newer Base4 adapters are not interchangeable with this pinned v1.0 profile. Revision, file hash and license links are in [runtime assets](./runtime-assets).

## Resource lifetime

Prepare and hash files once in their final location. Startup checks the resulting local receipt without rereading all model payloads. A persistent pipeline initializes before requests, uses one GPU serially for encoding, denoising and decoding, and retains CPU model copies for reuse. Report preparation, construction, first request and warm request separately. Cancelling an active request retires the pipeline; create a new instance before further work.
