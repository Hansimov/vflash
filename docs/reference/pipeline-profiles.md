# Complete model profiles

A prepared pipeline uses one fixed model, adapter and scheduler. Vflash 0.3.2 supports these complete pipelines:

| Profile | Hardware | Input | Transformer | Adapter | Video/audio shifts |
| --- | --- | --- | --- | --- | --- |
| `ref2va-turbo4-exact-sm89` | One RTX 4090 48 GB | Prompt and 1–3 ordered images, or one 2–5 second video | `transformer_ref` | Ref Turbo4 v0.1, alpha 8 / rank 128 | 12 / 3 |
| `ref2va-turbo4-exact-sm86` | Two RTX 3080 20 GB, `sequence-head` | Prompt and 1–3 ordered images | `transformer_ref` | Ref Turbo4 v0.1, alpha 8 / rank 128 | 12 / 3 |
| `t2va-turbo4-exact-sm89` | One RTX 4090 48 GB | Prompt without images | `transformer` | Base Turbo4 v1.0, alpha 128 / rank 128 | 6 / 3 |
| `i2va-base16-bf16-sm89` (preview) | One RTX 4090 48 GB | Prompt and one explicit first frame | `transformer` | None | 12 / 3 |

The released Turbo profiles use four evaluations, BF16 weights and separate adapter residuals. The preview I2VA profile uses the official Base transformer for 16 evaluations in BF16 without an adapter. All produce five seconds at 24 fps. The default remains SM89 Ref4. A running pipeline does not switch weights or profiles. Prepare separate assets and sessions when an application needs multiple modes. Native single-SM86 and Turbo8 interfaces have a different [validation scope](../guide/profiles).

## Prepare preview Base16 I2VA

Prepare and compile the pinned official Base transformer without `--adapter`, then build the ordinary six-field pipeline receipt. Its asset JSON sets `adapter_path` to `null` and supplies the official model and decoder directories plus the three compiled outputs.

```bash
python -m vflash.compiler prepare \
  --profile i2va-base16-bf16-sm89 \
  --transformer models/minimax-h3/transformer \
  --receipt base16-i2va-weights.json
python -m vflash.compiler compile \
  --receipt base16-i2va-weights.json --output models/base16-i2va-native --gpu 0
vflash prepare-pipeline \
  --profile i2va-base16-bf16-sm89 \
  --assets base16-i2va-assets.json --receipt base16-i2va-pipeline.json
vflash generate \
  --prepared-assets base16-i2va-pipeline.json --prompt-file prompt.txt \
  --first-frame first-frame.png --gpu 0 --seed 1234 \
  --output video.mp4 --trust-local-code
```

`--first-frame` is a frame-zero anchor, distinct from Ref2VA `--reference` inputs, and is exposed as `VideoRequest(first_frame=Path(...))` in Python. The prompt can label this image as `<Picture 1>`; labels beyond that single supplied image are rejected. Internally the official encoder uses its one-frame FL2VA conditioning path while the public request remains typed as I2VA.

## Prepare dual 3080 generation {#sm86}

Use the same official Ref4 files as in the [compiler recipe](../guide/compile-weights), but select `ref2va-turbo4-exact-sm86` when creating both receipts. Compile on one SM86 GPU; its timestep and modulation tables are specific to that target. Do not reuse a compiled SM89 pack.

```bash
python -m vflash.compiler prepare \
  --profile ref2va-turbo4-exact-sm86 \
  --transformer models/minimax-h3/transformer_ref \
  --adapter models/adapters/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --receipt ref4-sm86-weights.json
python -m vflash.compiler compile \
  --receipt ref4-sm86-weights.json --output models/ref4-sm86-native --gpu 0
vflash prepare-pipeline \
  --profile ref2va-turbo4-exact-sm86 \
  --assets ref4-sm86-assets.json --receipt ref4-sm86-pipeline.json
vflash generate \
  --prepared-assets ref4-sm86-pipeline.json --prompt-file prompt.txt \
  --reference subject.png --reference setting.png --reference style.png \
  --gpu 0 --peer-gpu 1 --strategy sequence-head \
  --output video.mp4 --seed 1234 --trust-local-code
```

The six fields in `ref4-sm86-assets.json` use the new SM86 artifact, schedule and auxiliary paths. Encoders, decoders and source LoRA can share the same immutable files as SM89. Encoding and decoding run on the primary GPU; both GPUs cooperate in native denoising. The complete pipeline rejects single-SM86 and `tensor` execution before loading models. The native latent API retains both parallel strategies.

## Prepare T2VA

The released `t2va-turbo4-exact-sm86` profile uses the same Base4 v1.0 source
files with two 3080s and `sequence-head`. SM86 compilation and the installed native
session have completed application-owned encoding/core/media requests at five seconds
928 × 512 and ten seconds 640 × 352, both 24 fps. This is the native integration boundary;
the standalone `H3Pipeline` wrapper has not been rerun on the new profile and still has
a five-second temporal contract. See [the evidence limits](./releases#v0-3-2).
It requires independent SM86 receipts and compiled tables; changing an SM89 or Ref
artifact label is invalid. Single-card, `tensor`, and eight-step T2VA are excluded.

For the new native SM86 profile, use `t2va-turbo4-exact-sm86` in the compiler preparation below and write separate SM86 outputs. After preparing compatible conditioning, run:

```bash
vflash denoise t2va-turbo4-exact-sm86 \
  --artifact models/base4-sm86-native/artifact \
  --schedule-overlay models/base4-sm86-native/schedule \
  --auxiliary-tensor models/base4-sm86-native/auxiliary.safetensors \
  --bundle inputs/t2-conditioning \
  --gpu 0 --peer-gpu 1 --strategy sequence-head \
  --output-latents output/latents.safetensors
```

The application supplies the matching conditioning bundle and decodes these latents; this command does not write an MP4. The SM89 example below retains the complete wrapper interface.

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

Prepare and hash files once in their final location. Startup checks the resulting local receipt without rereading all model payloads. In 0.3.2, `H3Pipeline.prepare()` explicitly preloads the stages; otherwise the first request loads them after CPU input validation. Models remain owned for reuse. Stage placement follows the selected profile above. Preloading does not run conditioning or prepare every input shape. Report asset preparation, model loading, first request and repeated requests separately. Cancelling an active request retires the pipeline; create a new instance before further work.

The [reference-video input](../guide/complete-pipeline#reference-video) uses the same single-SM89 Ref4 assets. It passed a complete image/video/image sequence without model reload. SM86 and Turbo8 video references remain outside the supported boundary.
