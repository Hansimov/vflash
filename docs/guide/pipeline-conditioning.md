# Conditioning, memory and hardware profiles

Use this guide when integrating the fixed-profile pipeline on a new device or measuring a change to its numerical path. Start with a playable control at the same resolution, profile and seed.

## Hybrid model and extended requests {#extended-hybrid}

`HybridModel(Path(".../transformer_ref"))` keeps the LightX v0.1 four-evaluation backbone and replaces block
25–49 AdaLN projections with the pinned official Ref component (about 55 MiB of tables, no second denoiser).
Keyframes use the ordinary FL graph and Ref uses a real official Ref graph sharing the prefix, text encoder
and VAEs. It runs on one SM89, SM90 or SM120 GPU (SM103 preview) with dense or explicit Veda attention;
the measured envelope of each GPU is in [choose hardware](./profiles#heterogeneous).

With Veda, the single-SM89 hybrid accepts **5–15 s** image-conditioned requests and up to **nine** ordered
references (`<Picture 1>` … `<Picture 9>`), within the [joint budget](./complete-pipeline#modes-and-limits). Standard
Ref4/Base16, SM86, paired execution and reference video do not inherit this expansion.

## Text encoder prefix {#exact-text-encoder-prefix}

Conditioning keeps 51 of the 64 Qwen3-VL decoder layers: the pinned H3 adapter consumes `hidden_states[50]`,
so the raw intermediate state is preserved while the unused tail is removed before offloading hooks are
installed. Weights, vision encoder and BF16 arithmetic are unchanged; the full checkpoint still loads first,
so this is not a smaller download or a guaranteed cold-start gain.

## Attention {#veda-dense-edges}

Single-SM89 official Base16 defaults to approximate Sol; other profiles and pairs stay dense. Select dense
explicitly with `attention_backend="torch-flash"`. [Veda](./veda) is an explicit sparse backend with five
dense layers at each edge by default (40 sparse / 10 dense); `veda_dense_edge_layers=3` selects 44 sparse /
6 dense. Each session owns its partition and reports the layers it ran. These are approximate inference
choices, not equivalence claims; Sol's [selection and limits](../reference/sol-engine-alignment#sol-default)
are documented separately.

## Conditioning reuse and handoff {#memory-conditioning}

- **Sibling reuse.** A trusted scheduler passes one opaque `ConditioningReuseScope` to sequential I2VA/L2VA/FL2VA
  candidates of the same accepted request. One copied CPU entry (≤128 MiB, one-hour idle expiry) holds text
  outputs, token tags and clean keyframe latents only; noise, scheduler state, activations and media are never
  reused. Never derive a scope from public content or reuse it across owners.
- **In-memory handoff.** `conditioning_transport="memory"` passes encoder tensors to native denoising once,
  without writing a bundle, on a single device. The default `file` transport keeps the persisted contract
  for standalone native calls and services.

## Approximate FP8 loaders {#fp8-loaders}

`create_row_fp8_native_loaders` and `load_row_fp8_text_encoder` load explicitly prepared E4M3 weights with FP32
row scales through `H3Pipeline`'s trusted `native_denoiser_loader` / `text_encoder_loader` callbacks. Row
tiling bounds temporaries while keeping full-sequence attention. W8A8 is approximate, requires SM90/SM120 and
never silently switches arithmetic; BF16 remains the default, and the prepared-asset receipt and CLI do not
accept FP8 bundles.

## Timing and diagnostics {#timing-and-diagnostics}

`VideoResult.elapsed_seconds` spans the successful `generate` call including a first model load.
`stages.initialization_seconds` is loading inside this call, `stages.request_elapsed_seconds` excludes it, and
`stages.session_initialization_seconds` (also `pipeline.initialization_seconds`) is the session's original
load. Encoding and media stages report resume, suspend and call durations; `capture_diagnostics` separates
the official call, hook waits, bundle finishing and per-process resource deltas. Nested durations are
attribution, not additive totals.

## What a correctness result means {#what-a-correctness-result-means}

Compare conditioning and final latents to the reference implementation separately from visual quality, and
judge motion, appearance, instructions and sound against the original request. H.264 and AAC are lossy, so an
MP4 hash is not numerical equivalence; the official audio VAE can differ slightly between repeated requests.
Release checks cover the 14 conditioning tensors, final FP32 latents, decoded media and playable delivery.

## Qualification evidence {#qualification-evidence}

The table summarises what the current interfaces have been run on; numbers, scopes and limits for each
are in the [release notes](../reference/releases) and [performance records](../reference/performance).

| Capability | Evidence |
| --- | --- |
| SM120 resident pixel-frame budget (0.15) | RTX PRO 6000 96 GiB, FP8 resident trunk, Veda 44/6, face and six-step audio repair: 1920×1088/5 s 163 s, 2720×1536/5 s 261 s, 1920×1088/10 s 245 s, 42.1 GiB peak |
| 2720-pixel native HD (0.14) | Five 120-frame I2VA requests on RTX 4090 48 GB, BF16 hybrid Veda 44/6: 2720×1536 ×2, 1536×2720, 2368×1760, 2720×1152 |
| Extended hybrid (0.6.12) | 15 s and nine-image cases on single SM89 with Veda |
| Six-step audio (0.11–0.13) | RTX PRO 6000 FP8 (1024², 1536×1344) and RTX 4090 48 GB BF16 ring (512², 1024², 1248×832/10 s); hidden-context reuse on two 512² cases |
| Face repaint (0.6.34–0.10) | RTX PRO 6000 Server complete-video scenes, two- and four-step; crop-encoder profiles with matched final frames |
| In-memory handoff | Six H100 SXM requests with identical decoded planes and PCM |
| Sibling reuse | Dual-SM86 A/B/A with matching streams |

Evidence on one GPU class or canvas does not qualify others; serving systems keep their own qualification.
