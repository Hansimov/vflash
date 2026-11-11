# Choose a runtime and interface

Original source-video editing: [Remake windows](./remake) (experimental 0.28.0a2 preview).

Vflash 0.27.0 provides several independently configured execution paths. Select the request mode,
result type and hardware first, then prepare that path’s assets. The package version alone does
not select a model recipe. A catalog profile, a Python class and a published Docker image are
separate interfaces.

| Need | Interface | Inputs and output | Boundary |
| --- | --- | --- | --- |
| Prepared Pika six-step picture with joint or Larry native sound | [`PikaAVRuntime`](./pika-flex#prepared-pika-av) | Text or one first frame → MP4 and execution receipts | Full 96 GiB SM120; four/five-second scope. Four seconds requires Larry10. Exact canvas, decoder and memory limits apply. |
| Faithful original Pika execution | [`PikaFlexRuntime`](./pika-flex#original-bf16-reference-execution) | Upstream FL or Ref input contract → aligned MP4 | Separate Python 3.12/upstream environment; initial GPU evidence is five-second 480p text/first-frame. Admitted modes are not all equally qualified. |
| Fixed-profile text, first/last frames or content references | [`H3Pipeline`](./complete-pipeline), `vflash generate` | Profile-compatible media → complete MP4 | Explicit base/LightX profile and assets; extended hybrid scope is separate. Does not automatically run Pika. |
| Prepared Pika content-reference images | [`PikaReferenceAVRuntime`](./complete-pipeline#experimental-pika-image-reference-runtime) | Ordered appearance images → five-second MP4 | Experimental separate Ref trunk and FP32 conditioning; optional FL/Larry sound. No source-video or reference-audio support in this runtime. |
| Larry/original segmented picture with native sound | [`ReferenceAVRuntime`](../reference/segmented-av) | One same-canvas first frame → MP4 | Separate SM120 runtime and recipe; its 4+6/12 defaults do not become Pika defaults. |
| Application-owned conditioning and decoding | [`NativeEngineSession`](./python#native-session), [native HTTP](../reference/http-api) | Compiled conditioning bundle → latents | One prepared profile, bounded serial execution. The HTTP queue is in-memory and does not implement application job durability. |

## Keep the deployment identities separate

- Pin the package release and each runtime’s source, model partition, adapter, schedule and decoder inventory.
- Follow the selected runtime’s installation instructions. In particular, the original BF16 Pika environment
  and the fixed-profile `pipeline` extra have different dependency contracts.
- The public **0.18.0 Docker images** retain their own interfaces. Installing 0.27.0 locally does not
  update a remote container, add a Pika HTTP route or change an existing job’s recipe.
- Development `main` may include unreleased contracts. Use the [release notes](../reference/releases)
  and pinned tag to determine which API ships in your wheel.

## Source media is not a workflow

A first or last image is a temporal anchor. Content-reference images supply appearance; a reference
video supplies model conditioning only within that runtime’s limits. Text-only analysis attachments
belong to the application’s prompt-preparation stage and are not automatically model conditions.

Character replacement requires subject selection/tracking, masks, identity references, time windows,
history and an audio policy. An adapter loader, repaint primitive or accepted reference video alone
does not implement or qualify that complete workflow. Vflash owns reusable inference and local
execution lifetimes; the application owns durable plans, media authorization, billing, scheduling,
recovery and result delivery.

## Interpret measurements in their actual scope

Keep model loading, the complete generation call and application receipt time separate. A successful
CPU validation checks the contract, not GPU fit or visual/audio quality. Compare complete outputs against
the brief, including motion, identity, texture, sound and temporal continuity; latency or tensor parity
alone does not qualify a replacement recipe. See [performance](../reference/performance) for measured
workloads and [hardware profiles](./profiles) for each path’s tested allocation.
