# Inside the inference engine

A complete film crosses three ownership boundaries: conditioning, denoising and media delivery. The API you choose determines which boundaries Vflash owns.

## From inputs to frames {#data-path}

The complete pipeline validates the request against its profile, encodes prompt and media, samples video and audio latents, then invokes VAEs and encodes the MP4. A native session starts at already compiled conditioning and returns the sampled latents. Pika and Remake have their own execution contracts and asset inventories.

## Where to change the code {#sessions}

| Concern | Source |
| --- | --- |
| Supported profile and device placement | `catalog.py`, `contracts.py`, `planner.py` |
| Weight conversion and integrity | `compiler/`, `model_assets.py` |
| Native model operations and lifecycle | `native/` |
| Request composition and stage lifetime | `pipeline/` |
| Encoder and decoder integration | `adapters/`, `media/` |
| HTTP admission and output retrieval | `server.py` |

Keep model math independent of service transport. The application owns identity, billing, user history and durable scheduling.

## Measure each boundary

Record initialization separately from repeated calls. Keep the same assets, device, seed, input and decoder while comparing a kernel change. Check delivered picture and sound as well as intermediate numerical error. See [performance measurements](./benchmarks).
