# Upgrade to Vflash 0.28

## What changes from 0.27

- **Edit a recorded clip.** The [Remake client](../guide/remake) supports distinct edit and swap models, source frames, ordered references, timeline pins and generated history. Keep the submission journal to recover the original run after a dropped connection.
- **Generate longer reference clips.** The [prepared Pika content-reference recipe](../guide/pika-reference-weights) accepts 5, 8 or 12 seconds with its own visual/audio clock. Five seconds is the default. The eight- and twelve-second runs were measured at 1280 × 720 on one 96 GiB RTX PRO 6000.
- **Validate delivered media.** For inference windows, Remake checks individual presentation timestamps at 24 fps, square pixels and the required audio stream on preserved-soundtrack results. A frame used as both a reference and a pin has independent storage for the two roles.

## Upgrade an installation

Install the [0.28.0 wheel](https://github.com/Hansimov/vflash/releases/tag/v0.28.0) in a fresh environment and run `vflash doctor`. Retain your pinned model revisions, original input media and submission receipts. A Remake client requires its separately provisioned executor. Published container images have a [separate inventory](https://github.com/Hansimov/vflash/blob/main/docker/images.json).

[Earlier published releases](https://github.com/Hansimov/vflash/releases)
