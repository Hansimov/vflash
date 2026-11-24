---
search: false
---

# Original-FL video with Larry sound

ReferenceAVRuntime splits ten visual evaluations into four Larry and six original-model steps. Twelve Larry audio evaluations then condition on the frozen video. The measured configuration uses one 96 GiB SM120, a single first image on the 32-aligned canvas, five seconds and a maximum area of 2 × 1024². It delivers 120 frames at 24 fps. Use [Pika video and sound](../guide/pika-flex) for the separately prepared Pika family.

[Read the versioned recipe and measurements](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/reference/segmented-av.md)
