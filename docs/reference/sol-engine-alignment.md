---
search: false
---

# Sol kernel compatibility

The two-device relayout uses destination-major ordering. Compare tensor equality before running a complete request, then measure communication and elapsed generation separately. Changing attention precision, the sampling schedule or model weights introduces a separate numerical change. The versioned engineering record below identifies the upstream mechanisms, measured shapes and deferred kernels.

[Read the versioned recipe and measurements](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/reference/sol-engine-alignment.md)
