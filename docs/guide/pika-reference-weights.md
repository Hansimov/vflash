# Generate from appearance photos

<span id="experimental-pika-image-reference-runtime" aria-hidden="true"></span>


PikaReferenceAVRuntime takes ordered photographs for appearance. Use it when the subject should look familiar but the camera can create a new shot.

## Select the length

The five-second recipe is the starting point. Explicit eight- and twelve-second recipes use ten audio steps. Match the request duration to the recipe. At 24 fps, the eight-second result contains 192 delivered frames and the twelve-second result 288.

```python
from vflash.pipeline.pika_reference_recipe import PikaReferenceRecipe

recipe = PikaReferenceRecipe(duration_seconds=8, audio_steps=10)
```

## Size the references

Keep processing within the runtime’s 1 MP reference scope and maintain input order in the prompt. Record the source and encoded dimensions from the conditioning receipt. Changing the reference encoding budget can change small lettering, hands and product details. Compare the completed movie at native size.

## Quantized visual weights

QAD visual weights have separate manifests and SM120 kernels. Earlier tests found reduced sharpness and motion in some clips. Integrators reproducing those settings can consult the [pinned technical record](https://github.com/Hansimov/vflash/blob/v0.28.0a8/docs/guide/pika-reference-weights.md).
