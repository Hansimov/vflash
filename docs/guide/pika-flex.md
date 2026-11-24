# Pika video generation and soundtrack selection

Choose the prepared runtime for repeated text or first-frame jobs on a complete 96 GiB SM120 device. Start with one five-second take before changing duration, canvas or audio recipe.

## Prepared Pika AV {#prepared-pika-av}

The Pika adapter performs six visual evaluations. Joint audio comes from the AV pass; Larry10 and Larry12 perform separate native audio refinement. A four-second request has a qualified Larry10 path. The Python constructor takes prepared visual and audio components, manifests and a text loader. Keep those assets bound to the same recipe.

```python
from vflash.pipeline import VideoRequest
from vflash.pipeline.pika_av import PikaAVRecipe
from vflash.pipeline.pika_av_runtime import PikaAVRuntime

# All paths, manifests and the text loader are explicit caller-owned components.
runtime = PikaAVRuntime(**prepared_components, **pika_components, trust_local_code=True)
try:
    result = runtime.generate(
        VideoRequest("A ceramic bird unfolds its wings.", width=960, height=544),
        output_directory,
        recipe=PikaAVRecipe(audio_steps=10, visual_attention="veda", overlap_decode=True),
    )
finally:
    runtime.close()
```

## Choose the reference contract

Use a first frame to fix the opening picture. For a new shot informed by appearance photos, use the [Pika reference runtime](./pika-reference-weights); it owns a separate Ref trunk. For a source film with an already-defined gesture, use [Remake](./remake).


<span id="bounded-decoder-tile-batches" aria-hidden="true"></span>

## Original BF16 implementation {#original-bf16-reference-execution}

Use the original upstream environment to compare Pika output or integrate a released upstream FL/Ref model. Pin the upstream source and model files in that environment and release the runtime after the last request.

```python
from pathlib import Path
from vflash.pipeline.pika_flex import PikaFlexRequest
from vflash.pipeline.pika_flex_runtime import PikaFlexRuntime

runtime = PikaFlexRuntime(
    source_root=Path("Pika-H3-Flex"),
    base_path=Path("models/MiniMax-H3"),
    adapters_root=Path("models/Pika-H3-Flex"),
    trust_local_code=True,
)
try:
    result = runtime.generate(
        PikaFlexRequest(prompt="integrated_multimodal_description: [Shot 1] ...\n"
                               "overall_soundscape: ...\nnon_diegetic_music: N/A"),
        Path("outputs/take.mp4"),
    )
finally:
    runtime.close()
```
