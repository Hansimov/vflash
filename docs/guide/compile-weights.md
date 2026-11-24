# Prepare the weights and verify the asset receipt

This procedure builds a fixed-profile pipeline from the pinned original H3 weights. Allocate disk space for the original snapshot, compiled transformer and decoder. Use the Pika or Remake setup for those runtimes.

## Download the inventory

Select the profile first. The inventory helper names the files consumed by its transformer and adapter. Download the corresponding decoder from the same revision.

```python
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from vflash.model_assets import MODEL_REVISION, model_profile, upstream_inventory

root = Path("models").resolve()
profile = model_profile("ref2va-turbo4-exact-sm89")
patterns = [
    name for name in upstream_inventory()
    if not name.startswith(("transformer/", "transformer_ref/"))
    or name.startswith(profile.transformer_component + "/")
]
snapshot_download(
    "MiniMaxAI/MiniMax-H3",
    revision=MODEL_REVISION,
    allow_patterns=patterns,
    local_dir=root / "minimax-h3",
)
adapter = profile.adapter
hf_hub_download(
    adapter.repository,
    adapter.filename,
    revision=adapter.revision,
    local_dir=root / "adapters",
)
```

## Compile for the selected profile

Prepare records input identity, then compile writes the profile artifact, schedule and auxiliary tensors. Keep the output directory together when binding it to a pipeline.

```bash
python -m vflash.compiler prepare \
  --transformer models/minimax-h3/transformer_ref \
  --adapter models/adapters/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --receipt raw-weights.json
```

```bash
python -m vflash.compiler compile \
  --receipt raw-weights.json \
  --gpu 0 \
  --output models/ref4-native
```

## Bind and reuse the result

Supply the encoder snapshot, decoder directory, adapter and native files. Preparation validates the inventory once. If any of those files moves or changes, prepare a new receipt before running [generation](./complete-pipeline).

```python
import json
from pathlib import Path
from vflash.pipeline import PipelineAssets, prepare_pipeline_assets

root = Path("models").resolve()
assets = PipelineAssets(
    model_directory=root / "minimax-h3",
    adapter_path=root / "adapters/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    decoder_directory=root / "minimax-h3/FL2VA",
    artifact=root / "ref4-native/artifact",
    schedule_overlay=root / "ref4-native/schedule",
    auxiliary_tensor=root / "ref4-native/auxiliary.safetensors",
)
Path("pipeline-assets.json").write_text(json.dumps(assets.to_mapping(), indent=2))
prepare_pipeline_assets(assets, Path("prepared-assets.json"))
```
