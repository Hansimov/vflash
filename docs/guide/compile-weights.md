# Prepare Ref4 from official weights

This development preview compiles the BF16 Ref2VA Turbo4 model directly from fixed official H3 weights and the Ref4 v0.1 LoRA. It does not need a captured request, reference image or assets from another project. Its first target is SM89. CPU checks are complete; the new compiler still awaits its own GPU comparison before a release.

## Get the source files

Install this branch with `python -m pip install '.[pipeline]'`. Use PyTorch 2.11.0 with CUDA 13.0, Linux, and an explicitly assigned SM89 GPU for compilation. Downloads and file verification use the CPU. The source files occupy about 146 GiB including the adapter; leave at least 48 GiB more free for the compiled pack. Model files keep their [upstream licenses](../reference/license).

The following download uses a fixed revision and the exact file list bundled with Vflash. It fetches the Ref model, text and reference encoders, and official decoders without downloading the separate Base model:

```python
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from vflash.model_assets import MODEL_REVISION, upstream_inventory
from vflash.native.h3_distilled_lora import LIGHTX_H3_REF_TURBO4_CONTRACT

root = Path("models").resolve()
snapshot_download(
    "MiniMaxAI/MiniMax-H3",
    revision=MODEL_REVISION,
    allow_patterns=list(upstream_inventory()),
    local_dir=root / "minimax-h3",
)
adapter = LIGHTX_H3_REF_TURBO4_CONTRACT
hf_hub_download(
    adapter.repository,
    adapter.filename,
    revision=adapter.revision,
    local_dir=root / "adapters",
)
```

Existing files can be reused when their bytes match the pinned sources. Downloading does not authorize executing Python from a model repository. The separate pipeline ingestion step verifies decoder source files before its explicit `trust_local_code=True` boundary.

## Verify and compile

Place the raw weights in their final location, then create a local verification receipt:

```bash
python -m vflash.compiler prepare \
  --transformer models/minimax-h3/transformer_ref \
  --adapter models/adapters/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors \
  --receipt raw-weights.json
```

This hashes all 14 transformer shards, their index and config, and the complete adapter file. It checks the shape and dtype of all 615 consumed base tensors and 600 transformer LoRA tensors. A later `python -m vflash.compiler check --receipt raw-weights.json` checks file identities and headers without initializing CUDA or rereading the full payloads. Moving or modifying the files invalidates the receipt.

Select an idle SM89 GPU owned by this process. Here `0` is the index reported by `nvidia-smi` in this environment:

```bash
python -m vflash.compiler compile \
  --receipt raw-weights.json \
  --gpu 0 \
  --output models/ref4-native
```

The compiler keeps large base matrices and LoRA residuals in BF16 on the CPU. Only the FP32 timestep MLP and one BF16 AdaLN projection at a time use CUDA. Time embeddings retain the original number of distinct timesteps per evaluation; padding is applied after the matrix operations. This avoids replacing the fixed CUDA arithmetic with a CPU approximation.

The destination must not exist. Files are built in a temporary sibling directory and published together with an atomic, non-replacing rename. A failed or cancelled compilation removes its temporary directory. Run compilation in a dedicated process so its CUDA context ends when the command exits.

## Use the result

The output contains `artifact/`, `schedule/`, `auxiliary.safetensors` and `compiled.json`. The artifact has all 50 transformer blocks. The schedule has four evaluations with video shift 12 and audio shift 3. The auxiliary pack contains the nine native input, output and normalization tensors.

Create the six-path configuration for the [complete pipeline](./complete-pipeline):

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

The raw-weight receipt and pipeline receipt serve different purposes. The first authorizes compilation of fixed source bytes; the second verifies every file consumed by the complete pipeline, including its encoders and decoders. Neither is portable across filesystem changes.

New artifacts use schema 5 and schedules use schema 2. Their provenance separately binds the base-weight digest and the LoRA repository, revision, digest, alpha 8, rank 128 and strength 1, along with the compilation recipe. It contains no synthetic request or replay identifiers. The retained combined transformer digest and `oracle` fields match the existing conditioning contract; they do not claim that the compiler ran a reference generation. Existing schema-4 artifacts and schema-1 schedules remain readable under their original provenance contracts.

This compiler does not quantize weights, merge the LoRA, compile Turbo8 or Base T2VA, or qualify model quality. Changing the model, adapter or schedule requires its own implementation and validation.
