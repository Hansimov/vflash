# Compile official weights

The compiler reads fixed official H3 weights and the matching LoRA for Ref4 v0.1 on SM86/SM89 or T2VA Base4 v1.0 on SM89. It does not need a captured request, reference image or assets from another project. The original SM89 profiles passed exact checks of all 50 blocks, schedule and auxiliary tensors against fixed runtime controls. SM86 additionally passed its own official timestep and 50-block modulation arithmetic checks before complete generation.

The Ref4 bootstrap used freshly compiled assets with the fixed official encoders and VAEs to generate a complete 928 × 512, five-second, 24 fps MP4, then cancel a second request after one denoising step. All 14 conditioning tensors and both final latents matched the qualified control exactly; all 120 decoded video frames also matched. T2VA passed separate compilation, complete-container and cancellation checks in [0.2.0](../reference/releases#v0-2-0). These close the official-weights-to-video installation path for the measured workloads, not a broad quality or hardware qualification.

Base4 uses alpha 128; Ref4 uses alpha 8. Both use rank 128. The commands below default to SM89 Ref4; see [model profiles](../reference/pipeline-profiles) for SM86 Ref4 and T2VA commands. Select the profile in both the raw-weight and pipeline receipts.

## Get the source files

Install the release checkout with `python -m pip install '.[pipeline]'`. Use PyTorch 2.11.0 with CUDA 13.0, Linux, and an explicitly assigned GPU matching the selected SM86 or SM89 profile for compilation. Downloads and file verification use the CPU. The source files occupy about 146 GiB including the adapter; leave at least 48 GiB more free for the compiled pack. Model files keep their [upstream licenses](../reference/license).

The following download selects only one model and its common components. It defaults to Ref4; choose `model_profile("t2va-turbo4-exact-sm89")` for T2VA Base4. Prepare separate native assets and receipts for the two modes:

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

Select an idle GPU of the receipt’s target architecture owned by this process. Here `0` is the index reported by `nvidia-smi` in this environment:

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

New artifacts use schema 5 and schedules use schema 2. Their provenance separately binds the base-weight digest and the LoRA repository, revision, digest, alpha (8 for Ref4; 128 for Base4), rank 128 and strength 1, along with the compilation recipe. It contains no synthetic request or replay identifiers. The retained combined transformer digest and `oracle` fields match the existing conditioning contract; they do not claim that the compiler ran a reference generation. Existing schema-4 artifacts and schema-1 schedules remain readable under their original provenance contracts.

This compiler does not quantize weights, merge the LoRA, compile Turbo8 profiles, or qualify model quality. Changing the model, adapter or schedule requires its own implementation and validation.
