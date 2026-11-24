# 准备权重并校验模型资源

以下流程从指定修订的 H3 原始权重构建固定配置流水线。应为原始快照、编译后的主干及解码器预留磁盘空间。Pika 和 Remake 有各自的准备步骤。

## 下载所需的权重

先选择配置。资源清单会给出该配置使用的主干及适配器文件，解码器使用同一修订。

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

## 按配置编译

先记录输入身份，再编译出配置产物、采样时序和辅助张量。绑定流水线时保留完整的输出目录。

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

## 绑定资源并复用

填写编码器快照、解码器目录、适配器和原生文件的路径。准备步骤会校验一次清单。移动或更换文件后重新生成校验文件，再执行[视频生成](./complete-pipeline)。

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
