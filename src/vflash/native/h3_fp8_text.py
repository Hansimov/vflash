"""Load mixed FP8 and unchanged text shards without replacing global factories."""

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from vflash.native.h3_fp8_linear import RowFP8Linear
from vflash.native.h3_fp8_runtime import mapped_fp8_weights


@dataclass(frozen=True)
class RowFP8TextShard:
    path: Path
    quantized: tuple[str, ...]


def load_row_fp8_text_encoder(
    model_class,
    config,
    shards,
    execution,
    original_directory: Path,
    *,
    open_weights=mapped_fp8_weights,
    submit=None,
):
    """Assign prepared E4M3 codes/FP32 row scales and original unchanged tensors.

    The model factory and all shard paths are trusted caller inputs. No model is
    downloaded, no global loader is patched and no GPU is selected here. An
    optional submit callback may own asynchronous copies of assigned tensors.
    It remains responsible for joining and releasing those copies on failure.
    """
    from accelerate import init_empty_weights

    with init_empty_weights():
        encoder = model_class(config)
    rows = tuple(shards)
    if not rows:
        raise ValueError("quantized text encoder missing")
    names = [name for row in rows for name in row.quantized]
    if len(set(names)) != len(names) or any(not name.endswith(".weight") for name in names):
        raise ValueError("duplicate or invalid quantized text matrix")
    for row in rows:
        for name in row.quantized:
            module_name = name.removesuffix(".weight")
            module = encoder.get_submodule(module_name)
            if not isinstance(module, torch.nn.Linear):
                raise ValueError("quantized checkpoint does not match a text Linear")
            encoder.set_submodule(module_name, RowFP8Linear(module, execution))
    loaded = set()
    expected = set(encoder.state_dict())
    for row in rows:
        with open_weights(row.path) as source:
            tensors = {}
            for name in source.keys():  # noqa: SIM118 -- safetensors reader is not iterable
                # Scale belongs to the replacement module, next to its weight.
                destination = (
                    name.removesuffix(".weight.row_scale") + ".row_scale"
                    if name.endswith(".weight.row_scale")
                    else name
                )
                if destination in loaded or destination not in expected:
                    raise ValueError("duplicate or unexpected text weight")
                tensors[destination] = source.get_tensor(name)
                loaded.add(destination)
            encoder.load_state_dict(tensors, strict=False, assign=True)
            if submit is not None:
                submit(tensors)
    # Shards containing only unchanged vision/embedding tensors are deliberately
    # absent from the sidecar. Read those exact tensors from the frozen source.
    missing = expected - loaded
    if missing:
        index_path = original_directory / "model.safetensors.index.json"
        if index_path.stat().st_size > 2 * 1024**2:
            raise ValueError("unbounded original text weight index")
        index = json.loads(index_path.read_text())["weight_map"]
        grouped = {}
        for name in missing:
            filename = index.get(name)
            if not filename or Path(filename).name != filename:
                raise ValueError("missing unchanged text tensor")
            grouped.setdefault(filename, []).append(name)
        for filename, names in grouped.items():
            with open_weights(original_directory / filename) as source:
                tensors = {name: source.get_tensor(name) for name in names}
                encoder.load_state_dict(tensors, strict=False, assign=True)
                if submit is not None:
                    submit(tensors)
                loaded.update(names)
    if loaded != expected or any(value.is_meta for value in encoder.state_dict().values()):
        raise ValueError("incomplete quantized text encoder")
    for module in encoder.modules():
        if isinstance(module, RowFP8Linear) and (
            module.weight.dtype != torch.float8_e4m3fn
            or module.row_scale.dtype != torch.float32
        ):
            raise ValueError("FP8 text codes/scales have an invalid dtype")
    encoder.eval().requires_grad_(False)
    return encoder
