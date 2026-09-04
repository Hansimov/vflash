"""Compact non-block weights consumed by the native H3 request runtime."""

from __future__ import annotations

from pathlib import Path

from vflash.native.h3_tensor_file import (
    H3SingleTensorStore,
    inspect_safetensors_header,
)


class H3RuntimeAuxiliaryError(ValueError):
    """The compact native-runtime auxiliary tensor pack is invalid."""


H3_RUNTIME_AUXILIARY_LAYOUT = "h3-native-runtime-auxiliary-v1"
H3_RUNTIME_AUXILIARY_SPECS: dict[str, tuple[str, tuple[int, ...]]] = {
    "proj_in.weight": ("F32", (5_376, 96)),
    "proj_in.bias": ("F32", (5_376,)),
    "audio_proj_in.weight": ("F32", (5_376, 32)),
    "audio_proj_in.bias": ("F32", (5_376,)),
    "norm_out.norm.weight": ("BF16", (5_376,)),
    "proj_out.weight": ("F32", (96, 5_376)),
    "proj_out.bias": ("F32", (96,)),
    "audio_proj_out.weight": ("F32", (32, 5_376)),
    "audio_proj_out.bias": ("F32", (32,)),
}


def load_h3_runtime_auxiliary(path: Path) -> H3SingleTensorStore:
    """Open the exact nine-tensor pack used outside the compiled block trunk."""

    store = H3SingleTensorStore(path)
    header = inspect_safetensors_header(store.path)
    if set(header) != set(H3_RUNTIME_AUXILIARY_SPECS):
        raise H3RuntimeAuxiliaryError("H3 runtime auxiliary tensor set is incomplete")
    for name, (dtype, shape) in H3_RUNTIME_AUXILIARY_SPECS.items():
        row = header[name]
        if row.get("dtype") != dtype or tuple(row.get("shape", ())) != shape:
            raise H3RuntimeAuxiliaryError(f"H3 runtime auxiliary tensor differs: {name}")
    return store
