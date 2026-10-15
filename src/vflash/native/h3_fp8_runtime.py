"""Owned native row-FP8 factories for explicitly prepared approximate weights.

No global classes, asset loaders, environment values or admission rules change.
The caller supplies an already validated artifact, encoded block directory and
optional bounded I/O owner. The ordinary BF16 loader remains unchanged.
"""

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vflash.native import h3_native_denoiser as denoiser
from vflash.native.h3_fp8_linear import RowFP8Execution


@contextmanager
def mapped_fp8_weights(path):
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as source:
        yield source


@dataclass(frozen=True)
class RowFP8NativeLoaders:
    block_class: type
    denoiser_loader: Callable[..., Any]


def create_row_fp8_native_loaders(
    execution: RowFP8Execution,
    *,
    weights_directory: Path,
    open_weights=mapped_fp8_weights,
    token_tile_rows: int = 0,
    whole_block_rows: int = 0,
    on_block_loaded: Callable[[], None] | None = None,
) -> RowFP8NativeLoaders:
    """Create factories with independent precision, storage and block ownership.

    ``open_weights(path)`` returns a context manager yielding ``get_tensor``.
    A custom reader may overlap reads/copies but must keep its tensors alive
    until the block has uploaded them. Encoded inventory validation belongs to
    the caller; original BF16 digests cannot authenticate FP8 payloads.
    Tiling is optional and does not enlarge the pipeline request envelope.
    """
    if type(token_tile_rows) is not int or (
        token_tile_rows and (not 1024 <= token_tile_rows <= 65536 or token_tile_rows % 16)
    ):
        raise ValueError("FP8 token tiles must be zero or aligned bounded rows")
    if (
        type(whole_block_rows) is not int
        or whole_block_rows not in (0, 32768)
        or (whole_block_rows and not token_tile_rows)
    ):
        raise ValueError("whole-block execution requires bounded token tiles")

    class FP8Block(denoiser.H3NativeBlockBF16Resident):
        backend_id = "cuda-row-fp8-resident-block-v1"

        def _resident_weight(self, weight, device):
            if weight.bits != 8:
                return super()._resident_weight(weight, device)
            return replace(
                weight, values=weight.values.to(device), scales=weight.scales.to(device)
            )

        @classmethod
        def load(cls, artifact_or_directory, block_index, **kwargs):
            artifact = (
                artifact_or_directory
                if isinstance(artifact_or_directory, denoiser.H3RuntimeArtifact)
                else denoiser.load_h3_runtime_artifact(artifact_or_directory)
            )
            block = next(row for row in artifact.blocks if row.index == block_index)
            file = Path(weights_directory) / block.path
            with open_weights(file) as source:
                _, weights = denoiser.load_h3_native_block(
                    artifact,
                    block_index,
                    adaln_table=kwargs.pop("adaln_table", None),
                    tensor_load=source.get_tensor,
                )
                changes = {}
                for attr, key in (
                    ("qkv", "attn.qkv.weight"),
                    ("attention_out", "attn.out.weight"),
                    ("ffn_in", "ffn.in.weight"),
                    ("ffn_out", "ffn.out.weight"),
                ):
                    weight = getattr(weights, attr)
                    changes[attr] = replace(
                        weight, bits=8, scales=source.get_tensor(key + ".row_scale")
                    )
                weights = replace(weights, **changes)
                block = cls(artifact, weights, **kwargs)
                if on_block_loaded is not None:
                    on_block_loaded()
                return block

        def forward_prevalidated(self, hidden_states, invocation):
            if token_tile_rows:
                from vflash.native.h3_tiled_block import tiled_forward

                return tiled_forward(
                    self,
                    hidden_states,
                    invocation,
                    rows=token_tile_rows,
                    measure=execution.measure,
                    whole_rows=whole_block_rows,
                )
            with execution.measure("block.total"):
                return super().forward_prevalidated(hidden_states, invocation)

        def _attention(self, query, key, value):
            with execution.measure("attention.total"):
                return super()._attention(query, key, value)

        def _linear(self, states, weight):
            return execution.linear(states, weight.values, weight.scales)

    class FP8Denoiser(denoiser.H3NativeDenoiserBF16Resident):
        backend_id = "cuda-row-fp8-resident-stack-v1"

    def native_factory(artifact, **options):
        return FP8Denoiser.load(artifact, block_loader=FP8Block.load, **options)

    return RowFP8NativeLoaders(FP8Block, native_factory)
