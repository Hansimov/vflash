"""Explicit local adapter configuration; importing it never initializes CUDA."""

import math
from dataclasses import dataclass
from pathlib import Path

from vflash.contracts import ContractError


@dataclass(frozen=True)
class AttentionAdapter:
    """Complete FP32 PEFT attention state, applied to the DiT trunk only.

    The scale includes alpha/rank. No automatic model selection, download,
    base-weight merge, or TokenRefiner adaptation is performed.
    """

    path: Path
    rank: int
    scale: float

    def __post_init__(self) -> None:
        if type(self.rank) is not int or not 1 <= self.rank <= 256:
            raise ContractError("integer adapter rank in [1, 256] required")
        if (
            type(self.scale) not in (int, float)
            or not math.isfinite(self.scale)
            or not -1 <= self.scale <= 1
        ):
            raise ContractError("explicit finite adapter scale in [-1, 1] required")
        path = Path(self.path).resolve(strict=True)
        if not path.is_file() or path.suffix != ".safetensors":
            raise ContractError("local attention adapter safetensors file required")
        object.__setattr__(self, "path", path)

    def metadata(self) -> dict:
        return dict(
            rank=self.rank, scale=self.scale, scope="dit-only", precision="fp32-residual"
        )


def open_attention_adapter(runtime, configuration: AttentionAdapter):
    from safetensors.torch import load_file

    from vflash.native.h3_attention_lora import apply_dit_attention_lora

    state = load_file(configuration.path, device="cpu")
    return apply_dit_attention_lora(
        runtime, state, rank=configuration.rank, scale=configuration.scale
    )
