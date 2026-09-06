"""Load only the H3 Transformer prefix needed to build request conditioning.

The official modular pipeline enters the Transformer to project and pack text,
video, and audio rows before block zero.  Loading the remaining 50-block DiT
for that boundary duplicates roughly forty GiB of weights that the native
denoiser already owns.  This module constructs the official Transformer on the
meta device, materializes only the five prefix modules, and replaces the trunk
and output head with an explicit capture boundary.
"""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from vflash.adapters.checkpoints import IndexedCheckpoint
from vflash.model_assets import DEFAULT_MODEL_PROFILE, model_profile
from vflash.native.h3_tensor_file import H3SingleTensorStore, inspect_safetensors_header


class H3ConditioningTransformerError(ValueError):
    """The checkpoint cannot form an exact H3 pre-block conditioning graph."""


class _EmptyWeightsFactory(Protocol):
    def __call__(self) -> AbstractContextManager[Any]: ...


_PREFIX_MODULES = (
    "proj_in",
    "audio_proj_in",
    "context_embedder",
    "time_embedder",
    "token_refiner",
)


@dataclass(frozen=True)
class H3ConditioningTransformerLoad:
    transformer: Any
    checkpoint_tensor_count: int
    checkpoint_bytes: int
    load_seconds: float


def apply_h3_conditioning_adapter(
    transformer: Any,
    adapter_path: Path,
    *,
    profile_id: str = DEFAULT_MODEL_PROFILE,
) -> dict[str, Any]:
    """Load a task-specific distilled LoRA's TokenRefiner residuals.

    The lightweight conditioning Transformer has no DiT trunk parameters. A
    full PEFT state load would therefore materialize roughly 1.3 GB of unused
    block adapters on the conditioner host. Validate the pinned safetensors
    structure, then hydrate only the 24 TokenRefiner tensors that execute
    before the capture boundary.
    """

    from peft import LoraConfig

    profile = model_profile(profile_id)
    contract = profile.adapter
    header = inspect_safetensors_header(adapter_path)
    expected = {name for name in header if name.startswith("token_refiner.")}
    if len(expected) != 24:
        raise H3ConditioningTransformerError("H3 requires 24 TokenRefiner LoRA tensors")
    transformer.add_adapter(
        LoraConfig(
            r=contract.rank,
            lora_alpha=int(contract.alpha),
            init_lora_weights=False,
            target_modules=[
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
                "ff.net.0.proj",
                "ff.net.2",
            ],
            use_rslora=False,
        )
    )
    parameters = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    if set(parameters) != expected:
        missing = sorted(expected - set(parameters))
        unexpected = sorted(set(parameters) - expected)
        raise H3ConditioningTransformerError(
            "H3 conditioning adapter targets differ from the pinned release: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    store = H3SingleTensorStore(adapter_path)
    state = store.load_many(sorted(expected))
    incompatible = transformer.load_state_dict(state, strict=False)
    missing_lora = [
        name for name in incompatible.missing_keys if ".lora_A." in name or ".lora_B." in name
    ]
    if missing_lora or incompatible.unexpected_keys:
        raise H3ConditioningTransformerError(
            "H3 conditioning adapter load was incomplete: "
            f"missing={missing_lora[:3]}, unexpected={incompatible.unexpected_keys[:3]}"
        )
    transformer.set_adapters("default", weights=contract.strength)
    transformer.eval().requires_grad_(False)
    metadata = {
        "profile": contract.profile_id,
        "repository": contract.repository,
        "revision": contract.revision,
        "filename": contract.filename,
        "sha256": contract.sha256,
        "rank": contract.rank,
        "alpha": contract.alpha,
        "workflow": profile.definition.mode.value,
        "nfe": contract.nfe,
        "video_flow_shift": profile.definition.video_flow_shift,
        "audio_flow_shift": profile.definition.audio_flow_shift,
        "strength": contract.strength,
        "loaded_tensor_count": len(state),
        "loaded_bytes": sum(
            int(tensor.numel()) * int(tensor.element_size()) for tensor in state.values()
        ),
        "validation": "prepared-immutable-assets",
    }
    prefix = getattr(transformer, "_vflash_conditioning_prefix", None)
    if isinstance(prefix, dict):
        prefix["adapter"] = dict(metadata)
    return metadata


def _parameter_bytes(tensor: Any) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _hydrate_module(
    module: Any,
    *,
    prefix: str,
    store: IndexedCheckpoint,
) -> tuple[int, int]:
    expected = tuple(module.state_dict())
    if not expected:
        raise H3ConditioningTransformerError(f"H3 conditioning module has no state: {prefix}")
    state = {}
    for local_name in expected:
        checkpoint_name = f"{prefix}.{local_name}"
        if checkpoint_name not in store.weight_map:
            raise H3ConditioningTransformerError(
                f"H3 conditioning checkpoint tensor is missing: {checkpoint_name}"
            )
        state[local_name] = store.load(checkpoint_name)
    try:
        incompatible = module.load_state_dict(state, strict=True, assign=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise H3ConditioningTransformerError(
            f"H3 conditioning checkpoint differs for {prefix}"
        ) from exc
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise H3ConditioningTransformerError(
            f"H3 conditioning checkpoint is incomplete for {prefix}"
        )
    return len(state), sum(_parameter_bytes(tensor) for tensor in state.values())


def load_h3_conditioning_transformer(
    transformer_class: Any,
    *,
    transformer_directory: Path,
    device: Any,
    torch_module: Any,
    init_empty_weights: _EmptyWeightsFactory,
) -> H3ConditioningTransformerLoad:
    """Create an official-type Transformer containing only its exact prefix.

    The returned object remains an instance of ``transformer_class`` so the
    Diffusers component contract accepts it.  Its first trunk module is an
    identity capture boundary: a pre-hook must stop execution there.  Reaching
    the boundary without an installed capture hook is a hard error.
    """

    started = time.monotonic()
    transformer_directory = transformer_directory.resolve(strict=True)
    try:
        config = transformer_class.load_config(str(transformer_directory))
        with init_empty_weights():
            transformer = transformer_class.from_config(config)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise H3ConditioningTransformerError(
            "cannot construct the H3 conditioning Transformer on meta"
        ) from exc

    class CaptureBoundary(torch_module.nn.Module):
        def forward(self, *_args: Any, **_kwargs: Any) -> Any:
            raise H3ConditioningTransformerError(
                "H3 conditioning execution reached block zero without a capture hook"
            )

    # Preserve the official 50-block structural contract used by the capture
    # session while dropping every trunk parameter before any checkpoint load.
    transformer.transformer_blocks = torch_module.nn.ModuleList(
        [CaptureBoundary(), *(torch_module.nn.Identity() for _ in range(49))]
    )
    transformer.norm_out = torch_module.nn.Identity()
    transformer.proj_out = torch_module.nn.Identity()
    transformer.audio_proj_out = torch_module.nn.Identity()

    store = IndexedCheckpoint(transformer_directory)
    tensor_count = 0
    checkpoint_bytes = 0
    for name in _PREFIX_MODULES:
        module = getattr(transformer, name, None)
        if module is None:
            raise H3ConditioningTransformerError(
                f"H3 conditioning Transformer has no prefix module: {name}"
            )
        count, size = _hydrate_module(module, prefix=name, store=store)
        tensor_count += count
        checkpoint_bytes += size

    remaining_meta = [
        name for name, parameter in transformer.named_parameters() if parameter.is_meta
    ]
    if remaining_meta:
        raise H3ConditioningTransformerError(
            "H3 conditioning Transformer left meta parameters: " + ", ".join(remaining_meta[:8])
        )
    transformer.to(device=device).eval().requires_grad_(False)
    transformer._vflash_conditioning_prefix = {
        "schema_version": 1,
        "checkpoint_tensor_count": tensor_count,
        "checkpoint_bytes": checkpoint_bytes,
        "capture_boundary": "before-transformer-block-zero",
    }
    return H3ConditioningTransformerLoad(
        transformer=transformer,
        checkpoint_tensor_count=tensor_count,
        checkpoint_bytes=checkpoint_bytes,
        load_seconds=time.monotonic() - started,
    )
