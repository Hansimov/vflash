"""Capture the official Ref2VA encoder boundary before the first native block."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from vflash.native.h3_conditioning_bundle import (
    _FILES,
    _TENSORS,
    H3ConditioningBundle,
    H3ConditioningBundleError,
    H3ConditioningProfile,
    h3_target_video_tokens,
    seal_h3_conditioning_bundle,
)
from vflash.native.h3_latent_layout import infer_h3_condition_prefix_counts
from vflash.native.h3_native_scheduler import H3NativeSchedule
from vflash.native.h3_tensor_file import save_safetensors_atomic

_STATIC_INPUTS = {
    "hidden_states": "initial_video_latents",
    "audio_hidden_states": "initial_audio_latents",
    "encoder_hidden_states": "encoder_hidden_states",
    "token_tags": "token_tags",
    "position_ids": "position_ids",
    "video_indices": "video_indices",
    "audio_indices": "audio_indices",
    "text_indices": "text_indices",
}


class H3ConditioningCaptureComplete(RuntimeError):
    """Expected control-flow signal raised before the first H3 block executes."""


def _cpu_tensor(value: Any, *, name: str) -> Any:
    import torch

    if not isinstance(value, torch.Tensor):
        raise H3ConditioningBundleError(f"H3 conditioning capture expected a Tensor for {name}")
    return value.detach().to(device="cpu", copy=True).contiguous()


class H3ConditioningCaptureSession:
    """Capture one request and stop immediately before transformer block zero."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._handles: list[Any] = []
        self._tensors: dict[str, Any] = {}
        self._installed = False
        self._transformer_entered = False
        self._captured = False

    @property
    def captured(self) -> bool:
        return self._captured

    def install(self, transformer: Any) -> None:
        if self._installed:
            raise H3ConditioningBundleError(
                "H3 conditioning capture session is already installed"
            )
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None or not hasattr(blocks, "__getitem__") or len(blocks) != 50:
            raise H3ConditioningBundleError(
                "H3 conditioning capture requires exactly 50 transformer blocks"
            )
        self._handles.append(
            transformer.register_forward_pre_hook(self._transformer_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            transformer.time_embedder.register_forward_hook(self._time_embedding_hook)
        )
        self._handles.append(transformer.rope.register_forward_hook(self._rotary_hook))
        self._handles.append(
            blocks[0].register_forward_pre_hook(
                self._block_zero_pre_hook,
                with_kwargs=True,
                prepend=True,
            )
        )
        self._installed = True

    def close(self) -> None:
        while self._handles:
            self._handles.pop().remove()
        self._installed = False

    def discard(self) -> None:
        self.close()
        self._tensors.clear()

    def _transformer_pre_hook(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        if args:
            raise H3ConditioningBundleError(
                "H3 conditioning capture requires keyword-bound transformer inputs"
            )
        if self._transformer_entered or self._captured:
            raise H3ConditioningBundleError(
                "H3 conditioning capture observed more than one evaluation"
            )
        for source, target in _STATIC_INPUTS.items():
            if source not in kwargs:
                raise H3ConditioningBundleError(
                    f"H3 conditioning capture is missing transformer input: {source}"
                )
            self._tensors[target] = _cpu_tensor(kwargs[source], name=source)
        self._tensors["first_timesteps"] = _cpu_tensor(kwargs.get("timestep"), name="timestep")
        self._tensors["first_timestep_indices"] = _cpu_tensor(
            kwargs.get("timestep_indices"), name="timestep_indices"
        )
        self._transformer_entered = True

    def _time_embedding_hook(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if not self._transformer_entered or "first_time_embeddings" in self._tensors:
            raise H3ConditioningBundleError(
                "H3 conditioning time embedding is outside the first evaluation"
            )
        self._tensors["first_time_embeddings"] = _cpu_tensor(output, name="time_embeddings")

    def _rotary_hook(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        if (
            not self._transformer_entered
            or "rotary_cos" in self._tensors
            or not isinstance(output, tuple)
            or len(output) != 2
        ):
            raise H3ConditioningBundleError("H3 conditioning rotary output is invalid")
        self._tensors["rotary_cos"] = _cpu_tensor(output[0], name="rotary_cos")
        self._tensors["rotary_sin"] = _cpu_tensor(output[1], name="rotary_sin")

    def _block_zero_pre_hook(
        self,
        _module: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> None:
        if not self._transformer_entered or self._captured:
            raise H3ConditioningBundleError(
                "H3 conditioning block-zero input is outside the first evaluation"
            )
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        self._tensors["first_packed_input"] = _cpu_tensor(
            hidden_states, name="first_packed_input"
        )
        if set(self._tensors) != _TENSORS:
            missing = sorted(_TENSORS - set(self._tensors))
            raise H3ConditioningBundleError(
                f"H3 conditioning capture is incomplete before block zero: {missing}"
            )
        self._captured = True
        raise H3ConditioningCaptureComplete(
            "H3 conditioning captured before transformer block zero"
        )

    def prefix_counts(self) -> tuple[int, int]:
        import torch

        if not self._captured:
            raise H3ConditioningBundleError(
                "H3 conditioning capture has not reached block zero"
            )
        timesteps = self._tensors["first_timesteps"]
        timestep_indices = self._tensors["first_timestep_indices"]
        return infer_h3_condition_prefix_counts(
            captured_timesteps=timesteps.unsqueeze(0),
            captured_timestep_counts=torch.tensor([int(timesteps.shape[0])], dtype=torch.int64),
            captured_timestep_indices=timestep_indices.unsqueeze(0),
            video_indices=self._tensors["video_indices"],
            audio_indices=self._tensors["audio_indices"],
        )

    def reference_token_budget(self, *, width: int, height: int, frames: int) -> int:
        if not self._captured:
            raise H3ConditioningBundleError(
                "H3 conditioning capture has not reached block zero"
            )
        rows = int(self._tensors["initial_video_latents"].shape[1])
        budget = rows - h3_target_video_tokens(width=width, height=height, frames=frames)
        if budget < 0:
            raise H3ConditioningBundleError(
                "H3 conditioning video rows are smaller than the target"
            )
        return budget

    def finish(
        self,
        *,
        bundle_id: str,
        profile: H3ConditioningProfile,
        request: Mapping[str, Any],
        source: Mapping[str, str],
        video_sigmas: Sequence[float] | Any,
        audio_sigmas: Sequence[float] | Any,
        update_rule: str,
        schema_version: int = 1,
    ) -> H3ConditioningBundle:
        if not self._captured or set(self._tensors) != _TENSORS:
            raise H3ConditioningBundleError("H3 conditioning capture is incomplete")
        observed_budget = self.reference_token_budget(
            width=profile.width,
            height=profile.height,
            frames=profile.frames,
        )
        video_prefix, audio_prefix = self.prefix_counts()
        if (
            observed_budget != profile.reference_token_budget
            or video_prefix != profile.num_condition_video_rows
            or audio_prefix != profile.num_condition_audio_rows
        ):
            raise H3ConditioningBundleError(
                "H3 conditioning observed prefixes differ from the profile"
            )
        if self.directory.exists() and any(self.directory.iterdir()):
            raise H3ConditioningBundleError("H3 conditioning capture directory must be empty")
        self.directory.mkdir(parents=True, exist_ok=True)
        save_safetensors_atomic(self.directory / _FILES["conditioning"], self._tensors)

        def values(rows: Sequence[float] | Any) -> tuple[float, ...]:
            if hasattr(rows, "detach"):
                rows = rows.detach().to(device="cpu").tolist()
            return tuple(float(item) for item in rows)

        schedule = H3NativeSchedule(
            video_sigmas=values(video_sigmas),
            audio_sigmas=values(audio_sigmas),
            update_rule=update_rule,
        )
        if schedule.nfe != profile.nfe:
            raise H3ConditioningBundleError(
                "H3 conditioning scheduler NFE differs from the profile"
            )
        (self.directory / _FILES["scheduler"]).write_text(
            json.dumps(schedule.to_mapping(), indent=2) + "\n",
            encoding="utf-8",
        )
        return seal_h3_conditioning_bundle(
            self.directory,
            bundle_id=bundle_id,
            profile=profile,
            request=request,
            source=source,
            schema_version=schema_version,
        )
