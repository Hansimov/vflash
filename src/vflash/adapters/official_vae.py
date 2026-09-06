"""LightX-independent adapters for the official MiniMax H3 VAE components."""

from __future__ import annotations

import hashlib
import importlib
import importlib.machinery
import json
import math
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vflash.native.errors import VflashNativeError


class H3NativeVAEError(VflashNativeError):
    """The official H3 VAE bundle or its latent contract is invalid."""


_PYTHON_REFERENCE = re.compile(
    r"(?P<module>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\."
    r"(?P<class_name>[A-Za-z_]\w*)\Z"
)


@dataclass(frozen=True)
class H3OfficialVAEComponent:
    """One loaded official remote-code component plus its pinned local config."""

    path: Path
    class_reference: str
    config: dict[str, Any]
    module: Any


def load_h3_vae_config(component_path: Path) -> dict[str, Any]:
    """Load and validate the common H3 VAE normalization contract."""

    component_path = component_path.resolve(strict=True)
    if not component_path.is_dir():
        raise H3NativeVAEError(f"H3 VAE component is not a directory: {component_path}")
    config_path = component_path / "config.json"
    try:
        value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise H3NativeVAEError(f"cannot load H3 VAE config: {config_path}") from exc
    if not isinstance(value, dict):
        raise H3NativeVAEError(f"H3 VAE config is not an object: {config_path}")
    channels = value.get("latent_channels")
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise H3NativeVAEError(f"H3 VAE latent_channels is invalid: {config_path}")
    for name in ("latents_mean", "latents_std"):
        values = value.get(name)
        if (
            not isinstance(values, list)
            or len(values) != channels
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                or (name == "latents_std" and item <= 0)
                for item in values
            )
        ):
            raise H3NativeVAEError(
                f"H3 VAE {name} must contain {channels} numeric values: {config_path}"
            )
    return value


def _remote_class_reference(config: dict[str, Any], component_path: Path) -> str:
    auto_map = config.get("auto_map")
    reference = auto_map.get("AutoModel") if isinstance(auto_map, dict) else None
    if not isinstance(reference, str) or _PYTHON_REFERENCE.fullmatch(reference) is None:
        raise H3NativeVAEError(
            f"{component_path / 'config.json'} must define a safe auto_map.AutoModel"
        )
    return reference


def _component_package(component_path: Path) -> str:
    identity = hashlib.sha256(str(component_path).encode()).hexdigest()[:20]
    package_name = f"_vflash_h3_vae_{identity}"
    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_paths = tuple(str(value) for value in getattr(existing, "__path__", ()))
        if existing_paths != (str(component_path),):
            raise H3NativeVAEError(f"dynamic H3 VAE package collision: {component_path}")
        return package_name

    package = types.ModuleType(package_name)
    package.__file__ = str(component_path / "__init__.py")
    package.__package__ = package_name
    package.__path__ = [str(component_path)]
    spec = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
    spec.submodule_search_locations = [str(component_path)]
    package.__spec__ = spec
    sys.modules[package_name] = package
    return package_name


def load_official_h3_vae_component(
    component_path: Path,
    *,
    torch_module: Any,
    trust_local_code: bool = False,
) -> H3OfficialVAEComponent:
    """Load the official self-contained remote-code bundle without LightX/vLLM.

    The component is always constructed on CPU. Callers explicitly choose when
    to move the multi-gigabyte module to an accelerator, which keeps 20 GB cards
    usable through sequential video/audio residency.
    """

    if trust_local_code is not True:
        raise H3NativeVAEError("loading the official VAE requires trust_local_code=True")
    component_path = component_path.resolve(strict=True)
    config = load_h3_vae_config(component_path)
    reference = _remote_class_reference(config, component_path)
    matched = _PYTHON_REFERENCE.fullmatch(reference)
    assert matched is not None
    package_name = _component_package(component_path)
    qualified_module = f"{package_name}.{matched.group('module')}"
    try:
        remote_module = importlib.import_module(qualified_module)
        component_class = getattr(remote_module, matched.group("class_name"))
        from_pretrained = component_class.from_pretrained
    except (AttributeError, ImportError, OSError) as exc:
        raise H3NativeVAEError(
            f"cannot import official H3 VAE class {reference} from {component_path}"
        ) from exc

    # This library never patches torch.nn.Module globally. The pinned official
    # loader builds on CPU; its cold allocation is released before GPU upload.
    try:
        with torch_module.device("cpu"):
            component = from_pretrained(str(component_path))
    except Exception as exc:
        raise H3NativeVAEError(
            f"cannot construct official H3 VAE class {reference} from {component_path}"
        ) from exc
    return H3OfficialVAEComponent(
        path=component_path,
        class_reference=reference,
        config=config,
        module=component,
    )


def _normalization_tensor(
    values: list[int | float],
    *,
    latent: Any,
    channels: int,
    dimensions: int,
    torch_module: Any,
) -> Any:
    shape = (1, channels, *(1 for _ in range(dimensions - 2)))
    return torch_module.tensor(
        values,
        device=latent.device,
        dtype=latent.dtype,
    ).view(*shape)


def prepare_official_h3_video_decoder(
    component: H3OfficialVAEComponent,
    *,
    device: Any,
    torch_module: Any,
) -> None:
    """Activate the official video decoder with its consumer-GPU dtype split."""

    remote = component.module
    marker = (str(device), "fp32-boundaries-fp16-linear-storage-v1")
    if getattr(remote, "_vflash_decoder_preparation", None) == marker:
        return
    model = getattr(remote, "model", None)
    if model is None:
        raise H3NativeVAEError("official H3 video VAE does not expose its model")
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise H3NativeVAEError("official H3 video VAE does not expose its decoder")

    # Keep normalization and residual boundaries in FP32, but store the bulk
    # decoder GEMM path in FP16. This mirrors the released consumer-GPU VAE
    # precision split without importing a framework-specific VAE wrapper.
    remote.eval().requires_grad_(False)
    post_quant_conv = getattr(model, "post_quant_conv", None)
    if post_quant_conv is not None:
        post_quant_conv.to(dtype=torch_module.float16)
    for child in decoder.modules():
        if isinstance(child, torch_module.nn.Linear):
            child.to(dtype=torch_module.float16)
    # Cast the released mixed-precision storage contract before the device
    # transfer.  Uploading the complete FP32 decoder first creates a needless
    # multi-GiB cold-start peak even though those Linear weights are immediately
    # narrowed and never execute in FP32.
    remote.to(device=device)
    remote._vflash_decoder_preparation = marker


def prepare_official_h3_audio_decoder(
    component: H3OfficialVAEComponent,
    *,
    device: Any,
    torch_module: Any,
) -> None:
    """Activate the smaller official audio decoder in its FP32 contract."""

    remote = component.module
    marker = (str(device), "fp32-v1")
    if getattr(remote, "_vflash_decoder_preparation", None) == marker:
        return
    remote.eval().requires_grad_(False).to(device=device, dtype=torch_module.float32)
    remote._vflash_decoder_preparation = marker


def decode_official_h3_video_latents(
    component: H3OfficialVAEComponent,
    latents: Any,
    *,
    device: Any,
    torch_module: Any,
) -> Any:
    """Decode normalized ``[B,24,T,H,W]`` H3 video latents to FP32 RGB."""

    channels = int(component.config["latent_channels"])
    if getattr(latents, "ndim", None) != 5 or int(latents.shape[1]) != channels:
        raise H3NativeVAEError(
            f"video latents must have shape [batch,{channels},frames,height,width]"
        )
    remote = component.module
    model = getattr(remote, "model", None)
    processor = getattr(model, "processor", None)
    if model is None or not callable(getattr(model, "decode_base", None)) or processor is None:
        raise H3NativeVAEError("official H3 video VAE does not expose decode_base/processor")

    prepare_official_h3_video_decoder(
        component,
        device=device,
        torch_module=torch_module,
    )
    latents = latents.to(device=device, dtype=torch_module.float32)
    mean = _normalization_tensor(
        component.config["latents_mean"],
        latent=latents,
        channels=channels,
        dimensions=5,
        torch_module=torch_module,
    )
    std = _normalization_tensor(
        component.config["latents_std"],
        latent=latents,
        channels=channels,
        dimensions=5,
        torch_module=torch_module,
    )
    autocast_enabled = getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"
    with (
        torch_module.inference_mode(),
        torch_module.autocast(
            device_type="cuda",
            # Match the released H3 VAE contract: bulk convolution/matmul in FP16
            # while latent statistics and sensitive boundary weights remain FP32.
            dtype=torch_module.float16,
            enabled=autocast_enabled,
        ),
    ):
        decoded = model.decode_base(latents * std + mean)
    frames = processor.revert_tensor(decoded)
    if frames.ndim == 4:
        frames = frames.unsqueeze(0).transpose(1, 2)
    if frames.ndim != 5 or int(frames.shape[1]) != 3:
        raise H3NativeVAEError(
            f"official H3 video VAE returned an invalid shape: {tuple(frames.shape)}"
        )
    return frames.float().contiguous()


def decode_official_h3_audio_latents(
    component: H3OfficialVAEComponent,
    latents: Any,
    *,
    device: Any,
    torch_module: Any,
) -> Any:
    """Decode normalized mono-as-batch H3 audio latents to stereo FP32."""

    channels = int(component.config["latent_channels"])
    if getattr(latents, "ndim", None) != 3 or int(latents.shape[1]) != channels:
        raise H3NativeVAEError(
            f"audio latents must have shape [channels-as-batch,{channels},frames]"
        )
    remote = component.module
    if not callable(getattr(remote, "decode", None)):
        raise H3NativeVAEError("official H3 audio VAE does not expose decode")
    prepare_official_h3_audio_decoder(
        component,
        device=device,
        torch_module=torch_module,
    )
    latents = latents.to(device=device, dtype=torch_module.float32)
    mean = _normalization_tensor(
        component.config["latents_mean"],
        latent=latents,
        channels=channels,
        dimensions=3,
        torch_module=torch_module,
    )
    std = _normalization_tensor(
        component.config["latents_std"],
        latent=latents,
        channels=channels,
        dimensions=3,
        torch_module=torch_module,
    )
    with (
        torch_module.inference_mode(),
        torch_module.autocast(
            device_type="cuda",
            enabled=False,
        ),
    ):
        waveform = remote.decode(latents * std + mean)
    if waveform.ndim != 3 or int(waveform.shape[1]) != 1:
        raise H3NativeVAEError(
            f"official H3 audio VAE returned an invalid shape: {tuple(waveform.shape)}"
        )
    return waveform.permute(1, 0, 2).contiguous().float()


__all__ = [
    "H3NativeVAEError",
    "H3OfficialVAEComponent",
    "decode_official_h3_audio_latents",
    "decode_official_h3_video_latents",
    "load_h3_vae_config",
    "load_official_h3_vae_component",
    "prepare_official_h3_audio_decoder",
    "prepare_official_h3_video_decoder",
]
