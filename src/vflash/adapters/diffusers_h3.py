"""Pinned official text/reference encoding adapted to the native Vflash boundary."""

from __future__ import annotations

import gc
import hashlib
import time
from importlib.metadata import version
from pathlib import Path
from traceback import clear_frames
from typing import Any

from vflash.adapters.conditioning_capture import (
    H3ConditioningCaptureComplete,
    H3ConditioningCaptureSession,
)
from vflash.adapters.conditioning_prefix import (
    apply_h3_conditioning_adapter,
    load_h3_conditioning_transformer,
)
from vflash.adapters.conditioning_vae import load_h3_image_conditioning_vae_components
from vflash.adapters.modular_config import local_modular_config
from vflash.adapters.references import DecodedReference, install_match_reference_setup_block
from vflash.contracts import ContractError
from vflash.native.h3_conditioning_bundle import H3ConditioningBundle, H3ConditioningProfile
from vflash.pipeline.assets import (
    PreparedPipelineAssets,
    canonical_sha256,
    conditioning_source,
)
from vflash.pipeline.contracts import VideoRequest
from vflash.pipeline.residency import capture_cpu_master, restore_cpu_master


def validate_adapter_dependencies() -> dict[str, str]:
    """Fail before loading model components when the tested adapter API is unavailable."""
    import diffusers
    import torch

    expected = {
        "transformers": "5.9.0",
        "accelerate": "1.13.0",
        "peft": "0.19.1",
        "huggingface-hub": "1.23.0",
    }
    actual = {name: version(name) for name in expected}
    if diffusers.__version__ != "0.40.0" or actual != expected:
        raise ContractError("install the pinned vflash pipeline extra for official H3 encoding")
    if torch.__version__.split("+")[0] != "2.11.0":
        raise ContractError("the complete pipeline preview requires PyTorch 2.11.0")
    return {**actual, "diffusers": diffusers.__version__, "torch": torch.__version__}


class DiffusersConditioner:
    """Own CPU model components and the temporary CUDA encoding residency.

    This is an official Diffusers/Transformers/PEFT adapter. It does not execute
    a Diffusers denoising block, decode a video, polish a prompt, or load local
    application settings. Construct and invoke serially in the pipeline process.
    """

    def __init__(self, prepared: PreparedPipelineAssets, *, device: str = "cuda:0") -> None:
        prepared.check_unchanged()
        self.versions = validate_adapter_dependencies()
        import torch

        self._torch = torch
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ContractError("the official conditioner requires a CUDA execution device")
        self.prepared = prepared
        self.pipe = self.transformer = self._reference_setup = None
        self._cpu_masters: tuple[tuple[Any, Any], ...] = ()
        self._text_groups: tuple[Any, ...] = ()
        self._onload_handle = None
        self._offload_installed = self._cuda_active = self._cuda_touched = False
        self._closed = self._released = False
        started = time.monotonic()
        try:
            self._load_components()
        except BaseException as exc:
            self.close()
            clear_frames(exc.__traceback__)
            raise
        self.initialization_seconds = time.monotonic() - started

    def _load_components(self) -> None:
        from accelerate import init_empty_weights
        from diffusers import (
            AutoencoderKLMiniMaxH3,
            AutoencoderKLMiniMaxH3Audio,
            MiniMaxH3ModularPipeline,
            MiniMaxH3Transformer3DModel,
        )
        from transformers import Qwen3VLForConditionalGeneration

        torch, model = self._torch, self.prepared.assets.model_directory
        self.pipe = MiniMaxH3ModularPipeline(
            pretrained_model_name_or_path=model,
            workflow="ref2va",
            modular_config_dict=local_modular_config(model),
        )
        prefix = load_h3_conditioning_transformer(
            MiniMaxH3Transformer3DModel,
            transformer_directory=model / "transformer_ref",
            device=torch.device("cpu"),
            torch_module=torch,
            init_empty_weights=init_empty_weights,
        )
        self.transformer = prefix.transformer
        self.adapter_metadata = apply_h3_conditioning_adapter(
            self.transformer,
            self.prepared.assets.adapter_path,
        )
        encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            str(model),
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        vae = load_h3_image_conditioning_vae_components(
            video_component_path=model / "vae",
            audio_component_path=model / "audio_vae",
            video_class=AutoencoderKLMiniMaxH3,
            audio_class=AutoencoderKLMiniMaxH3Audio,
            device=torch.device("cpu"),
            torch_module=torch,
            init_empty_weights=init_empty_weights,
        )
        self.pipe.update_components(
            text_encoder=encoder,
            vae=vae.video_vae,
            audio_vae=vae.audio_vae,
            transformer_ref=self.transformer,
        )
        self.pipe.load_components(
            names=["tokenizer", "processor", "scheduler", "audio_scheduler"],
            dtype=torch.bfloat16,
            pretrained_model_name_or_path=str(model),
            local_files_only=True,
        )
        self.pipe.update_components(
            scheduler=type(self.pipe.scheduler).from_config(
                self.pipe.scheduler.config, shift=12
            ),
            audio_scheduler=type(self.pipe.audio_scheduler).from_config(
                self.pipe.audio_scheduler.config,
                shift=3,
            ),
        )
        if self.pipe.text_encoder_layer != 50:
            raise ContractError("the official H3 encoder must consume hidden_states[50]")
        if any(getattr(self.pipe, name, None) is None for name in self.pipe.component_names):
            raise ContractError(
                "the official pipeline did not load every conditioning component"
            )
        self._reference_setup = install_match_reference_setup_block(self.pipe)
        for module in (self.transformer, encoder, self.pipe.vae, self.pipe.audio_vae):
            module.eval().requires_grad_(False)
        self._cpu_masters = tuple(
            (module, capture_cpu_master(module))
            for module in (
                self.transformer,
                self.pipe.vae.encoder,
                self.pipe.vae.quant_conv,
            )
        )
        self.pipe.set_progress_bar_config(disable=True)

    def _require_open(self) -> None:
        if self._closed:
            raise ContractError("the conditioner is closed")

    def _install_offload(self) -> None:
        from diffusers.hooks import apply_group_offloading

        apply_group_offloading(
            self.pipe.text_encoder.model,
            offload_type="leaf_level",
            onload_device=self.device,
            offload_device=self._torch.device("cpu"),
            use_stream=True,
            low_cpu_mem_usage=True,
        )
        groups = {}
        for module in self.pipe.text_encoder.model.modules():
            registry = getattr(module, "_diffusers_hook", None)
            hook = registry.get_hook("group_offloading") if registry is not None else None
            if hook is not None:
                groups[id(hook.group)] = hook.group
        self._text_groups = tuple(groups.values())
        self._offload_installed = True

    def resume_cuda(self) -> float:
        self._require_open()
        if self._cuda_active:
            return 0.0
        started = time.monotonic()
        self._cuda_touched = True
        try:
            if tuple(self._torch.cuda.get_device_capability(self.device)) != (8, 9):
                raise ContractError("this complete pipeline preview requires SM89")
            if not self._offload_installed:
                self._install_offload()
            self.pipe.vae.encoder.to(self.device)
            self.pipe.vae.quant_conv.to(self.device)

            def onload_transformer(module: Any, _args: tuple[Any, ...]) -> None:
                module.to(self.device)
                self._torch.cuda.synchronize(self.device)
                self._onload_handle.remove()
                self._onload_handle = None

            # Defer the prefix until Qwen has finished. This leaves room for
            # streamed text weights and intermediates on the shared device.
            self._onload_handle = self.transformer.register_forward_pre_hook(
                onload_transformer,
                prepend=True,
            )
            self._torch.cuda.synchronize(self.device)
        except BaseException:
            self._closed = True
            raise
        self._cuda_active = True
        return time.monotonic() - started

    def suspend_cuda(self) -> float:
        self._require_open()
        if not self._cuda_touched:
            return 0.0
        started = time.monotonic()
        try:
            self._torch.cuda.synchronize(self.device)
            if self._onload_handle is not None:
                self._onload_handle.remove()
                self._onload_handle = None
            # On normal completion these groups are already offloaded. The
            # explicit sweep also retires the final prefetched, unused leaf.
            for group in self._text_groups:
                group.offload_()
            for module, master in self._cpu_masters:
                restore_cpu_master(module, master)
        except BaseException:
            self._closed = True
            raise
        self._cuda_active = self._cuda_touched = False
        self._torch.cuda.empty_cache()
        return time.monotonic() - started

    def _invoke(self, request: VideoRequest, references: tuple[DecodedReference, ...]) -> None:
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

        self.pipe(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_frames=request.model_frames,
            num_inference_steps=5,
            generator=self._torch.Generator().manual_seed(request.seed),
            output=["videos", "audio", "sampling_rate"],
            references=[
                MiniMaxH3ImageReference(image=reference.image) for reference in references
            ],
        )

    def capture(
        self,
        request: VideoRequest,
        references: tuple[DecodedReference, ...],
        directory: Path,
    ) -> H3ConditioningBundle:
        self._require_open()
        if not self._cuda_active:
            raise ContractError("resume the conditioner before encoding a request")
        if len(references) != len(request.ordered_references):
            raise ContractError("decoded reference count differs from the request")
        if directory.exists() and any(directory.iterdir()):
            raise ContractError("conditioning output must be a new or empty directory")
        capture = H3ConditioningCaptureSession(directory)
        try:
            capture.install(self.transformer)
            try:
                self._invoke(request, references)
            except H3ConditioningCaptureComplete as complete:
                self._torch.cuda.synchronize(self.device)
                clear_frames(complete.__traceback__)
            else:
                raise ContractError("the official conditioner did not stop before denoising")
            capture.close()
            video_prefix, audio_prefix = capture.prefix_counts()
            source = conditioning_source(runtime_versions=self.versions)
            request_metadata = {
                "source_case_id": "vflash-live",
                "prompt": request.prompt,
                "prompt_sha256": hashlib.sha256(request.prompt.encode()).hexdigest(),
                "seed": request.seed,
                "reference_image_policy": "match",
                "delivery_profiles": [
                    {
                        "temporal_profile": "native-24fps-5s",
                        "frames": 120,
                        "fps": 24,
                    }
                ],
                "references": [
                    {
                        "picture_index": index,
                        "role": "reference",
                        "size_bytes": reference.size_bytes,
                        "sha256": reference.sha256,
                    }
                    for index, reference in enumerate(references, 1)
                ],
            }
            profile = H3ConditioningProfile(
                task="ref2va",
                width=request.width,
                height=request.height,
                frames=request.model_frames,
                nfe=4,
                video_flow_shift=12.0,
                audio_flow_shift=3.0,
                reference_token_budget=capture.reference_token_budget(
                    width=request.width,
                    height=request.height,
                    frames=request.model_frames,
                ),
                num_condition_video_rows=video_prefix,
                num_condition_audio_rows=audio_prefix,
            )
            from dataclasses import asdict

            identity = canonical_sha256(
                {
                    "request": request_metadata,
                    "profile": asdict(profile),
                    "source": source,
                }
            )
            return capture.finish(
                bundle_id=f"h3-conditioning-{identity[:32]}",
                profile=profile,
                request=request_metadata,
                source=source,
                video_sigmas=self.pipe.scheduler.sigmas,
                audio_sigmas=self.pipe.audio_scheduler.sigmas,
                update_rule="training_euler",
            )
        finally:
            capture.discard()

    def close(self) -> None:
        if self._released:
            return
        self._closed = True
        if self._cuda_touched:
            self._torch.cuda.synchronize(self.device)
        if self._onload_handle is not None:
            self._onload_handle.remove()
            self._onload_handle = None
        if self.pipe is not None and getattr(self.pipe, "text_encoder", None) is not None:
            for module in self.pipe.text_encoder.model.modules():
                registry = getattr(module, "_diffusers_hook", None)
                if registry is not None:
                    for name in (
                        "layer_execution_tracker",
                        "lazy_prefetch_group_offloading",
                        "group_offloading",
                    ):
                        registry.remove_hook(name, recurse=False)
        self._cpu_masters = self._text_groups = ()
        self.pipe = self.transformer = self._reference_setup = None
        self._cuda_active = self._cuda_touched = False
        self._released = True
        gc.collect()
