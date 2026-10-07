"""Non-owning official reference graph sharing one FL conditioner component set."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from traceback import clear_frames

from vflash.native.h3_hybrid import _digest, hybrid_reference_source


class HybridReferenceGraph:
    """A non-owning official Ref pipeline over one resident FL component set."""

    def __init__(self, owner, variant):
        from diffusers import MiniMaxH3ModularPipeline

        from vflash.adapters.modular_config import local_modular_config
        from vflash.adapters.references import install_match_reference_setup_block
        from vflash.pipeline.assets import conditioning_source

        if owner.prepared.profile_id not in {
            "i2va-turbo4-v01-544-exact-sm89",
            "fl2va-turbo4-v01-544-exact-sm89",
        }:
            raise ValueError("hybrid reference requires original FL v0.1")
        self.owner = owner
        model = owner.prepared.assets.model_directory
        self.pipe = MiniMaxH3ModularPipeline(
            pretrained_model_name_or_path=model,
            workflow="ref2va",
            modular_config_dict=local_modular_config(
                model, transformer_component="transformer_ref"
            ),
        )
        self.pipe.update_components(
            transformer_ref=owner.transformer,
            **{
                name: getattr(owner.pipe, name)
                for name in (
                    "text_encoder",
                    "tokenizer",
                    "processor",
                    "vae",
                    "audio_vae",
                    "scheduler",
                    "audio_scheduler",
                )
            },
        )
        if any(getattr(self.pipe, name, None) is None for name in self.pipe.component_names):
            raise ValueError("hybrid Ref graph has missing components")
        self.setup = install_match_reference_setup_block(self.pipe)
        self.pipe.set_progress_bar_config(disable=True)
        self.source = hybrid_reference_source(
            conditioning_source(profile_id=owner.prepared.profile_id),
            reference_revision=variant["reference_checkpoint_revision"],
            runtime_versions=owner.versions,
        )

    def capture(self, request, references, directory: Path):
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference

        from vflash.adapters.conditioning_capture import (
            H3ConditioningCaptureComplete,
            H3ConditioningCaptureSession,
        )
        from vflash.native.h3_conditioning_bundle import H3ConditioningProfile

        if (
            request.mode != "ref2va"
            or request.reference_video is not None
            or len(references) != len(request.ordered_references)
            or not 1 <= len(references) <= 9
            or not self.owner._cuda_active
        ):
            raise ValueError("resume the shared conditioner and supply ordered Ref images")
        capture = H3ConditioningCaptureSession(directory)
        try:
            capture.install(self.owner.transformer)
            try:
                self.pipe(
                    prompt=request.prompt,
                    height=request.height,
                    width=request.width,
                    num_frames=request.model_frames,
                    num_inference_steps=5,
                    generator=self.owner._torch.Generator().manual_seed(request.seed),
                    references=[MiniMaxH3ImageReference(image=r.image) for r in references],
                    output=["videos", "audio", "sampling_rate"],
                )
            except H3ConditioningCaptureComplete as complete:
                self.owner._torch.cuda.synchronize(self.owner.device)
                clear_frames(complete.__traceback__)
            else:
                raise ValueError("official Ref graph did not stop before native denoising")
            capture.close()
            video_prefix, audio_prefix = capture.prefix_counts()
            profile = H3ConditioningProfile(
                task="ref2va",
                width=request.width,
                height=request.height,
                frames=request.model_frames,
                nfe=4,
                video_flow_shift=12,
                audio_flow_shift=3,
                reference_token_budget=capture.reference_token_budget(
                    width=request.width, height=request.height, frames=request.model_frames
                ),
                num_condition_video_rows=video_prefix,
                num_condition_audio_rows=audio_prefix,
            )
            metadata = dict(
                source_case_id="complete-pipeline",
                prompt=request.prompt,
                prompt_sha256=hashlib.sha256(request.prompt.encode()).hexdigest(),
                seed=request.seed,
                reference_image_policy="match",
                references=[
                    dict(
                        picture_index=i,
                        role="reference",
                        size_bytes=r.size_bytes,
                        sha256=r.sha256,
                    )
                    for i, r in enumerate(references, 1)
                ],
                delivery_profiles=[
                    dict(
                        temporal_profile=f"native-24fps-{request.duration_seconds}s",
                        frames=request.delivery_frames,
                        fps=24,
                    )
                ],
            )
            identity = _digest(
                dict(request=metadata, profile=asdict(profile), source=self.source)
            )
            return capture.finish_in_memory(
                bundle_id="h3-conditioning-" + identity[:32],
                profile=profile,
                request=metadata,
                source=self.source,
                video_sigmas=self.pipe.scheduler.sigmas,
                audio_sigmas=self.pipe.audio_scheduler.sigmas,
                update_rule="training_euler",
            )
        finally:
            capture.discard()

    def close(self):
        """Drop borrowed references before the owning conditioner releases its weights."""
        self.pipe = self.owner = self.setup = None
