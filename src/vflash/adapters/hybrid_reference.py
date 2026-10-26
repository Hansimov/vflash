"""Non-owning official reference graph sharing one FL conditioner component set."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path
from traceback import clear_frames

from vflash.adapters.video_references import DecodedVideoReference
from vflash.hardware_profiles import V01_PROFILES
from vflash.native.h3_conditioning_bundle import (
    H3_VIDEO_REFERENCE_POLICY,
    H3_VIDEO_REFERENCE_SPATIAL_HALF_POLICY,
)
from vflash.native.h3_hybrid import _digest, hybrid_reference_source


class HybridReferenceGraph:
    """A non-owning official Ref pipeline over one resident FL component set."""

    def __init__(self, owner, variant):
        from diffusers import MiniMaxH3ModularPipeline

        from vflash.adapters.modular_config import local_modular_config
        from vflash.adapters.references import install_match_reference_setup_block
        from vflash.pipeline.assets import conditioning_source

        if owner.prepared.profile_id not in V01_PROFILES:
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
        from vflash.adapters.reference_cache import install_reference_latent_cache

        self.reference_cache = install_reference_latent_cache(self.pipe)
        self.last_capture_diagnostics = {}
        self.pipe.set_progress_bar_config(disable=True)
        self.reference_revision = variant["reference_checkpoint_revision"]
        self.source = hybrid_reference_source(
            conditioning_source(profile_id=owner.prepared.profile_id),
            reference_revision=variant["reference_checkpoint_revision"],
            runtime_versions=owner.versions,
        )

    def capture(self, request, references, directory: Path):
        from diffusers.modular_pipelines.minimax_h3 import (
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        from vflash.adapters.conditioning_capture import (
            H3ConditioningCaptureComplete,
            H3ConditioningCaptureSession,
        )
        from vflash.native.h3_conditioning_bundle import H3ConditioningProfile

        is_video = request.reference_video is not None
        image_count = len(request.ordered_references)
        mixed = is_video and image_count > 0
        expected = image_count + int(is_video)
        if (
            request.mode != "ref2va"
            or len(references) != expected
            or not 1 <= len(references) <= 9
            or any(isinstance(ref, DecodedVideoReference) for ref in references[:image_count])
            or (is_video and not isinstance(references[-1], DecodedVideoReference))
            or not self.owner._cuda_active
        ):
            raise ValueError(
                "resume the shared conditioner with ordered images then optional video"
            )
        source = self.source
        video_policy = (
            H3_VIDEO_REFERENCE_SPATIAL_HALF_POLICY
            if request.reference_video_profile == "spatial-half"
            else H3_VIDEO_REFERENCE_POLICY
        )
        if is_video:
            from vflash.pipeline.assets import conditioning_source

            source = hybrid_reference_source(
                conditioning_source(profile_id=self.owner.prepared.profile_id),
                reference_revision=self.reference_revision,
                runtime_versions=self.owner.versions,
                reference_policy=video_policy,
            )
        capture = H3ConditioningCaptureSession(directory)
        binding = self.reference_cache.bind(
            references[-1] if is_video else None,
            spatial_divisor=2 if video_policy == H3_VIDEO_REFERENCE_SPATIAL_HALF_POLICY else 1,
            context_key=(
                "mixed-match-v1",
                request.width,
                request.height,
                *(ref.sha256 for ref in references[:image_count]),
            )
            if mixed
            else (),
        )
        binding.__enter__()
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
                    references=[
                        MiniMaxH3VideoReference(frames=r.require_frames(), fps=24, audio=None)
                        if isinstance(r, DecodedVideoReference)
                        else MiniMaxH3ImageReference(image=r.image)
                        for r in references
                    ],
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
                ]
                if not is_video
                else [],
                delivery_profiles=[
                    dict(
                        temporal_profile=f"native-24fps-{request.duration_seconds}s",
                        frames=request.delivery_frames,
                        fps=24,
                    )
                ],
            )
            if is_video:
                reference = dict(references[-1].metadata)
                if video_policy == H3_VIDEO_REFERENCE_SPATIAL_HALF_POLICY:
                    reference["latent_spatial_divisor"] = 2
                    reference["condition_video_rows"] = (
                        reference["vae_latent_frames"]
                        * max(1, reference["width"] // 64)
                        * max(1, reference["height"] // 64)
                    )
                sizes = (self.setup.last_metadata or {}).get("resolved_images", [])
                image_rows = sum(v["width"] * v["height"] // 1024 for v in sizes)
                if mixed and len(sizes) != image_count:
                    raise ValueError("mixed reference image geometry is missing")
                if (video_prefix, audio_prefix) != (
                    reference["condition_video_rows"] + image_rows,
                    0,
                ):
                    raise ValueError("captured video prefix differs from the decoded reference")
                metadata.pop("reference_image_policy")
                metadata.update(reference_video_policy=video_policy, references=[reference])
                if mixed:
                    metadata.update(
                        reference_image_policy="match",
                        reference_images=[
                            dict(
                                picture_index=i,
                                role="reference",
                                size_bytes=r.size_bytes,
                                sha256=r.sha256,
                                **size,
                            )
                            for i, (r, size) in enumerate(
                                zip(references[:image_count], sizes, strict=True), 1
                            )
                        ],
                    )
            identity = _digest(dict(request=metadata, profile=asdict(profile), source=source))
            return capture.finish_in_memory(
                bundle_id="h3-conditioning-" + identity[:32],
                profile=profile,
                request=metadata,
                source=source,
                video_sigmas=self.pipe.scheduler.sigmas,
                audio_sigmas=self.pipe.audio_scheduler.sigmas,
                update_rule="training_euler",
                schema_version=6 if mixed else 2 if is_video else 1,
            )
        finally:
            self.last_capture_diagnostics = {
                "reference_latents": dict(self.reference_cache.diagnostics)
            }
            binding.__exit__(None, None, None)
            capture.discard()

    def close(self):
        """Drop borrowed references before the owning conditioner releases its weights."""
        self.reference_cache.clear()
        self.pipe = self.owner = self.setup = None
