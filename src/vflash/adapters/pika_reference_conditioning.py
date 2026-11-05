"""Ordered image references over the released Pika geometry and FP32 policy.

The caller owns the quantized text encoder, original FP32 VAE encoder and GPU.
This adapter does not turn content references into temporal keyframes.
"""

from __future__ import annotations

from types import SimpleNamespace

from vflash.contracts import ContractError


def reference_conditioning(
    pipe, request, *, source, conditioning, reference_type, device="cuda:0"
):
    import torch
    from PIL import Image

    if (
        request.mode != "ref2va"
        or request.reference_video is not None
        or not 1 <= len(request.ordered_references) <= 9
        or request.duration_seconds != 5
        or request.width * request.height > 1024**2
    ):
        raise ContractError(
            "Prepared Ref conditioning requires 1-9 ordered images and five seconds within 1 MP"
        )
    if pipe.vae.dtype != torch.float32 or torch.is_autocast_enabled("cuda"):
        raise ContractError("Ref conditioning requires the original FP32 VAE outside autocast")
    geometry = conditioning.resolve_geometry(
        conditioning.Request(
            task="ref2va",
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_frames=124,
        )
    )
    references, sizes = [], []
    for path in request.ordered_references:
        with Image.open(path) as original:
            image, size = conditioning.fit_reference_image(
                original.convert("RGB"), geometry, "tiered"
            )
        references.append(reference_type(image=image))
        sizes.append(size)
    with torch.inference_mode():
        embeddings, tags = pipe.encode_prompt(
            request.prompt, references=references, device=device, dtype=torch.bfloat16
        )
        latents, audio = pipe.encode_reference_latents(references, device=device)
    if audio or len(latents) != len(references):
        raise ContractError("The image reference encoder changed the ordered modalities")
    entries = []
    rows = 0
    for latent, size in zip(latents, sizes, strict=True):
        height, width = size["encoded"]
        if latent.dtype != torch.float32 or tuple(latent.shape) != (
            1,
            24,
            1,
            height // 16,
            width // 16,
        ):
            raise ContractError("Reference latent geometry or FP32 precision differs")
        external = source.packing.video_storage_to_external(latent[0]).unsqueeze(0)
        entries.append({"kind": "image", "video_latents": external.contiguous()})
        rows += height // 32 * (width // 32)
    cond = source.numerics.prepare_conditioning(
        {
            "prompt_embeds": embeddings,
            "text_token_tags": tags,
            "geometry": geometry,
            "keyframes": [],
            "references": entries,
        },
        device,
        policy=source.numerics.FP32_CONDITIONING,
    )
    return SimpleNamespace(
        conditional_dict=cond,
        metadata={
            "mode": "ref2va",
            "reference_order": ["image"] * len(entries),
            "reference_images": sizes,
            "reference_video_rows": rows,
            "numerics_policy": source.numerics.FP32_CONDITIONING,
            "temporal_keyframes": 0,
        },
    )
