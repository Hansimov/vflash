"""Optional source-region conditioning for an owned Remake runtime.

Black in a canvas-sized static mask preserves the source; white permits editing.
The existing H3 sampler injects clean regions at its visual condition timestep.
This module neither detects regions nor composites decoded output pixels.
"""


def preserved_video_latent(video, noise_mask, source, edit_mask):
    """Keep generated history and expose the source in newly preserved regions."""
    import torch
    import torch.nn.functional as functional

    if (
        source.shape != video.shape
        or noise_mask.shape != video.shape
        or edit_mask.ndim != 4
        or edit_mask.shape[0] != 1
        or edit_mask.shape[-1] < 3
        or edit_mask.shape[1:3] != (video.shape[-2] * 16, video.shape[-1] * 16)
        or not torch.isfinite(edit_mask).all()
        or edit_mask.min() < 0
        or edit_mask.max() > 1
    ):
        raise ValueError("Use one static source mask on the exact prepared video canvas")
    # A latent cell intersecting the edited region remains editable. H3 itself
    # pools this mask to its token grid and uses the trained condition timestep.
    mask = edit_mask[..., :3].amax(dim=-1).unsqueeze(1)
    mask = functional.adaptive_max_pool2d(mask, video.shape[-2:]).unsqueeze(2)
    mask = mask.to(device=noise_mask.device, dtype=noise_mask.dtype)
    mask = mask.expand(video.shape[0], 1, video.shape[2], *video.shape[-2:])
    newly_preserved = (mask < 1) & (noise_mask > 0)
    samples = torch.where(newly_preserved, source.to(video), video)
    return samples, torch.minimum(noise_mask, mask)


class VFlashRemakeSourceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "edit_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Vflash/Remake"

    def run(self, positive, latent, edit_mask):
        from comfy.nested_tensor import NestedTensor

        refs = positive[0][1].get("minimax_refs") or []
        if not refs or refs[0].get("kind") != "video":
            raise ValueError("The original source video must be the first Remake reference")
        video, audio = latent["samples"].unbind()
        video_mask, audio_mask = latent["noise_mask"].unbind()
        video, video_mask = preserved_video_latent(
            video,
            video_mask,
            refs[0]["latent"],
            edit_mask,
        )
        return (
            {
                **latent,
                "samples": NestedTensor((video, audio)),
                "noise_mask": NestedTensor((video_mask, audio_mask)),
            },
        )


NODE_CLASS_MAPPINGS = {"VFlashRemakeSourceMask": VFlashRemakeSourceMask}
