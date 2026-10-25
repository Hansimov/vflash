"""CPU-only request admission, shared by planning and complete inference.

Admission verifies a supported software envelope; it does not establish hardware
availability, prepared weights or measured performance on a prospective worker.
"""

from __future__ import annotations

from vflash.contracts import ContractError
from vflash.model_assets import supported_request_modes
from vflash.pipeline.contracts import VideoRequest


def validate_request(
    request: VideoRequest,
    *,
    profile_id: str,
    hybrid: bool = False,
    parallel_strategy: str = "single",
    attention_backend: str = "torch-flash",
    weight_residency: str = "block-ring",
) -> None:
    """Reject unsupported requests before model preparation or GPU procurement.

    Pass the resolved pipeline profile, strategy and attention, not a catalog GPU
    label. The live pipeline calls the same validator before activating stages.
    Input media and prepared-asset integrity are checked separately at execution.
    """
    if not isinstance(request, VideoRequest) or type(hybrid) is not bool:
        raise ContractError(
            "request admission requires VideoRequest and an explicit hybrid flag"
        )
    if weight_residency not in {"block-ring", "resident"}:
        raise ContractError("unknown weight residency")
    if request.reference_video is not None:
        native_ref = profile_id == "ref2va-turbo4-exact-sm89"
        hybrid_ref = (
            hybrid
            and profile_id
            in {"i2va-turbo4-v01-544-exact-sm89", "i2va-turbo4-v01-544-exact-sm120"}
            and attention_backend in {"veda-sm89", "veda-triton"}
        )
        if parallel_strategy != "single" or not (native_ref or hybrid_ref):
            raise ContractError(
                "video references require a single supported Ref4 or hybrid pipeline"
            )
        if request.reference_video_profile != "full" and not hybrid_ref:
            raise ContractError(
                "spatial video-reference compression requires a hybrid pipeline"
            )
        maximum_pixels = 1024**2 if hybrid_ref else 928 * 512
        if request.width * request.height > maximum_pixels or request.duration_seconds != 5:
            raise ContractError("video-reference output exceeds its five-second profile canvas")
    modes = supported_request_modes(profile_id)
    if hybrid:
        modes = (*modes, "ref2va")
    if request.mode not in modes:
        raise ContractError("the request mode differs from the prepared pipeline profile")
    from vflash.pipeline.residency import validate_resident_request

    validate_resident_request(weight_residency, request, profile_id, hybrid=hybrid)
    if request.width * request.height > 1024**2 and (
        not hybrid
        or parallel_strategy != "single"
        or attention_backend not in {"veda-sm89", "veda-triton"}
    ):
        raise ContractError(
            "native HD requires the original-v0.1 single-device hybrid Veda pipeline"
        )
    extended = (
        request.duration_seconds > 10
        or len(request.ordered_references) > 3
        or (request.mode == "ref2va" and request.duration_seconds != 5)
        or (request.width * request.height > 1024**2 and request.duration_seconds != 5)
    )
    if extended and (
        not hybrid
        or parallel_strategy != "single"
        or attention_backend not in {"veda-sm89", "veda-triton"}
    ):
        raise ContractError(
            "extended duration/references require the original-v0.1 "
            "single-device hybrid Veda pipeline"
        )
    if extended and (
        request.width * request.height * request.model_frames > 2048**2 * 124
        or (
            len(request.ordered_references) > 1
            and request.mode == "ref2va"
            and (request.duration_seconds != 5 or request.width * request.height > 960 * 544)
        )
    ):
        raise ContractError(
            "extended request exceeds the qualified joint canvas/duration/reference budget"
        )
