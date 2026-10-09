import subprocess
import sys
from pathlib import Path

import pytest

from vflash.contracts import ContractError
from vflash.pipeline import VideoRequest, validate_request


@pytest.mark.parametrize("arch", ["sm89", "sm90", "sm120"])
def test_cpu_admission_distinguishes_individual_and_joint_limits(arch):
    options = dict(
        profile_id=f"i2va-turbo4-v01-544-exact-{arch}",
        hybrid=True,
        attention_backend="veda-triton",
    )
    frame = Path("/unused/source.png")
    validate_request(
        VideoRequest("Motion", first_frame=frame, width=2048, height=2048), **options
    )
    validate_request(
        VideoRequest("Motion", first_frame=frame, width=1024, height=1024, duration_seconds=15),
        **options,
    )
    with pytest.raises(ContractError, match="joint"):
        validate_request(
            VideoRequest(
                "Motion", first_frame=frame, width=2048, height=2048, duration_seconds=15
            ),
            **options,
        )
    refs = tuple(Path(f"/unused/ref-{i}.png") for i in range(9))
    validate_request(VideoRequest("Gallery", references=refs, width=960, height=544), **options)
    with pytest.raises(ContractError, match="joint"):
        validate_request(
            VideoRequest("Gallery", references=refs, width=1024, height=1024), **options
        )


def test_cpu_admission_keeps_residency_and_topology_constraints():
    request = VideoRequest(
        "Motion", first_frame=Path("/unused/frame.png"), width=2048, height=2048
    )
    options = dict(
        profile_id="i2va-turbo4-v01-544-exact-sm90",
        hybrid=True,
        attention_backend="veda-triton",
    )
    with pytest.raises(ContractError, match="resident I2VA"):
        validate_request(request, weight_residency="resident", **options)
    with pytest.raises(ContractError, match="native HD"):
        validate_request(request, parallel_strategy="sequence-head", **options)


def test_admission_import_does_not_load_accelerator_libraries():
    program = (
        "from vflash.pipeline import validate_request; import sys; "
        "assert 'torch' not in sys.modules; "
        "assert 'vflash.pipeline.runtime' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True, timeout=10)
