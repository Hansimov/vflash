from types import SimpleNamespace

import pytest

from vflash.pipeline.residency import (
    CPUResidencyError,
    capture_cpu_master,
    staged_text_conditioning,
)


@pytest.mark.parametrize("failure", [None, "upload", "conditioning"])
def test_staged_text_retains_original_storages_across_requests(monkeypatch, failure):
    torch = pytest.importorskip("torch")
    encoder = torch.nn.Linear(2, 2, bias=False)
    encoder.register_buffer("phase", torch.arange(2), persistent=False)
    master = capture_cpu_master(encoder)
    pointer = encoder.weight.data_ptr()
    phase_pointer = encoder.phase.data_ptr()
    initial = encoder.weight.detach().clone()
    calls = []

    def upload(*, device):
        assert device == "cuda:0"
        calls.append(device)
        encoder.weight = torch.nn.Parameter(initial.clone())
        encoder.phase = encoder.phase.clone()
        if failure == "upload":
            raise RuntimeError("upload failed")
        return encoder

    monkeypatch.setattr(encoder, "to", upload)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda *_: master.tensor_bytes if encoder.weight.data_ptr() != pointer else 0,
    )
    for _ in range(2):
        if failure:
            with (
                pytest.raises(RuntimeError, match="failed"),
                staged_text_conditioning(encoder, master=master, enabled=True),
            ):
                raise RuntimeError("conditioning failed")
        else:
            with staged_text_conditioning(encoder, master=master, enabled=True) as record:
                assert encoder.weight.data_ptr() != pointer
                assert torch.equal(encoder.weight, initial)
            assert record["released_cuda_bytes"] == master.tensor_bytes
            assert record["retained_cpu_bytes"] == master.tensor_bytes
        assert encoder.weight.data_ptr() == pointer
        assert encoder.phase.data_ptr() == phase_pointer
        assert torch.equal(encoder.weight, initial)
    assert len(calls) == 2


def test_resident_is_unchanged_and_staging_requires_a_cpu_owner():
    untouched = SimpleNamespace()
    with staged_text_conditioning(untouched) as report:
        assert report == {"policy": "resident", "applied": False}
    with (
        pytest.raises(CPUResidencyError, match="original CPU"),
        staged_text_conditioning(untouched, enabled=True),
    ):
        pass
    with (
        pytest.raises(ValueError, match="boolean"),
        staged_text_conditioning(untouched, enabled="yes"),
    ):
        pass
