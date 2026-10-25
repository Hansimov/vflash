import threading
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from vflash.media.encoding import MediaError
from vflash.media.overlap import AudioVideoOverlap
from vflash.media.runtime import DecodedVideo


def decoder(*, free=48 * 1024**3, fail=False):
    state = {"active": False, "resumes": 0}
    cuda = SimpleNamespace(
        mem_get_info=lambda _: (free, free),
        memory_reserved=lambda _: 0,
        memory_allocated=lambda _: 0,
        Stream=lambda **_: SimpleNamespace(synchronize=lambda: None),
        device=lambda _: nullcontext(),
        stream=lambda _: nullcontext(),
    )
    value = SimpleNamespace(
        device="cuda:0", _torch=SimpleNamespace(cuda=cuda, inference_mode=nullcontext)
    )

    def resume(**kwargs):
        state["resumes"] += 1
        return 0.01

    def decode(latents, **kwargs):
        state["active"] = True
        state["thread"] = threading.get_ident()
        try:
            if fail:
                raise RuntimeError("decoder failure")
            return DecodedVideo(value, latents, seconds=0.01, **kwargs)
        finally:
            state["active"] = False

    value.resume_cuda, value.decode_video = resume, decode
    return value, state


def test_overlapped_result_has_one_consumer_and_original_owner():
    owner, state = decoder()
    pixels = object()
    with AudioVideoOverlap(owner, width=512, height=512) as overlap:
        overlap.submit(pixels)
        result = overlap.finish()
    assert state["thread"] != threading.get_ident() and not state["active"]
    with pytest.raises(MediaError):
        result.consume(object(), width=512, height=512)
    with pytest.raises(MediaError):
        result.consume(owner, width=1024, height=512)
    assert result.consume(owner, width=512, height=512) is pixels
    with pytest.raises(MediaError):
        result.consume(owner, width=512, height=512)


def test_headroom_skip_keeps_serial_decoder_available():
    owner, state = decoder(free=1)
    with AudioVideoOverlap(owner, width=1024, height=1024) as overlap:
        overlap.submit(object())
        assert overlap.finish() is None
    assert state["resumes"] == 0 and overlap.executor is None
    assert overlap.record["reason"] == "device_headroom"


def test_decoder_failure_is_joined_before_context_exits():
    owner, state = decoder(fail=True)
    with (
        pytest.raises(RuntimeError, match="decoder failure"),
        AudioVideoOverlap(owner, width=512, height=512) as overlap,
    ):
        overlap.submit(object())
    assert not state["active"] and overlap.future.done()
    assert all(not thread.is_alive() for thread in overlap.executor._threads)


def test_main_failure_keeps_its_identity_and_joins_decoder():
    owner, state = decoder(fail=True)
    with (
        pytest.raises(ValueError, match="audio failure") as caught,
        AudioVideoOverlap(owner, width=512, height=512) as overlap,
    ):
        overlap.submit(object())
        raise ValueError("audio failure")
    assert not state["active"] and overlap.future.done()
    assert any("RuntimeError" in note for note in caught.value.__notes__)
