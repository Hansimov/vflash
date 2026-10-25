from dataclasses import replace

import pytest
from test_h3_audio_hidden_cache import core_fixture

from vflash.native.h3_audio_hidden_cache import frozen_video_hidden
from vflash.native.h3_audio_kv_cache import frozen_video_kv

torch = pytest.importorskip("torch")


def test_kv_retains_the_frozen_context_with_fresh_audio_and_restores_blocks():
    engine, hidden, invocation = core_fixture(0)

    def run(scope):
        result = []
        with scope(engine) as record:
            for evaluation in range(6):
                values = hidden.clone()
                values[:, engine.audio_indices] += evaluation * 0.125
                for block in engine.denoiser.blocks:
                    values = block.forward_prevalidated(
                        values, replace(invocation, evaluation_index=evaluation)
                    )
                result.append(values)
        return result, record

    hidden_results, hidden_record = run(frozen_video_hidden)
    kv_results, record = run(frozen_video_kv)
    for a, b in zip(hidden_results, kv_results, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert record["build_block_calls"] == 3 and record["reuse_block_calls"] == 15
    assert record["rebuilt_projection_rows"] == 0 < hidden_record["rebuilt_projection_rows"]
    assert all(
        "forward_prevalidated" not in b.__dict__ and "_attention" not in b.__dict__
        for b in engine.denoiser.blocks
    )


def test_kv_budget_and_cross_evaluation_failures_restore_the_original_methods():
    engine, hidden, invocation = core_fixture(0)
    block = engine.denoiser.blocks[0]
    with (
        pytest.raises(ValueError, match="byte limit"),
        frozen_video_kv(engine, maximum_bytes=1),
    ):
        block.forward_prevalidated(hidden, invocation)
    assert "forward_prevalidated" not in block.__dict__ and "_attention" not in block.__dict__
    with pytest.raises(ValueError, match="skip evaluations"), frozen_video_kv(engine):
        block.forward_prevalidated(hidden, invocation)
        block.forward_prevalidated(hidden, replace(invocation, evaluation_index=2))
    assert "forward_prevalidated" not in block.__dict__ and "_attention" not in block.__dict__
