"""CPU layout and dispatch checks; CUDA qualification is a separate requirement."""

import pytest

from vflash.native.h3_sol_attention import protected_prefix_length

torch = pytest.importorskip("torch")


def layout():
    return {
        "text_indices": torch.tensor([0, 1]),
        "audio_indices": torch.tensor([3, 4]),
        "video_indices": torch.tensor([2, 5, 6, 7]),
    }


def test_protects_text_condition_and_audio_but_not_target_video():
    assert protected_prefix_length(layout(), 1) == (5, 8)


@pytest.mark.parametrize(
    "field,values",
    [
        ("video_indices", [2, 5, 7, 6]),
        ("text_indices", [0, 0]),
        ("audio_indices", [3, 5]),
    ],
)
def test_rejects_noncanonical_or_overlapping_layout(field, values):
    tensors = layout()
    tensors[field] = torch.tensor(values)
    with pytest.raises(ValueError, match="protected prefix"):
        protected_prefix_length(tensors, 1)


@pytest.mark.parametrize("count", [-1, 4])
def test_rejects_missing_target_video(count):
    with pytest.raises(ValueError, match="target-video"):
        protected_prefix_length(layout(), count)


def test_approximate_metadata_cannot_report_exact():
    from vflash.native.h3_native_conditioning_runtime import _attention_policy
    from vflash.native.h3_sol_attention import SolVideoAttention

    operator = object.__new__(SolVideoAttention)
    operator.prefix, operator.sequence_length, operator.calls = 5, 8, 800
    policy = _attention_policy(operator)
    assert policy["exact"] is False and policy["fallback_enabled"] is False
    assert policy["operator_calls"] == 800
    assert policy["effective_backends"] == ["cute_sm89", "torch-flash"]
    assert _attention_policy()["attention_backend"] == "torch-flash"
    pending = _attention_policy(configured_backend="sol-sm89")
    assert pending["exact"] is False and pending["operator_calls"] == 0
    assert pending["effective_backends"] == []
