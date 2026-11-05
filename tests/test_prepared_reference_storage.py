"""Prepared model storage moves retain compact descriptors and explicit partitions."""

from types import SimpleNamespace

import pytest
import torch

from vflash.adapters.h3_reference_assets import CachedMatrix, CachedProjection, _trunk_directory
from vflash.native.h3_fp8_linear import RowFP8Execution
from vflash.pipeline.residency import capture_cpu_master, restore_cpu_master


def test_projection_clone_and_restore_never_expand_the_original_matrix():
    original = torch.nn.Linear(4096, 8192, device="meta")
    projection = CachedProjection(original, SimpleNamespace(), 0, device="cpu")
    master = capture_cpu_master(projection)
    assert master.tensor_bytes == 2
    projection._apply(lambda value: value.clone())
    assert projection.weight.shape == original.weight.shape
    assert projection.weight.untyped_storage().nbytes() == 2
    assert (
        projection.bias.untyped_storage().data_ptr()
        == projection.weight.untyped_storage().data_ptr()
    )
    assert projection.weight.data_ptr() != master.state_dict["weight"].data_ptr()
    restore_cpu_master(projection, master)
    assert projection.weight.data_ptr() == master.state_dict["weight"].data_ptr()
    with pytest.raises(ValueError, match="BF16 descriptor"):
        projection.to(dtype=torch.float32)
    assert projection.weight.dtype == torch.bfloat16
    assert capture_cpu_master(projection).tensor_bytes == 2


def test_prepared_matrix_can_retain_its_cpu_storage_before_upload():
    base = torch.nn.Linear(2, 2, bias=False, device="meta")
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(torch.float8_e4m3fn)
    scales = torch.ones((2, 1), dtype=torch.float32)
    layer = CachedMatrix(base, weights, scales, RowFP8Execution("weight-only"), device="cpu")
    assert layer.weight.data_ptr() == weights.data_ptr()
    assert layer.row_scale.data_ptr() == scales.data_ptr()
    value = torch.tensor([[2.0, 1.0]], dtype=torch.bfloat16)
    assert torch.equal(layer(value), torch.tensor([[4.0, 10.0]], dtype=torch.bfloat16))


def test_prepared_trunks_keep_separate_prefixes(tmp_path):
    assert _trunk_directory(tmp_path, "transformer").name == "transformer"
    assert _trunk_directory(tmp_path, "transformer_ref").name == "transformer_ref"
    with pytest.raises(ValueError, match="explicit FL or Ref"):
        _trunk_directory(tmp_path, "../transformer")
