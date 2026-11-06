# ruff: noqa: E402
import threading
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
from vflash.adapters.h3_qad import (
    QAD_REPOSITORY,
    QAD_REVISION,
    H3QADVisualWeights,
    validate_manifest,
)
from vflash.pipeline.pika_flex import PIKA_ADAPTER_SHA256, PIKA_BASE_REVISION


def manifest():
    return dict(
        schema=1,
        source_repository=QAD_REPOSITORY,
        source_revision=QAD_REVISION,
        partition="transformer",
        pika_adapter_already_merged=True,
        merged_adapter_sha256=PIKA_ADAPTER_SHA256["fl2va"],
        projection_base_revision=PIKA_BASE_REVISION,
        projection_relation="all-106-source-tensors-byte-identical-v1",
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(partition="transformer_ref"),
        dict(pika_adapter_already_merged=False),
        dict(source_revision="other"),
        dict(projection_relation="assumed_equal"),
    ],
)
def test_wrong_partition_or_unproved_projection_reuse_is_rejected(change):
    validate_manifest(manifest())
    with pytest.raises(ValueError, match="projection provenance"):
        validate_manifest({**manifest(), **change})


@pytest.mark.parametrize("fail_inside", [False, True])
def test_real_scope_restores_parameters_and_methods_on_any_failure(fail_inside):
    main = torch.nn.Linear(2, 2, bias=False)
    main.weight = torch.nn.Parameter(torch.eye(2), requires_grad=False)
    auxiliary = torch.nn.Linear(2, 1, bias=False)
    auxiliary.weight = torch.nn.Parameter(torch.ones(1, 2), requires_grad=False)
    old_parameter = auxiliary.weight
    candidate = torch.nn.Parameter(torch.tensor([[3.0, 4.0]]), requires_grad=False)
    scope = H3QADVisualWeights.__new__(H3QADVisualWeights)
    scope.model = torch.nn.Module().eval()
    scope.closed = scope.active = False
    scope.owner = threading.Lock()
    from vflash.native.h3_nvfp4_linear import NVFP4Weight

    weight = NVFP4Weight(
        torch.zeros((128, 32), dtype=torch.uint8),
        torch.zeros((128, 4), dtype=torch.uint8),
        torch.tensor(1.0),
        torch.tensor(1.0),
    )
    scope.cpu_matrices = {"one": (main, weight)}
    scope.cpu_dense = [(auxiliary, "weight", candidate)]
    scope.device = torch.device("cpu")
    scope.matrices, scope.dense = {}, []
    scope.refiner_calls = {}
    scope.execution = NS(contract="fixture", linear=lambda x, _: x * 9)
    value = torch.tensor([[2.0, 3.0]])
    with (
        pytest.raises(RuntimeError if fail_inside else ValueError),
        scope.selected(adapters=(NS(strength=0), NS(strength=0))) as record,
    ):
        assert torch.equal(main(value), value * 9)
        assert auxiliary(value).item() == 18
        if fail_inside:
            raise RuntimeError("caller failed during picture generation")
        # Incomplete execution fails its original coverage check after yielding.
    assert not scope.active and not scope.owner.locked() and record["restored"]
    assert auxiliary.weight is old_parameter and auxiliary(value).item() == 5
    assert torch.equal(main(value), value) and "forward" not in main.__dict__
    assert not scope.matrices and not scope.dense
    assert scope.cpu_matrices and scope.cpu_dense
    assert record["released_visual_bytes"] > 0 and record["visual_copies_after_scope"] == 0
    # A second visual scope rematerializes the immutable source, then drops the
    # copies before the original model is consumed again (including on failure).
    with pytest.raises(ValueError), scope.selected(adapters=(NS(strength=0),)):
        assert torch.equal(main(value), value * 9)
    assert not scope.matrices and not scope.dense and auxiliary.weight is old_parameter
    with (
        pytest.raises(ValueError, match="second Pika or Larry"),
        scope.selected(adapters=(NS(strength=1),)),
    ):
        raise AssertionError("active adapter must fail before changing any weight")
    assert auxiliary.weight is old_parameter and not scope.owner.locked()


def test_partial_visual_materialization_drops_copies_and_keeps_mapped_sources():
    scope = H3QADVisualWeights.__new__(H3QADVisualWeights)
    scope.model = torch.nn.Module().eval()
    scope.closed = scope.active = False
    scope.owner = threading.Lock()
    scope.device = torch.device("cpu")
    layer = torch.nn.Linear(2, 1, bias=False)
    original = layer.weight
    scope.cpu_dense = [(layer, "weight", torch.ones(1, 2))]

    def fail_copy(_device):
        raise RuntimeError("fixture device allocation failure")

    scope.cpu_matrices = {"failed": (layer, NS(packed=NS(to=fail_copy)))}
    scope.matrices, scope.dense = {}, []
    scope.execution = NS(contract="fixture")
    with (
        pytest.raises(RuntimeError, match="allocation failure"),
        scope.selected(adapters=(NS(strength=0),)),
    ):
        raise AssertionError("Incomplete weights must not reach visual execution")
    assert layer.weight is original
    assert not scope.matrices and not scope.dense
    assert scope.cpu_matrices and scope.cpu_dense
    assert not scope.active and not scope.owner.locked()
