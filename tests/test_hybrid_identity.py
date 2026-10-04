from types import SimpleNamespace

import pytest

from vflash.native.h3_hybrid import (
    hybrid_reference_source,
    validate_hybrid_reference_bundle,
)


def source():
    return hybrid_reference_source(
        dict(
            model_repository="official",
            model_revision="r",
            oracle="oracle",
            oracle_revision="o",
            oracle_profile="i2va-adapter-bf16-sm89",
            transformer_sha256="a" * 64,
        ),
        reference_revision="ref-r",
    )


def bundle():
    clock = SimpleNamespace(to_mapping=lambda: {"nfe": 4})
    return SimpleNamespace(
        profile=SimpleNamespace(
            task="ref2va",
            nfe=4,
            video_flow_shift=12,
            audio_flow_shift=3,
            num_condition_video_rows=1500,
            num_condition_audio_rows=0,
        ),
        schema_version=1,
        request={"references": [1, 2, 3]},
        schedule=clock,
        source=source(),
    )


def test_reference_identity_is_not_official_fl_or_ref_alias():
    value = source()
    assert value["transformer_sha256"] != "a" * 64
    assert value["oracle_profile"] == "ref2va-hybrid-fl-v01-bf16-sm89"
    assert (
        hybrid_reference_source(
            dict(
                model_repository="official",
                model_revision="r",
                oracle="oracle",
                oracle_revision="o",
                oracle_profile="i2va-adapter-bf16-sm89",
                transformer_sha256="a" * 64,
            ),
            reference_revision="another",
        )["transformer_sha256"]
        != value["transformer_sha256"]
    )


def test_real_reference_contract_accepts_runtime_provenance_change_only():
    b = bundle()
    b.source["oracle_runtime_sha256"] = "b" * 64
    validate_hybrid_reference_bundle(b, source=source(), schedule=b.schedule)


@pytest.mark.parametrize(
    "field,value",
    [
        ("task", "i2va"),
        ("nfe", 8),
        ("video_flow_shift", 7),
        ("audio_flow_shift", 2),
        ("num_condition_video_rows", 0),
        ("num_condition_audio_rows", 2),
    ],
)
def test_wrong_modes_clocks_and_modality_fail(field, value):
    b = bundle()
    setattr(b.profile, field, value)
    with pytest.raises(ValueError):
        validate_hybrid_reference_bundle(b, source=source(), schedule=b.schedule)


@pytest.mark.parametrize(
    "field", ["transformer_sha256", "oracle_profile", "model_revision", "oracle_config_sha256"]
)
def test_wrong_combination_identity_fails(field):
    b = bundle()
    b.source[field] = "wrong"
    with pytest.raises(ValueError):
        validate_hybrid_reference_bundle(b, source=source(), schedule=b.schedule)


@pytest.mark.parametrize("count", [0, 4])
def test_reference_count_is_ordered_one_to_three(count):
    b = bundle()
    b.request["references"] = list(range(count))
    with pytest.raises(ValueError):
        validate_hybrid_reference_bundle(b, source=source(), schedule=b.schedule)
