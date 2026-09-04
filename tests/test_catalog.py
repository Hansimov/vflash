import json
from importlib.resources import files

import pytest

from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.native.runner import WEIGHT_PROFILES


def test_every_shipped_profile_has_a_native_executor() -> None:
    catalog = ProfileCatalog.bundled()
    assert {profile.id for profile in catalog.profiles} == set(WEIGHT_PROFILES)
    assert all(profile.selectable for profile in catalog.profiles)
    assert {target.compute_capability for target in catalog.targets} == {"8.6", "8.9"}


@pytest.mark.parametrize("fault", ["unknown-target", "duplicate-id", "nonfinite-memory"])
def test_rejects_catalog_that_cannot_make_a_valid_plan(fault: str) -> None:
    payload = json.loads(files("vflash").joinpath("data/h3-profiles.json").read_text())
    if fault == "unknown-target":
        payload["profiles"][0]["target_ids"] = ["missing"]
    elif fault == "duplicate-id":
        payload["profiles"].append(payload["profiles"][0])
    else:
        payload["targets"][0]["minimum_memory_gib"] = float("nan")
    with pytest.raises(ContractError):
        ProfileCatalog.from_dict(payload)
