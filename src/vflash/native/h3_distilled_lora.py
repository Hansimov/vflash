"""Pinned Ref2VA adapter provenance and runtime low-rank scaling."""

from __future__ import annotations

from dataclasses import dataclass


class H3DistilledLoraError(ValueError):
    """A distilled H3 LoRA differs from its pinned release contract."""


@dataclass(frozen=True)
class H3DistilledLoraContract:
    profile_id: str
    revision: str
    filename: str
    size_bytes: int
    sha256: str
    nfe: int
    repository: str = "lightx2v/Minimax-h3-Turbo"
    rank: int = 128
    alpha: float = 8.0
    strength: float = 1.0

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank * self.strength


LIGHTX_H3_REF_TURBO8_CONTRACT = H3DistilledLoraContract(
    profile_id="lightx-turbo8-v1.0",
    revision="0eebcc7e79f9cb200927c80b8e7595265b770e34",
    filename="minimax_h3_ref2v_turbo_8step_v1.0_768p_bf16.safetensors",
    size_bytes=1_383_677_808,
    sha256="9bac880b1a5d7ac052171cf6cce769f0cceaaa42ffa51de4b8e41143a2bdd2d2",
    nfe=8,
)
LIGHTX_H3_REF_TURBO4_CONTRACT = H3DistilledLoraContract(
    profile_id="lightx-ref-turbo4-v0.1",
    revision="83b617309219e859c1c264520eba07492d22e958",
    filename="minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    size_bytes=1_383_677_768,
    sha256="9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99",
    nfe=4,
)


def h3_distilled_lora_contract_for_profile(
    profile_id: str, *, workflow: str
) -> H3DistilledLoraContract:
    """Resolve the immutable Ref2VA release recorded by a runtime artifact."""
    if workflow == "ref2va":
        for contract in (LIGHTX_H3_REF_TURBO4_CONTRACT, LIGHTX_H3_REF_TURBO8_CONTRACT):
            if profile_id == contract.profile_id:
                return contract
    raise H3DistilledLoraError(f"unknown H3 distilled LoRA profile: {profile_id}/{workflow}")
