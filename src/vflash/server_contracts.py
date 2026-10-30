"""Transport schemas for the native latent API; no model or GPU initialization."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Response(BaseModel):
    # Preserve additive runtime metadata returned by current/future executors.
    model_config = ConfigDict(extra="allow")


class Capabilities(Response):
    input: Literal["compiled-conditioning-bundle"]
    output: Literal["vae-ready-video-audio-latents"]
    output_download: bool
    prompt_to_mp4: bool


class Health(Response):
    status: Literal["ok"]
    service: Literal["vflash"]
    capabilities: Capabilities


class Readiness(Response):
    ready: bool
    version: str
    profile_id: str
    capabilities: Capabilities
    checks: dict[str, bool]
    gpu: dict[str, Any] | None
    parallel: dict[str, Any] | None
    attention_selection: dict[str, Any] | None
    error: str | None


class Profile(Response):
    id: str
    mode: str
    availability: str
    nfe: int
    precision: str
    baseline_attention: str
    requested_attention_backend: str
    capabilities: Capabilities


class Profiles(Response):
    profiles: list[Profile]


class Job(Response):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    profile_id: str
    bundle: str = Field(description="Directory relative to the configured bundle root.")
    output_file: str = Field(
        description="Server-owned filename; download through the job output route."
    )
    submitted_at: str = Field(json_schema_extra={"format": "date-time"})
    started_at: str | None = Field(json_schema_extra={"format": "date-time"})
    finished_at: str | None = Field(json_schema_extra={"format": "date-time"})
    result: dict[str, Any] | None = Field(
        description="Executor diagnostics on success; additive and profile-specific."
    )
    error: str | None


class Error(Response):
    detail: str | dict[str, Any] | list[dict[str, Any]]
