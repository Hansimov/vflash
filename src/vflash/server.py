"""HTTP boundary for the released native-core preview.

The service is deliberately CPU-only.  A resident spawned child owns the selected GPU
and loaded weights; the HTTP process never initializes CUDA.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from vflash import __version__
from vflash.catalog import ProfileCatalog
from vflash.contracts import ContractError
from vflash.hardware import NvidiaDevice, discover_nvidia_devices
from vflash.native.runner import WEIGHT_PROFILES
from vflash.native.worker import ResidentDenoiseWorker
from vflash.planner import resolve_plan

LOGGER = logging.getLogger(__name__)
DEFAULT_PROFILE_ID = "ref2va-turbo4-exact-sm89"
SUPPORTED_PROFILE_IDS = frozenset(WEIGHT_PROFILES)
CAPABILITIES = {
    "input": "compiled-conditioning-bundle",
    "output": "vae-ready-video-audio-latents",
    "output_download": True,
    "prompt_to_mp4": False,
}


@dataclass(frozen=True, slots=True)
class ServerSettings:
    """Operator-owned paths and the one container-visible GPU."""

    profile_id: str = DEFAULT_PROFILE_ID
    gpu_index: int = 0
    artifact_path: Path = Path("/runtime/artifact")
    schedule_overlay_path: Path = Path("/runtime/schedule")
    auxiliary_tensor_path: Path = Path("/runtime/auxiliary.safetensors")
    bundle_root: Path = Path("/runtime/bundles")
    output_root: Path = Path("/outputs")
    job_timeout_seconds: float = 1800.0
    max_pending_jobs: int = 8
    job_history_limit: int = 128

    def __post_init__(self) -> None:
        if self.gpu_index < 0:
            raise ContractError("GPU index must be non-negative")
        if not isfinite(self.job_timeout_seconds) or self.job_timeout_seconds <= 0:
            raise ContractError("job timeout must be finite and positive")
        if self.max_pending_jobs < 1 or self.job_history_limit < 1:
            raise ContractError("pending job capacity and history limit must be positive")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ServerSettings:
        values = os.environ if environ is None else environ
        return cls(
            profile_id=values.get("VFLASH_PROFILE_ID", DEFAULT_PROFILE_ID),
            gpu_index=_integer(values, "VFLASH_GPU_INDEX", 0),
            artifact_path=Path(values.get("VFLASH_ARTIFACT_PATH", "/runtime/artifact")),
            schedule_overlay_path=Path(
                values.get("VFLASH_SCHEDULE_OVERLAY_PATH", "/runtime/schedule")
            ),
            auxiliary_tensor_path=Path(
                values.get("VFLASH_AUXILIARY_TENSOR_PATH", "/runtime/auxiliary.safetensors")
            ),
            bundle_root=Path(values.get("VFLASH_BUNDLE_ROOT", "/runtime/bundles")),
            output_root=Path(values.get("VFLASH_OUTPUT_ROOT", "/outputs")),
            job_timeout_seconds=_positive_number(values, "VFLASH_JOB_TIMEOUT_SECONDS", 1800.0),
            max_pending_jobs=_integer(values, "VFLASH_MAX_PENDING_JOBS", 8),
            job_history_limit=_integer(values, "VFLASH_JOB_HISTORY_LIMIT", 128),
        )


class DenoiseJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bundle: str = Field(
        min_length=1,
        description="Conditioning-bundle directory relative to VFLASH_BUNDLE_ROOT",
    )


class DenoiseExecutor(Protocol):
    def __call__(self, bundle: Path, output: Path) -> dict[str, Any]: ...


class NativeDenoiseExecutor:
    """Lazily start one isolated GPU worker and reuse its fixed-profile session."""

    def __init__(
        self,
        settings: ServerSettings,
        device_provider: Callable[[], tuple[NvidiaDevice, ...]] = discover_nvidia_devices,
    ) -> None:
        self.settings = settings
        self.device_provider = device_provider
        self.worker: ResidentDenoiseWorker | None = None

    @property
    def available(self) -> bool:
        return self.worker is None or self.worker.available

    def __call__(self, bundle: Path, output: Path) -> dict[str, Any]:
        if self.worker is None:
            devices = {device.index: device for device in self.device_provider()}
            plan = resolve_plan(
                ProfileCatalog.bundled(),
                profile_id=self.settings.profile_id,
                device=devices[self.settings.gpu_index],
            )
            self.worker = ResidentDenoiseWorker(
                plan,
                timeout_seconds=self.settings.job_timeout_seconds,
                artifact=self.settings.artifact_path,
                schedule_overlay=self.settings.schedule_overlay_path,
                auxiliary_tensor=self.settings.auxiliary_tensor_path,
            )
        payload = self.worker.generate(bundle, output)
        generation = dict(payload["generation"])
        generation.pop("output_path", None)
        generation["output_file"] = output.name
        return {
            "profile_id": payload["profile_id"],
            "runtime": payload["runtime"],
            "session": payload["session"],
            "generation": generation,
        }

    def close(self) -> None:
        if self.worker is not None:
            self.worker.close()


class JobQueueFull(Exception):
    """The execution lane has no room for another request."""


class DenoiseJobManager:
    """A bounded, volatile execution lane; durable orchestration belongs to the caller."""

    def __init__(self, settings: ServerSettings, executor: DenoiseExecutor) -> None:
        self.settings = settings
        self.executor = executor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vflash-denoise")
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._finished: deque[str] = deque()
        self._pending = 0
        self._closed = False

    def submit(self, bundle: Path) -> dict[str, Any]:
        job_id = uuid4().hex
        output = self.settings.output_root / f"{job_id}.safetensors"
        record = {
            "id": job_id,
            "status": "queued",
            "profile_id": self.settings.profile_id,
            "bundle": str(bundle.relative_to(self.settings.bundle_root.resolve())),
            "output_file": output.name,
            "submitted_at": _now(),
            "started_at": None,
            "finished_at": None,
            "result": None,
            "error": None,
        }
        with self._lock:
            if self._closed:
                raise RuntimeError("execution lane is stopping")
            if self._pending >= self.settings.max_pending_jobs:
                raise JobQueueFull("execution queue is full; retry after a job finishes")
            self._jobs[job_id] = record
            self._pending += 1
            self._pool.submit(self._execute, job_id, bundle, output)
            # A fast executor may finish and expire this record before the caller polls.
            return dict(record)

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return dict(self._jobs[job_id])

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)
        close = getattr(self.executor, "close", None)
        if close is not None:
            close()

    def _execute(self, job_id: str, bundle: Path, output: Path) -> None:
        with self._lock:
            self._jobs[job_id].update(status="running", started_at=_now())
        try:
            result = self.executor(bundle, output)
        except Exception as exc:
            LOGGER.exception("native denoise job %s failed", job_id)
            changes = {"status": "failed", "error": str(exc)[:1000]}
        else:
            changes = {"status": "succeeded", "result": result}
        with self._lock:
            self._jobs[job_id].update(finished_at=_now(), **changes)
            self._pending -= 1
            self._finished.append(job_id)
            while len(self._finished) > self.settings.job_history_limit:
                del self._jobs[self._finished.popleft()]


def create_app(
    settings: ServerSettings | None = None,
    *,
    device_provider: Callable[[], tuple[NvidiaDevice, ...]] = discover_nvidia_devices,
    denoise_executor: DenoiseExecutor | None = None,
) -> FastAPI:
    resolved = settings or ServerSettings.from_env()
    catalog = ProfileCatalog.bundled()
    profile = catalog.profile(resolved.profile_id)
    if resolved.profile_id not in SUPPORTED_PROFILE_IDS or not profile.selectable:
        raise ContractError(f"profile is not exposed by the native API: {resolved.profile_id}")
    manager = DenoiseJobManager(
        resolved, denoise_executor or NativeDenoiseExecutor(resolved, device_provider)
    )

    def readiness_payload() -> dict[str, Any]:
        payload = _readiness(resolved, catalog, device_provider)
        worker_available = getattr(manager.executor, "available", True)
        payload["checks"]["worker"] = worker_available
        if not worker_available:
            payload["ready"] = False
            payload["error"] = "native worker stopped; restart the service"
        return payload

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.close()

    app = FastAPI(
        title="Vflash API",
        version=__version__,
        description=(
            "Ref2VA denoising on RTX 3080 20 GB and RTX 4090 48 GB. "
            "A compiled conditioning bundle is converted "
            "to VAE-ready latents. Prompt-to-MP4 generation is not exposed."
        ),
        lifespan=lifespan,
    )

    @app.get("/healthz")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "service": "vflash",
            "capabilities": CAPABILITIES,
        }

    @app.get("/readyz")
    def ready() -> JSONResponse:
        payload = readiness_payload()
        code = status.HTTP_200_OK if payload["ready"] else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=code, content=payload)

    @app.get("/v1/profiles")
    def profiles() -> dict[str, Any]:
        return {
            "profiles": [
                {
                    "id": profile.id,
                    "mode": profile.mode.value,
                    "availability": profile.availability.value,
                    "nfe": profile.nfe,
                    "precision": profile.precision,
                    "attention": profile.attention.backend,
                    "capabilities": CAPABILITIES,
                }
            ]
        }

    @app.post("/v1/denoise/jobs", status_code=status.HTTP_202_ACCEPTED)
    def submit_job(request: DenoiseJobRequest) -> dict[str, Any]:
        readiness = readiness_payload()
        if not readiness["ready"]:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"message": "Vflash is not ready", **readiness},
            )
        bundle = _resolve_bundle(resolved.bundle_root, request.bundle)
        try:
            return manager.submit(bundle)
        except JobQueueFull as exc:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=str(exc),
                headers={"Retry-After": "1"},
            ) from exc
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

    @app.get("/v1/denoise/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
            ) from exc

    @app.get("/v1/denoise/jobs/{job_id}/output")
    def download_job_output(job_id: str) -> FileResponse:
        try:
            job = manager.get(job_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="job not found"
            ) from exc
        if job["status"] != "succeeded":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="job output is not ready",
            )
        output = resolved.output_root.resolve() / job["output_file"]
        if not output.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="job output is unavailable",
            )
        return FileResponse(
            path=output,
            filename=job["output_file"],
            media_type="application/octet-stream",
        )

    return app


def main() -> None:
    import uvicorn

    host = os.environ.get("VFLASH_API_HOST", "0.0.0.0")
    port = _integer(os.environ, "VFLASH_API_PORT", 8000)
    uvicorn.run(create_app(), host=host, port=port, workers=1)


def _readiness(
    settings: ServerSettings,
    catalog: ProfileCatalog,
    device_provider: Callable[[], tuple[NvidiaDevice, ...]],
) -> dict[str, Any]:
    checks = {
        "artifact": (settings.artifact_path / "artifact.json").is_file(),
        "schedule": all(
            (settings.schedule_overlay_path / name).is_file()
            for name in ("overlay.json", "schedule.safetensors")
        ),
        "auxiliary_tensor": settings.auxiliary_tensor_path.is_file(),
        "bundle_root": settings.bundle_root.is_dir(),
        "output_root": settings.output_root.is_dir()
        and os.access(settings.output_root, os.W_OK),
        "gpu": False,
    }
    gpu: dict[str, Any] | None = None
    error: str | None = None
    try:
        devices = {device.index: device for device in device_provider()}
        device = devices[settings.gpu_index]
        plan = resolve_plan(catalog, profile_id=settings.profile_id, device=device)
        checks["gpu"] = True
        gpu = {
            "index": device.index,
            "name": device.name,
            "memory_gib": device.memory_gib,
            "compute_capability": device.compute_capability,
            "target_id": plan.target.id,
        }
    except (ContractError, KeyError) as exc:
        error = str(exc) or f"GPU index {settings.gpu_index} was not found"
    return {
        "ready": all(checks.values()),
        "profile_id": settings.profile_id,
        "capabilities": CAPABILITIES,
        "checks": checks,
        "gpu": gpu,
        "error": error,
    }


def _resolve_bundle(root: Path, value: str) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        candidate = (resolved_root / value).resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="bundle not found"
        ) from exc
    if not candidate.is_dir() or not candidate.is_relative_to(resolved_root):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="bundle not found")
    if not all(
        (candidate / name).is_file() for name in ("bundle.json", "conditioning.safetensors")
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="bundle must contain bundle.json and conditioning.safetensors",
        )
    return candidate


def _integer(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    try:
        return default if raw is None else int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc


def _positive_number(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be numeric") from exc
    if not isfinite(value) or value <= 0:
        raise RuntimeError(f"{name} must be positive")
    return value


def _now() -> str:
    return datetime.now(UTC).isoformat()


if __name__ == "__main__":
    main()
