import time
from dataclasses import replace
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient

from vflash.hardware import NvidiaDevice
from vflash.server import NativeDenoiseExecutor, ServerSettings, create_app


def _settings(tmp_path: Path) -> ServerSettings:
    artifact = tmp_path / "artifact"
    schedule = tmp_path / "schedule"
    bundles = tmp_path / "bundles"
    output = tmp_path / "outputs"
    bundle = bundles / "example"
    artifact.mkdir()
    schedule.mkdir()
    bundle.mkdir(parents=True)
    output.mkdir()
    (artifact / "artifact.json").write_text("{}", encoding="utf-8")
    (schedule / "overlay.json").write_text("{}", encoding="utf-8")
    (schedule / "schedule.safetensors").write_bytes(b"schedule")
    auxiliary = tmp_path / "auxiliary.safetensors"
    auxiliary.write_bytes(b"auxiliary")
    (bundle / "bundle.json").write_text("{}", encoding="utf-8")
    (bundle / "conditioning.safetensors").write_bytes(b"conditioning")
    return ServerSettings(
        gpu_index=0,
        artifact_path=artifact,
        schedule_overlay_path=schedule,
        auxiliary_tensor_path=auxiliary,
        bundle_root=bundles,
        output_root=output,
        job_timeout_seconds=60,
    )


def _sm89_devices() -> tuple[NvidiaDevice, ...]:
    return (NvidiaDevice(0, "test-uuid", "RTX 4090", 48.0, "8.9", 450.0),)


def test_health_readiness_and_profiles_are_honest(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings, device_provider=_sm89_devices)
    with TestClient(app) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
        profiles = client.get("/v1/profiles")
        settings.auxiliary_tensor_path.unlink()
        not_ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json()["capabilities"] == {
        "input": "compiled-conditioning-bundle",
        "output": "vae-ready-video-audio-latents",
        "output_download": True,
        "prompt_to_mp4": False,
    }
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert ready.json()["gpu"]["target_id"] == "sm89-48g-resident"
    assert "uuid" not in ready.text
    assert [item["id"] for item in profiles.json()["profiles"]] == ["ref2va-turbo4-exact-sm89"]
    assert not_ready.status_code == 503
    assert not_ready.json()["checks"]["auxiliary_tensor"] is False


def test_denoise_job_uses_relative_bundle_and_server_owned_output(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def execute(bundle: Path, output: Path) -> dict[str, object]:
        assert bundle == settings.bundle_root / "example"
        output.write_bytes(b"latents")
        return {"generation": {"output_file": output.name}}

    app = create_app(
        settings,
        device_provider=_sm89_devices,
        denoise_executor=execute,
    )
    with TestClient(app) as client:
        accepted = client.post("/v1/denoise/jobs", json={"bundle": "example"})
        assert accepted.status_code == 202
        job_id = accepted.json()["id"]
        for _ in range(100):
            job = client.get(f"/v1/denoise/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        downloaded = client.get(f"/v1/denoise/jobs/{job_id}/output")

    assert job["status"] == "succeeded"
    assert job["bundle"] == "example"
    assert job["output_file"] == f"{job_id}.safetensors"
    assert (settings.output_root / job["output_file"]).read_bytes() == b"latents"
    assert downloaded.status_code == 200
    assert downloaded.content == b"latents"


def test_api_rejects_unmounted_or_expanded_input_surface(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings, device_provider=_sm89_devices, denoise_executor=lambda *_: {})
    with TestClient(app) as client:
        traversal = client.post("/v1/denoise/jobs", json={"bundle": "../artifact"})
        prompt_request = client.post(
            "/v1/denoise/jobs",
            json={"bundle": "example", "prompt": "not a supported API input"},
        )

    assert traversal.status_code == 404
    assert prompt_request.status_code == 422


def test_stopped_worker_is_not_ready_to_accept_more_work(tmp_path: Path) -> None:
    class StoppedExecutor:
        available = False

        def __call__(self, *_args):
            raise AssertionError("a stopped worker must not receive a job")

    app = create_app(
        _settings(tmp_path), device_provider=_sm89_devices, denoise_executor=StoppedExecutor()
    )
    with TestClient(app) as client:
        ready = client.get("/readyz")
        submitted = client.post("/v1/denoise/jobs", json={"bundle": "example"})
    assert ready.status_code == 503
    assert ready.json()["checks"]["worker"] is False
    assert submitted.status_code == 503


def test_native_executor_reuses_one_worker_and_closes_it(tmp_path: Path, monkeypatch) -> None:
    settings = _settings(tmp_path)
    workers = []

    class Worker:
        def __init__(self, plan, **_options):
            self.plan = plan
            self.calls = 0
            self.closed = False
            workers.append(self)

        def generate(self, bundle, output):
            self.calls += 1
            return {
                "profile_id": self.plan.profile.id,
                "runtime": {"backend": "native"},
                "session": {"request_index": self.calls},
                "generation": {"output_path": str(output), "nfe": 4},
            }

        def close(self):
            self.closed = True

    monkeypatch.setattr("vflash.server.ResidentDenoiseWorker", Worker)
    executor = NativeDenoiseExecutor(settings, _sm89_devices)
    first = executor(
        settings.bundle_root / "example", settings.output_root / "first.safetensors"
    )
    second = executor(
        settings.bundle_root / "example", settings.output_root / "second.safetensors"
    )
    executor.close()

    assert len(workers) == 1
    assert workers[0].plan.gpu_uuid == "test-uuid"
    assert workers[0].closed
    assert first["session"]["request_index"] == 1
    assert second["session"]["request_index"] == 2
    assert second["generation"] == {"nfe": 4, "output_file": "second.safetensors"}


def test_capacity_fifo_and_history_do_not_discard_output_files(tmp_path: Path) -> None:
    settings = replace(_settings(tmp_path), max_pending_jobs=2, job_history_limit=1)
    release = Event()
    entered = Event()
    order = []

    def execute(bundle: Path, output: Path) -> dict[str, object]:
        order.append(output.stem)
        entered.set()
        assert release.wait(5)
        output.write_bytes(b"latents")
        return {"generation": {"output_file": output.name}}

    app = create_app(settings, device_provider=_sm89_devices, denoise_executor=execute)
    with TestClient(app) as client:
        try:
            first = client.post("/v1/denoise/jobs", json={"bundle": "example"}).json()
            assert entered.wait(2)
            second = client.post("/v1/denoise/jobs", json={"bundle": "example"}).json()
            rejected = client.post("/v1/denoise/jobs", json={"bundle": "example"})
            assert rejected.status_code == 429
            assert rejected.headers["Retry-After"] == "1"
        finally:
            release.set()
        for _ in range(100):
            latest = client.get(f"/v1/denoise/jobs/{second['id']}").json()
            if latest["status"] == "succeeded":
                break
            time.sleep(0.01)
        assert latest["status"] == "succeeded"
        assert order == [first["id"], second["id"]]
        assert client.get(f"/v1/denoise/jobs/{first['id']}").status_code == 404
        assert (settings.output_root / first["output_file"]).read_bytes() == b"latents"
        assert client.post("/v1/denoise/jobs", json={"bundle": "example"}).status_code == 202
