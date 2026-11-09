"""One owned Remake execution against an already provisioned local runtime.

The runtime owner supplies its stop operation. This client never provisions a GPU,
downloads a model, retries a submission or accepts an arbitrary graph. The original
request and its outcome survive a lost HTTP response in a private execution journal.
"""

import json
import math
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import UUID, uuid4

from vflash.cancellation import GenerationCancelled
from vflash.media.probe import probe_normalized_video
from vflash.pipeline.remake import COMFY_REVISION, SOURCE_REVISION, RemakeRequest, remake_graph


class RemakeExecutionError(RuntimeError):
    """The original journal owns the execution outcome and any completed media."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RemakeExecutionError("The owned model endpoint must not redirect")


class LocalRemakeRuntime:
    """Transport for one serial lane on an explicitly owned loopback server."""

    def __init__(self, endpoint: str):
        parsed = urllib.parse.urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or not 1 <= parsed.port <= 65535
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Remake requires the owned HTTP loopback endpoint")
        self.endpoint = endpoint.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            _NoRedirect(),
        )
        self.lane = threading.Lock()

    def request(self, route: str, *, timeout: float, body: dict | None = None):
        request = urllib.request.Request(
            self.endpoint + route,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with self.opener.open(request, timeout=timeout) as response:
            data = response.read(2 * 1024**2 + 1)
        if len(data) > 2 * 1024**2:
            raise RemakeExecutionError("Model control response exceeds its bounded contract")
        result = json.loads(data)
        if not isinstance(result, dict):
            raise RemakeExecutionError("Model control response must be an object")
        return result


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact(entry, output_directory, prefix, request, cancel_check):
    if entry.get("status", {}).get("status_str") != "success":
        raise RemakeExecutionError("Original model execution did not complete successfully")
    # LoadVideo also returns an MP4 in history. Only the fixed SaveVideo output
    # belongs to this result; never mistake the input preview for a generated clip.
    outputs = entry.get("outputs", {}).get("20", {}).get("images", [])
    if len(outputs) != 1 or not isinstance(outputs[0], dict):
        raise RemakeExecutionError("The fixed graph must return exactly one saved video")
    item = outputs[0]
    name, folder = item.get("filename"), item.get("subfolder")
    if (
        item.get("type") != "output"
        or not isinstance(name, str)
        or Path(name).name != name
        or not name.endswith(".mp4")
        or folder != prefix
    ):
        raise RemakeExecutionError("The result must belong to its original output namespace")
    path = output_directory / folder / name
    if (
        path.is_symlink()
        or not path.is_file()
        or path.parent.is_symlink()
        or not path.resolve().is_relative_to(output_directory)
    ):
        raise RemakeExecutionError("The runtime returned an unowned result path")
    video, audio = probe_normalized_video(path, cancel_check)
    if (video["width"], video["height"], int(video["nb_frames"])) != (
        request.width,
        request.height,
        request.frames,
    ) or (request.soundtrack == "silent" and audio):
        raise RemakeExecutionError("The saved result differs from its frozen media contract")
    return path, {
        "frames": request.frames,
        "width": request.width,
        "height": request.height,
        "fps": 24,
        "has_audio": bool(audio),
        "bytes": path.stat().st_size,
    }


def _identity(path):
    if not isinstance(path, Path) or path.is_symlink() or not path.is_file():
        raise RemakeExecutionError("Recovery requires the original owned regular files")
    info = path.stat()
    return {"path": str(path.resolve()), "bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def _input_identities(request):
    paths = [request.source, *(ref.path for ref in request.references)]
    if request.prefix:
        paths.append(request.prefix.take)
    return [_identity(path) for path in paths]


def _read_receipt(path):
    _identity(path)
    if path.stat().st_size > 2 * 1024**2:
        raise RemakeExecutionError("The execution receipt exceeds its bounded contract")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise RemakeExecutionError("The execution receipt must be an object")
    return result


def recover_completed_remake(
    request: RemakeRequest,
    *,
    input_directory: Path,
    output_directory: Path,
    execution_directory: Path,
    cancel_check=None,
) -> dict:
    """Read a completed original execution after a caller restart, without any HTTP.

    Only a durable completion with its exact graph, history and unchanged files is
    recoverable here. Interrupted/unknown executions still require their original
    owner's reconciliation. This operation never submits, stops or rents anything.
    Receipts written before input/output identities were recorded are not adopted.
    """
    for directory in (execution_directory, output_directory):
        if not isinstance(directory, Path) or directory.is_symlink() or not directory.is_dir():
            raise ValueError(
                "Recovery needs the original owned execution and output directories"
            )
    state = _read_receipt(execution_directory / "execution.json")
    ident = state.get("prompt_id")
    try:
        valid_id = isinstance(ident, str) and str(UUID(ident)) == ident
    except ValueError:
        valid_id = False
    if (
        state.get("status") != "completed"
        or not valid_id
        or state.get("submission_count") != 1
        or state.get("source_revision") != SOURCE_REVISION
        or state.get("comfy_revision") != COMFY_REVISION
        or state.get("output_prefix") != "remake-" + ident
        or type(state.get("seconds")) not in {int, float}
        or not math.isfinite(state["seconds"])
        or state["seconds"] < 0
    ):
        raise RemakeExecutionError(
            "Reconcile the original execution: no valid completion receipt"
        )
    graph = remake_graph(
        request,
        input_directory=input_directory,
        output_prefix=state["output_prefix"] + "/video",
        cancel_check=cancel_check,
    )
    if graph != state.get("graph") or _input_identities(request) != state.get("inputs"):
        raise RemakeExecutionError(
            "Recovery must retain the original graph and input identities"
        )
    entry = _read_receipt(execution_directory / "history.json")
    executed = entry.get("prompt")
    if (
        not isinstance(executed, list)
        or len(executed) < 3
        or executed[1] != ident
        or executed[2] != graph
    ):
        raise RemakeExecutionError("Executed graph differs from the frozen request")
    path, media = _artifact(
        entry,
        output_directory.resolve(strict=True),
        state["output_prefix"],
        request,
        cancel_check,
    )
    if (
        str(path) != state.get("artifact")
        or media != state.get("media")
        or _identity(path) != state.get("artifact_identity")
    ):
        raise RemakeExecutionError("The completed artifact changed after publication")
    return {key: state[key] for key in ("prompt_id", "artifact", "media", "seconds")}


def execute_remake(
    request: RemakeRequest,
    *,
    input_directory: Path,
    output_directory: Path,
    execution_directory: Path,
    runtime: LocalRemakeRuntime,
    stop_runtime,
    timeout_seconds: float = 900,
    cancel_check=None,
) -> dict:
    """Submit once, observe that original ID, and return its verified raw artifact.

    A distinct execution directory is mandatory. Reusing an old directory fails
    before any network operation: recovery must inspect the existing identity,
    never implicitly resubmit. A lost POST reply continues observing the SAME ID.
    Cancellation, timeout or uncertain execution stops the owner's runtime; the
    provisioning/claim controller retains responsibility for returning its GPU.
    Source-audio assembly and removal of history/padding belong to the caller.
    """
    if not isinstance(runtime, LocalRemakeRuntime) or not runtime.lane.acquire(blocking=False):
        raise ValueError("Select one available owned runtime lane")
    try:
        return _execute_remake(
            request,
            input_directory=input_directory,
            output_directory=output_directory,
            execution_directory=execution_directory,
            runtime=runtime,
            stop_runtime=stop_runtime,
            timeout_seconds=timeout_seconds,
            cancel_check=cancel_check,
        )
    finally:
        runtime.lane.release()


def _execute_remake(
    request,
    *,
    input_directory,
    output_directory,
    execution_directory,
    runtime,
    stop_runtime,
    timeout_seconds,
    cancel_check,
):
    if (
        type(timeout_seconds) not in {int, float}
        or not 0 < timeout_seconds <= 1800
        or not callable(stop_runtime)
        or not isinstance(runtime, LocalRemakeRuntime)
        or not isinstance(execution_directory, Path)
        or execution_directory.exists()
        or execution_directory.is_symlink()
        or not isinstance(output_directory, Path)
        or output_directory.is_symlink()
        or not output_directory.is_dir()
    ):
        raise ValueError("A fresh private execution and finite owned runtime are required")
    output_directory = output_directory.resolve(strict=True)
    started = time.monotonic()
    deadline = started + timeout_seconds

    def remaining():
        if cancel_check and cancel_check():
            raise GenerationCancelled("Owned Remake execution was cancelled")
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Owned Remake execution exceeded its deadline")
        return value

    def cancelled():
        remaining()
        return False

    prompt_id = str(uuid4())
    prefix = "remake-" + prompt_id
    graph = remake_graph(
        request,
        input_directory=input_directory,
        output_prefix=prefix + "/video",
        cancel_check=cancelled,
    )
    execution_directory.mkdir(mode=0o700)
    parent = os.open(execution_directory.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    journal = execution_directory / "execution.json"
    state = {
        "prompt_id": prompt_id,
        "status": "prepared",
        "started_at": time.time(),
        "source_revision": SOURCE_REVISION,
        "comfy_revision": COMFY_REVISION,
        "graph": graph,
        "output_prefix": prefix,
        "submission_count": 0,
        "inputs": _input_identities(request),
    }
    _save(journal, state)
    submitted, completed, rejected = False, False, False

    try:
        queue = runtime.request("/queue", timeout=min(5, remaining()))
        if any(
            not isinstance(queue.get(key), list) for key in ("queue_running", "queue_pending")
        ):
            raise RemakeExecutionError("The runtime queue response is incomplete")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RemakeExecutionError("The owned runtime lane is occupied")
        state.update(status="submission_unknown", submission_count=1, submitted_at=time.time())
        _save(journal, state)
        submitted = True
        try:
            accepted = runtime.request(
                "/prompt",
                timeout=min(30, remaining()),
                body={
                    "prompt_id": prompt_id,
                    "client_id": prompt_id,
                    "prompt": graph,
                },
            )
        except urllib.error.HTTPError as error:
            # Validation is known to precede queue admission. A server error is
            # uncertain and must still be reconciled through this original ID.
            if error.code in {400, 422}:
                rejected = True
                raise RemakeExecutionError(
                    "The original graph was rejected before admission"
                ) from error
            state["submission_response"] = "server_error_unknown"
        except (OSError, ValueError):
            state["submission_response"] = "lost_or_invalid"
        else:
            if accepted.get("prompt_id") != prompt_id:
                raise RemakeExecutionError("The model response changed the original request ID")
            state.update(status="observing", submission_response="accepted")
        _save(journal, state)
        while True:
            duration = remaining()
            try:
                history = runtime.request("/history/" + prompt_id, timeout=min(5, duration))
            except (OSError, ValueError):
                history = {}
            if prompt_id in history:
                entry = history[prompt_id]
                _save(execution_directory / "history.json", entry)
                executed = entry.get("prompt")
                if (
                    not isinstance(executed, list)
                    or len(executed) < 3
                    or executed[1] != prompt_id
                    or executed[2] != graph
                ):
                    raise RemakeExecutionError("Executed graph differs from the frozen request")
                path, media = _artifact(entry, output_directory, prefix, request, cancelled)
                state.update(
                    status="completed",
                    completed_at=time.time(),
                    seconds=time.monotonic() - started,
                    artifact=str(path),
                    artifact_identity=_identity(path),
                    media=media,
                )
                _save(journal, state)
                completed = True
                return {
                    key: state[key] for key in ("prompt_id", "artifact", "media", "seconds")
                }
            time.sleep(min(0.2, remaining()))
    except BaseException as error:
        state.update(
            status="rejected" if rejected else "interrupted",
            error_type=type(error).__name__,
            elapsed_seconds=time.monotonic() - started,
        )
        _save(journal, state)
        raise
    finally:
        if submitted and not completed and not rejected:
            # Preserve the primary failure, but retain an explicit missing-stop
            # receipt so the lease owner can finish resource reconciliation.
            try:
                if stop_runtime() is not True:
                    raise RuntimeError("The owner has not confirmed the runtime stopped")
                state["runtime_stop"] = "confirmed"
            except BaseException as error:
                state["runtime_stop"] = "needs_reconciliation"
                state["stop_error_type"] = type(error).__name__
            _save(journal, state)
