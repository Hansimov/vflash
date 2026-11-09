"""Actual local HTTP and MP4s exercise ownership, never claim model quality."""

import json
import shutil
import socket
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from uuid import UUID

import pytest
from PIL import Image

from vflash.cancellation import GenerationCancelled
from vflash.pipeline.remake import RemakeReference, RemakeRequest
from vflash.pipeline.remake_execution import (
    LocalRemakeRuntime,
    RemakeExecutionError,
    execute_remake,
)


@pytest.fixture
def media(tmp_path):
    inputs, outputs = tmp_path / "input", tmp_path / "output"
    inputs.mkdir()
    outputs.mkdir()
    source = inputs / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "color=blue:size=480x480:rate=24",
            "-frames:v",
            "90",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
        capture_output=True,
        timeout=15,
    )
    picture = inputs / "picture.png"
    Image.new("RGB", (480, 480), "green").save(picture)
    request = RemakeRequest(
        source,
        "Apply <Picture 1> to <Video 1>.",
        "edit",
        90,
        480,
        480,
        (RemakeReference(picture, 0),),
        "silent",
    )
    return request, inputs, outputs


@contextmanager
def model_server(media, *, mode="complete"):
    request, _, outputs = media
    state = {"posts": [], "history": {}, "stops": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/queue":
                self.reply({"queue_running": [], "queue_pending": []})
            else:
                ident = self.path.removeprefix("/history/")
                self.reply(
                    {ident: state["history"][ident]} if ident in state["history"] else {}
                )

        def do_POST(self):
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert (
                self.path == "/prompt" and str(UUID(value["prompt_id"])) == value["prompt_id"]
            )
            state["posts"].append(value)
            ident = value["prompt_id"]
            if mode == "rejected":
                self.reply({"error": {"type": "prompt_outputs_failed_validation"}}, 400)
                return
            prefix = Path(value["prompt"]["20"]["inputs"]["filename_prefix"]).parent.as_posix()
            directory = outputs / prefix
            directory.mkdir()
            result = directory / "video_00001_.mp4"
            shutil.copyfile(request.source, result)
            entry = {
                "status": {"status_str": "success"},
                "outputs": {
                    "7": {
                        "images": [{"type": "input", "filename": "source.mp4", "subfolder": ""}]
                    },
                    "20": {
                        "images": [
                            {
                                "type": "output",
                                "filename": result.name,
                                "subfolder": prefix if mode != "foreign" else "other-owner",
                            }
                        ]
                    },
                },
                "prompt": [0, ident, value["prompt"], {}, ["20"]],
            }
            if mode == "wrong_graph":
                entry["prompt"][2]["12"]["inputs"]["seed"] += 1
            if mode != "missing":
                state["history"][ident] = entry
            if mode == "lost_reply":
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
            else:
                self.reply({"prompt_id": ident})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.02), daemon=True
    )
    thread.start()

    def stop():
        state["stops"] += 1
        return True

    try:
        yield LocalRemakeRuntime(f"http://127.0.0.1:{server.server_port}"), stop, state
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


@pytest.mark.parametrize("mode", ["complete", "lost_reply"])
def test_single_submission_recovers_lost_reply_and_only_publishes_real_output(
    tmp_path,
    media,
    mode,
):
    request, inputs, outputs = media
    with model_server(media, mode=mode) as (runtime, stop, server):
        result = execute_remake(
            request,
            input_directory=inputs,
            output_directory=outputs,
            execution_directory=tmp_path / "execution",
            runtime=runtime,
            stop_runtime=stop,
            timeout_seconds=5,
        )
        assert result["media"]["frames"] == 90 and not result["media"]["has_audio"]
        assert Path(result["artifact"]).read_bytes() == request.source.read_bytes()
        assert len(server["posts"]) == 1 and server["stops"] == 0
        journal = json.loads((tmp_path / "execution/execution.json").read_text())
        assert journal["status"] == "completed" and journal["submission_count"] == 1
        assert journal["submission_response"] == (
            "lost_or_invalid" if mode == "lost_reply" else "accepted"
        )
        with pytest.raises(ValueError, match="fresh"):
            execute_remake(
                request,
                input_directory=inputs,
                output_directory=outputs,
                execution_directory=tmp_path / "execution",
                runtime=runtime,
                stop_runtime=stop,
            )
        assert len(server["posts"]) == 1


@pytest.mark.parametrize("mode", ["rejected", "foreign", "missing", "cancelled", "wrong_graph"])
def test_failed_or_interrupted_original_execution_has_explicit_runtime_ownership(
    tmp_path,
    media,
    mode,
):
    request, inputs, outputs = media
    with model_server(media, mode=mode) as (runtime, stop, server):
        expected = (
            TimeoutError
            if mode == "missing"
            else (GenerationCancelled if mode == "cancelled" else RemakeExecutionError)
        )
        with pytest.raises(expected):
            execute_remake(
                request,
                input_directory=inputs,
                output_directory=outputs,
                execution_directory=tmp_path / "execution",
                runtime=runtime,
                stop_runtime=stop,
                timeout_seconds=1,
                cancel_check=lambda: mode == "cancelled" and bool(server["posts"]),
            )
        journal = json.loads((tmp_path / "execution/execution.json").read_text())
        assert len(server["posts"]) == 1
        assert journal["status"] == ("rejected" if mode == "rejected" else "interrupted")
        assert server["stops"] == (0 if mode == "rejected" else 1)
        if mode != "rejected":
            assert journal["runtime_stop"] == "confirmed"


def test_runtime_rejects_foreign_endpoint_and_concurrent_lane(tmp_path, media):
    for endpoint in [
        "http://example.com:8188",
        "http://127.0.0.1:8188/other",
        "http://127.0.0.1:0",
    ]:
        with pytest.raises(ValueError):
            LocalRemakeRuntime(endpoint)
    request, inputs, outputs = media
    with model_server(media) as (runtime, stop, server), runtime.lane:
        with pytest.raises(ValueError, match="available"):
            execute_remake(
                request,
                input_directory=inputs,
                output_directory=outputs,
                execution_directory=tmp_path / "execution",
                runtime=runtime,
                stop_runtime=stop,
            )
        assert not server["posts"] and not (tmp_path / "execution").exists()


def test_unconfirmed_runtime_stop_is_recorded_without_hiding_original_timeout(tmp_path, media):
    request, inputs, outputs = media
    with model_server(media, mode="missing") as (runtime, _, server):
        with pytest.raises(TimeoutError):
            execute_remake(
                request,
                input_directory=inputs,
                output_directory=outputs,
                execution_directory=tmp_path / "execution",
                runtime=runtime,
                stop_runtime=lambda: False,
                timeout_seconds=0.5,
            )
        journal = json.loads((tmp_path / "execution/execution.json").read_text())
        assert journal["runtime_stop"] == "needs_reconciliation"
        assert journal["error_type"] == "TimeoutError" and len(server["posts"]) == 1
