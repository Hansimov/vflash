"""CPU-side ownership of one serial, resident CUDA worker.

Only paths, the execution contract and small metrics cross the pipe. CUDA tensors
and model resources belong exclusively to the spawned process.
"""

from __future__ import annotations

import multiprocessing
from contextlib import suppress
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from vflash.contracts import ExecutionPlan


def _serve_session(connection: Connection, plan: ExecutionPlan, paths: dict[str, Path]) -> None:
    from vflash.native.runner import NativeEngineSession

    session = None
    try:
        session = NativeEngineSession(plan, **paths)
        while (request := connection.recv()) is not None:
            bundle, output = request
            connection.send({"result": session.generate(bundle, output)})
    except EOFError:
        pass
    except Exception as exc:
        # A failed trajectory may leave mutated buffers or a poisoned CUDA context.
        connection.send({"error": f"{type(exc).__name__}: {exc}"})
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            connection.close()


class ResidentDenoiseWorker:
    """One execution lane; calls must be serialized by its owner."""

    def __init__(self, plan: ExecutionPlan, *, timeout_seconds: float, **paths: Path) -> None:
        context = multiprocessing.get_context("spawn")
        self._connection, child = context.Pipe()
        self._process = context.Process(
            target=_serve_session, args=(child, plan, paths), name="vflash-gpu", daemon=True
        )
        self.timeout_seconds = timeout_seconds
        self._closed = False
        self._process.start()
        child.close()

    @property
    def available(self) -> bool:
        return not self._closed and self._process.is_alive()

    def generate(self, bundle: Path, output: Path) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError(
                "native worker is closed; restart the service before submitting jobs"
            )
        try:
            self._connection.send((bundle, output))
            if not self._connection.poll(self.timeout_seconds):
                raise TimeoutError(f"native worker exceeded {self.timeout_seconds:g} seconds")
            message = self._connection.recv()
            if "error" in message:
                raise RuntimeError(message["error"])
            return message["result"]
        except (EOFError, BrokenPipeError) as exc:
            self.close(terminate=True)
            raise RuntimeError("native worker exited before returning a result") from exc
        except Exception:
            self.close(terminate=True)
            raise

    def close(self, *, terminate: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.is_alive():
            if terminate:
                self._process.terminate()
            else:
                with suppress(BrokenPipeError):
                    self._connection.send(None)
        self._connection.close()
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()
            self._process.join()
        self._process.close()
