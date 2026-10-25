"""Request-owned video decoding alongside an independent audio refinement pass."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor


class AudioVideoOverlap:
    """Join every decoder future before the owning request can release CUDA."""

    def __init__(self, decoder, *, width, height):
        self.decoder = decoder
        self.width, self.height = width, height
        self.future = self.executor = None
        self.called = False
        self.record = {"applied": False, "method": "audio-refinement-video-decode-streams"}

    def __enter__(self):
        return self

    def submit(self, video_latents):
        if self.called:
            raise ValueError("the completed video can be submitted exactly once")
        self.called = True
        torch = self.decoder._torch
        device = self.decoder.device
        free, _ = torch.cuda.mem_get_info(device)
        reusable = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        required = 8 * 1024**3 + self.width * self.height * 16384
        if free + reusable < required:
            self.record.update(reason="device_headroom", required_bytes=required)
            return
        self.record["submitted_at"] = time.monotonic()
        self.record["decoder_resume_seconds"] = self.decoder.resume_cuda(include_audio=False)
        stream = torch.cuda.Stream(device=device)
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="vflash-video-decode"
        )

        def decode():
            with torch.inference_mode(), torch.cuda.device(device), torch.cuda.stream(stream):
                started = time.monotonic()
                self.record["decode_started_at"] = started
                result = self.decoder.decode_video(
                    video_latents, height=self.height, width=self.width
                )
                stream.synchronize()
                self.record.update(
                    decode_seconds=time.monotonic() - started,
                    decode_finished_at=time.monotonic(),
                )
                return result

        self.future = self.executor.submit(decode)
        self.record["applied"] = True

    def finish(self):
        if self.future is None:
            return None
        result = self.future.result()
        self.record["joined_at"] = time.monotonic()
        self.record["overlap_wall_seconds"] = (
            self.record["joined_at"] - self.record["submitted_at"]
        )
        return result

    def __exit__(self, kind, error, traceback):
        try:
            self.finish()
        except BaseException as decoder_error:
            if error is None:
                raise
            error.add_note(
                "The owned video decoder also failed: " + type(decoder_error).__name__
            )
        finally:
            if self.executor is not None:
                self.executor.shutdown(wait=True, cancel_futures=True)
