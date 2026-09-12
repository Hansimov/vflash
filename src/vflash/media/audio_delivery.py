"""Public, bounded audio delivery profiles for decoded H3 waveforms."""

from __future__ import annotations

import json
import math
import subprocess
import time
from collections.abc import Sequence

from vflash.native.errors import VflashNativeError

AUDIO_DELIVERY_PROFILES = ("unchanged", "web-v1")
WEB_AUDIO_PROFILE = "web-v1"
TARGET_LUFS = -18.0
TRUE_PEAK_DBTP = -2.0
TARGET_LRA = 11.0
MAX_GAIN_DB = 18.0
QUIET_FLOOR_LUFS = -55.0


class AudioDeliveryError(VflashNativeError):
    """The requested delivery profile could not measure decoded audio."""


def _loudness_measurement(
    ffmpeg: str,
    input_arguments: Sequence[str],
    preceding_filters: Sequence[str] = (),
) -> tuple[dict[str, float], float]:
    started = time.monotonic()
    options = f"I={TARGET_LUFS}:TP={TRUE_PEAK_DBTP}:LRA={TARGET_LRA}"
    try:
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-nostdin",
                *input_arguments,
                "-map",
                "0:a:0",
                "-af",
                ",".join([*preceding_filters, f"loudnorm={options}:print_format=json"]),
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioDeliveryError("FFmpeg could not measure audio loudness") from exc
    if completed.returncode:
        raise AudioDeliveryError(
            "FFmpeg audio loudness measurement failed: "
            + completed.stderr[-2000:].strip()
        )
    try:
        stats, _ = json.JSONDecoder().raw_decode(
            completed.stderr[completed.stderr.rindex("{") :]
        )
        measured = {
            name: float(stats[name])
            for name in ("input_i", "input_tp", "input_lra", "input_thresh")
        }
    except (ValueError, KeyError, TypeError):
        raise AudioDeliveryError(
            "FFmpeg did not report usable audio loudness measurements"
        ) from None
    return measured, time.monotonic() - started


def web_loudness_filter(
    ffmpeg: str,
    input_arguments: Sequence[str],
    preceding_filters: Sequence[str] = (),
) -> tuple[str | None, dict[str, float | str | None]]:
    """Return one bounded gain/limiter filter and its non-semantic evidence."""

    measured, measurement_seconds = _loudness_measurement(
        ffmpeg, input_arguments, preceding_filters
    )
    report: dict[str, float | str | None] = {
        "profile": WEB_AUDIO_PROFILE,
        "input_lufs": measured["input_i"] if math.isfinite(measured["input_i"]) else None,
        "input_true_peak_dbtp": measured["input_tp"]
        if math.isfinite(measured["input_tp"])
        else None,
        "measurement_seconds": measurement_seconds,
    }
    if measured["input_i"] < QUIET_FLOOR_LUFS or not all(
        math.isfinite(value) for value in measured.values()
    ):
        return None, {**report, "status": "quiet_bypass"}
    gain = min(MAX_GAIN_DB, TARGET_LUFS - measured["input_i"])
    limited = measured["input_tp"] + gain > TRUE_PEAK_DBTP
    specification = f"volume={gain:.4f}dB"
    if limited:
        peak = 10 ** (TRUE_PEAK_DBTP / 20)
        specification += (
            f",aresample=192000,alimiter=limit={peak:.9f}:level=false:latency=true"
        )
    return specification, {
        **report,
        "status": "normalized",
        "target_lufs": measured["input_i"] + gain,
        "true_peak_limit_dbtp": TRUE_PEAK_DBTP,
        "gain_db": gain,
        "max_gain_db": MAX_GAIN_DB,
        "mode": "limited" if limited else "linear",
    }
