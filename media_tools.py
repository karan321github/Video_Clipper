"""
media_tools.py — shared subprocess / ffprobe helpers
=====================================================
• run_process(): runs a child process without blocking on pipe buffers,
  enforces a timeout, supports cancellation, and streams output lines to
  callbacks (used for ffmpeg progress and yt-dlp download progress).
• run_ffmpeg(): run_process() + ffmpeg `-progress` parsing.
• probe_media(): one ffprobe call returning duration and stream info.
• redact(): strips query strings from URLs before commands are logged.
"""

from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import config

log = logging.getLogger(__name__)

_URL_RE = re.compile(r"(https?://[^\s?'\"]+)\?[^\s'\"]*")


def redact(text: str) -> str:
    """Remove URL query strings (signed tokens, keys) from loggable text."""
    return _URL_RE.sub(r"\1?<redacted>", text)


def format_cmd(cmd: Sequence[str]) -> str:
    return redact(shlex.join(str(part) for part in cmd))


@dataclass
class ProcessResult:
    returncode: int
    stdout_tail: list[str]
    stderr_tail: list[str]
    timed_out: bool = False
    cancelled: bool = False

    @property
    def stderr(self) -> str:
        return "\n".join(self.stderr_tail)

    def last_error_line(self) -> str:
        """
        The most informative stderr line. ffmpeg prints the root cause
        first, followed by a cascade of generic "task finished" messages,
        so return the first line that isn't known noise.
        """
        lines = [l.strip() for l in self.stderr_tail if l.strip()]
        for line in lines:
            if not any(noise in line for noise in _STDERR_NOISE):
                return redact(_STREAM_PREFIX_RE.sub("", line))
        return redact(_STREAM_PREFIX_RE.sub("", lines[-1])) if lines else ""

    @property
    def signed_returncode(self) -> int:
        """Windows reports negative exit codes as unsigned 32-bit values."""
        rc = self.returncode
        return rc - 2**32 if rc is not None and rc >= 2**31 else rc


_STDERR_NOISE = (
    "Fontconfig error",
    "Task finished with error code",
    "Terminating thread with return code",
    "Nothing was written into output file",
    "Conversion failed",
    "Error sending frames to consumers",
    "Could not open encoder before EOF",
    "Error while opening encoder - maybe incorrect parameters",
    "WARNING: ",
)
_STREAM_PREFIX_RE = re.compile(r"^(?:\[[^\]]+ @ [0-9A-Fa-fx]+\]\s*)+")


def run_process(
    cmd: Sequence[str],
    *,
    timeout: float,
    on_stdout: Optional[Callable[[str], None]] = None,
    on_stderr: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cwd: Optional[Path] = None,
    tail_lines: int = 200,
) -> ProcessResult:
    """Run `cmd`, draining stdout/stderr on background threads."""
    stdout_tail: deque[str] = deque(maxlen=tail_lines)
    stderr_tail: deque[str] = deque(maxlen=tail_lines)

    proc = subprocess.Popen(
        [str(part) for part in cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd else None,
    )

    def _pump(stream, tail, callback):
        try:
            for line in stream:
                line = line.rstrip("\r\n")
                tail.append(line)
                if callback:
                    try:
                        callback(line)
                    except Exception:  # never let a callback kill the reader
                        log.debug("output callback failed", exc_info=True)
        finally:
            stream.close()

    readers = [
        threading.Thread(target=_pump, args=(proc.stdout, stdout_tail, on_stdout), daemon=True),
        threading.Thread(target=_pump, args=(proc.stderr, stderr_tail, on_stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    timed_out = cancelled = False
    while True:
        try:
            proc.wait(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
            elif time.monotonic() > deadline:
                timed_out = True
            else:
                continue
            proc.kill()
            proc.wait()
            break

    for reader in readers:
        reader.join(timeout=5)

    return ProcessResult(
        returncode=proc.returncode,
        stdout_tail=list(stdout_tail),
        stderr_tail=list(stderr_tail),
        timed_out=timed_out,
        cancelled=cancelled,
    )


def run_ffmpeg(
    cmd: Sequence[str],
    *,
    timeout: float,
    expected_duration: Optional[float] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancel_event: Optional[threading.Event] = None,
    cwd: Optional[Path] = None,
) -> ProcessResult:
    """
    Run an ffmpeg command. When `on_progress` and `expected_duration` are
    given, ffmpeg's machine-readable progress is parsed and reported as a
    0.0–1.0 fraction.
    """
    cmd = list(cmd)
    on_stdout = None
    if on_progress and expected_duration and expected_duration > 0:
        cmd[1:1] = ["-progress", "pipe:1", "-nostats"]

        def on_stdout(line: str) -> None:
            if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
                value = line.split("=", 1)[1].strip()
                if value.lstrip("-").isdigit():
                    seconds = int(value) / 1_000_000
                    on_progress(max(0.0, min(1.0, seconds / expected_duration)))
            elif line == "progress=end":
                on_progress(1.0)

    log.debug("ffmpeg command: %s", format_cmd(cmd))
    return run_process(
        cmd, timeout=timeout, on_stdout=on_stdout, cancel_event=cancel_event, cwd=cwd
    )


# ---------------------------------------------------------------------------
# ffprobe
# ---------------------------------------------------------------------------

@dataclass
class MediaInfo:
    path: Path
    duration: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    width: int = 0
    height: int = 0
    video_codec: str = ""
    audio_codec: str = ""
    format_name: str = ""


def probe_media(path: Path, timeout: float = 30) -> Optional[MediaInfo]:
    """Return stream/duration information for a local media file, or None."""
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration,format_name",
        "-show_entries", "stream=codec_type,codec_name,width,height,duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log.error("ffprobe failed for %s: %s", path, exc)
        return None
    if result.returncode != 0:
        log.error("ffprobe could not read %s: %s", path, result.stderr.strip()[-500:])
        return None
    try:
        meta = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        log.error("ffprobe returned invalid JSON for %s", path)
        return None

    info = MediaInfo(path=path, format_name=meta.get("format", {}).get("format_name", ""))
    try:
        info.duration = float(meta.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0

    stream_durations = []
    for stream in meta.get("streams", []):
        kind = stream.get("codec_type")
        if kind == "video" and not info.has_video and stream.get("codec_name") not in ("mjpeg", "png"):
            info.has_video = True
            info.video_codec = stream.get("codec_name", "")
            info.width = int(stream.get("width") or 0)
            info.height = int(stream.get("height") or 0)
        elif kind == "audio" and not info.has_audio:
            info.has_audio = True
            info.audio_codec = stream.get("codec_name", "")
        try:
            stream_durations.append(float(stream.get("duration")))
        except (TypeError, ValueError):
            pass
    if info.duration <= 0 and stream_durations:
        info.duration = max(stream_durations)
    return info
