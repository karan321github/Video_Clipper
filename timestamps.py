"""
timestamps.py — timestamp parsing, formatting and clip-range validation
========================================================================
All clip ranges are expressed in seconds on the *source video's* timeline
(0 = first frame of the video), so detected hype moments, user-entered
timestamps and the FFmpeg cut points all share one coordinate system.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

# Allow a little slack for container durations that are rounded by YouTube.
DURATION_EPSILON = 0.05


class TimestampError(ValueError):
    """Raised for malformed or out-of-range timestamps."""


_HMS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{1,2}(?:\.\d+)?)$")


def parse_timestamp(value: str | float | int) -> float:
    """
    Parse "HH:MM:SS[.fff]", "MM:SS[.fff]" or plain seconds into seconds.

    >>> parse_timestamp("00:02:15")
    135.0
    >>> parse_timestamp("2:15")
    135.0
    >>> parse_timestamp("135")
    135.0
    """
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value).strip()
        if not text:
            raise TimestampError("Timestamp is empty.")
        if text.startswith("-"):
            raise TimestampError(f"Timestamp cannot be negative: {text!r}.")
        match = _HMS_RE.match(text)
        if match:
            hours, minutes, secs = match.groups()
            minutes_i, secs_f = int(minutes), float(secs)
            if secs_f >= 60:
                raise TimestampError(f"Seconds must be below 60 in {text!r}.")
            if hours is not None and minutes_i >= 60:
                raise TimestampError(f"Minutes must be below 60 in {text!r}.")
            seconds = int(hours or 0) * 3600 + minutes_i * 60 + secs_f
        else:
            try:
                seconds = float(text)
            except ValueError:
                raise TimestampError(
                    f"Invalid timestamp {text!r}. Use HH:MM:SS, MM:SS or seconds."
                ) from None
    if math.isnan(seconds) or math.isinf(seconds):
        raise TimestampError(f"Invalid timestamp: {value!r}.")
    if seconds < 0:
        raise TimestampError(f"Timestamp cannot be negative: {value!r}.")
    return seconds


def format_timestamp(seconds: float, millis: bool = False) -> str:
    """Format seconds as HH:MM:SS (or HH:MM:SS.mmm)."""
    seconds = max(0.0, seconds)
    if millis:
        whole = int(seconds)
        ms = int(round((seconds - whole) * 1000))
        if ms == 1000:
            whole, ms = whole + 1, 0
        h, rem = divmod(whole, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    whole = int(round(seconds))
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def filename_timestamp(seconds: float) -> str:
    """Filesystem-safe timestamp, e.g. 00h02m15s."""
    whole = int(seconds)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


@dataclass
class ClipRange:
    start: float
    end: float
    hype_time: Optional[float] = None   # detected moment, if any

    @property
    def duration(self) -> float:
        return self.end - self.start

    def label(self) -> str:
        return f"{format_timestamp(self.start)}-{format_timestamp(self.end)}"


def validate_range(
    start: float,
    end: float,
    video_duration: float,
    min_duration: float = 1.0,
) -> ClipRange:
    """
    Validate a [start, end] range against the source duration.
    Raises TimestampError with a user-readable explanation.
    """
    if start < 0 or end < 0:
        raise TimestampError("Timestamps cannot be negative.")
    if end <= start:
        raise TimestampError(
            f"End time ({format_timestamp(end, True)}) must be after "
            f"start time ({format_timestamp(start, True)})."
        )
    if video_duration > 0:
        if start >= video_duration:
            raise TimestampError(
                f"Start time {format_timestamp(start, True)} is beyond the video "
                f"duration ({format_timestamp(video_duration, True)})."
            )
        if end > video_duration + DURATION_EPSILON:
            raise TimestampError(
                f"End time {format_timestamp(end, True)} is beyond the video "
                f"duration ({format_timestamp(video_duration, True)})."
            )
        end = min(end, video_duration)
    if end - start < min_duration:
        raise TimestampError(
            f"Clip is only {end - start:.2f} s long; minimum is {min_duration:g} s."
        )
    return ClipRange(start=start, end=end)


def parse_range(text: str) -> tuple[float, float]:
    """Parse "START-END" (e.g. "00:02:15-00:02:55" or "135-175")."""
    parts = [p.strip() for p in text.split("-")]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise TimestampError(
            f"Invalid range {text!r}. Use START-END, e.g. 00:02:15-00:02:55."
        )
    return parse_timestamp(parts[0]), parse_timestamp(parts[1])


def build_hype_range(
    hype_time: float,
    before: float,
    after: float,
    video_duration: float,
) -> ClipRange:
    """
    Convert a detected hype moment into a clip window [t-before, t+after]
    on the source timeline. The window is shifted (not shrunk) when it
    would run past either end of the video, and only shrunk when the
    video itself is shorter than the requested window.
    """
    if before < 0 or after < 0:
        raise TimestampError("Padding cannot be negative.")
    length = before + after
    if length <= 0:
        raise TimestampError("Clip duration must be greater than zero.")
    hype_time = max(0.0, hype_time)
    if video_duration > 0:
        hype_time = min(hype_time, video_duration)
        length = min(length, video_duration)
    start = hype_time - before
    if start < 0:
        start = 0.0
    end = start + length
    if video_duration > 0 and end > video_duration:
        end = video_duration
        start = max(0.0, end - length)
    return ClipRange(start=start, end=end, hype_time=hype_time)
