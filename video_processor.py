"""
video_processor.py — GTAClipper Regular Video Processor
=========================================================
Handles non-live YouTube videos (published VODs, completed livestream
replays, podcasts).

Workflow
--------
1. Probe the URL with yt-dlp to detect the video type
   (live / upcoming / post-live / archived replay / regular video).
2. Download the source once (≤1080p, any codec) — or reuse a validated
   cached copy — merging separate audio/video streams into one file.
3. Scan the audio track in a single ffmpeg pass (VODHypeDetector) to find
   hype moments as offsets on the source timeline.
4. ClipMaker cuts each [start, end] range straight from the source file.

Note: earlier versions stream-copied the download into MPEG-TS segments
(OfflineSegmentProvider). That silently dropped VP9/AV1 video, so the VOD
pipeline no longer uses it; the class is kept for compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from media_tools import probe_media, run_process, format_cmd
from recorder import SegmentInfo, StreamRecorder
from chat_reader import ChatReader, ChatMessage

log = logging.getLogger(__name__)


class VideoSourceError(Exception):
    """A video could not be probed or downloaded; message is user-readable."""


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

@dataclass
class VideoInfo:
    """Metadata retrieved from yt-dlp for a given URL."""
    url:         str
    video_id:    str         = ""
    title:       str         = ""
    duration:    float       = 0.0          # seconds (0 if live / unknown)
    is_live:     bool        = False        # currently live
    was_live:    bool        = False        # archived livestream
    is_upcoming: bool        = False        # scheduled but not started
    uploader:    str         = ""
    source_type: str         = "unknown"   # "live" | "archived" | "video"
    live_status: str         = ""          # raw yt-dlp live_status


# (substring in yt-dlp stderr, lower-cased) → user-readable reason
_YTDLP_ERRORS: list[tuple[str, str]] = [
    ("private video", "The video is private."),
    ("confirm your age", "The video is age-restricted and requires sign-in."),
    ("age-restricted", "The video is age-restricted and requires sign-in."),
    ("not a bot", "YouTube requires sign-in to confirm you're not a bot (try again later or use cookies)."),
    ("members-only", "The video is members-only."),
    ("join this channel", "The video is members-only."),
    ("this live event will begin", "The livestream has not started yet."),
    ("premieres in", "The video is a premiere that has not started yet."),
    ("not available in your country", "The video is region-restricted."),
    ("blocked it in your country", "The video is region-restricted."),
    ("copyright", "The video was removed for copyright reasons."),
    ("has been removed", "The video has been removed."),
    ("video unavailable", "The video is unavailable."),
    ("video is unavailable", "The video is unavailable."),
    ("is not available", "The video is unavailable."),
    ("unsupported url", "This URL is not supported."),
    ("requested format is not available", "No downloadable format is available for this video."),
    ("no space left", "Insufficient disk space for the download."),
    ("unable to download", "Network error while contacting YouTube."),
    ("timed out", "Network timeout while contacting YouTube."),
]


def describe_ytdlp_error(stderr: str) -> str:
    """Translate yt-dlp stderr into a short user-readable reason."""
    lowered = stderr.lower()
    for needle, reason in _YTDLP_ERRORS:
        if needle in lowered:
            return reason
    errors = [l for l in stderr.splitlines() if l.startswith("ERROR")]
    return (errors[-1] if errors else "yt-dlp failed.")[:300]


def probe_video_detailed(url: str) -> VideoInfo:
    """
    Run yt-dlp --dump-json to retrieve video metadata without downloading.
    Raises VideoSourceError with a user-readable reason on failure.
    """
    log.info("Probing video metadata: %s", url)
    cmd = [
        config.YTDLP_BIN,
        *config.ytdlp_js_runtime_args(url),
        "--dump-json",
        "--no-download",
        "--no-playlist",
        url,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90,
        )
    except FileNotFoundError:
        raise VideoSourceError("yt-dlp not found. Install with: pip install yt-dlp") from None
    except subprocess.TimeoutExpired:
        raise VideoSourceError("Timed out retrieving video information.") from None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        reason = describe_ytdlp_error(stderr)
        log.error("yt-dlp metadata probe failed (%s):\n%s", reason, stderr[-1500:])
        raise VideoSourceError(reason)

    try:
        meta = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise VideoSourceError(f"Could not parse yt-dlp output: {exc}") from None

    live_status = str(meta.get("live_status") or "")
    is_live     = bool(meta.get("is_live", False)) or live_status == "is_live"
    is_upcoming = bool(meta.get("is_upcoming", False)) or live_status == "is_upcoming"
    was_live    = (
        bool(meta.get("was_live", False))
        or live_status in ("was_live", "post_live")
    )
    duration    = float(meta.get("duration") or 0.0)

    if is_live:
        source_type = "live"
    elif was_live:
        source_type = "archived"
    else:
        source_type = "video"

    info = VideoInfo(
        url=url,
        video_id=meta.get("id", ""),
        title=meta.get("title", ""),
        duration=duration,
        is_live=is_live,
        was_live=was_live,
        is_upcoming=is_upcoming,
        uploader=meta.get("uploader", ""),
        source_type=source_type,
        live_status=live_status,
    )

    log.info(
        "Video info — id=%s | type=%s | live_status=%s | duration=%.0fs | title=%r",
        info.video_id, info.source_type, live_status or "?", info.duration, info.title,
    )
    if live_status == "post_live":
        log.warning(
            "This stream just ended and YouTube is still processing the replay; "
            "the downloadable portion may be incomplete."
        )
    return info


def probe_video(url: str) -> Optional[VideoInfo]:
    """Like probe_video_detailed() but returns None on failure."""
    try:
        return probe_video_detailed(url)
    except VideoSourceError as exc:
        log.error("Could not retrieve video info for %s: %s", url, exc)
        return None

# ---------------------------------------------------------------------------
# Normalize YouTube URL variants
# ---------------------------------------------------------------------------

_YT_ID_RE = re.compile(
    r"""
    (?:youtube\.com/(?:watch\?v=|live/|shorts/|embed/)|
       youtu\.be/)
    ([A-Za-z0-9_\-]{11})
    """,
    re.VERBOSE,
)


def normalize_youtube_url(url: str) -> str:
    """
    Normalise any supported YouTube URL to the canonical watch?v=ID form
    so that yt-dlp resolves it consistently.
    """
    m = _YT_ID_RE.search(url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    # Fallback: return as-is; yt-dlp handles most URL variants natively
    return url


# ---------------------------------------------------------------------------
# Offline / VOD recorder — builds virtual segment list from a local file
# ---------------------------------------------------------------------------

class OfflineSegmentProvider:
    """
    Mimics StreamRecorder's interface for a locally downloaded video file.

    Splits the file into virtual SEGMENT_DURATION-second chunks (without
    re-encoding) and registers them as SegmentInfo objects so the
    HypeDetector and ClipMaker can operate unmodified.
    """

    def __init__(self, video_path: Path, duration: float):
        self._path     = video_path
        self._duration = duration
        self._segments: list[SegmentInfo] = []
        self._lock     = threading.Lock()
        self.video_start_time = 0.0

    def prepare(self) -> bool:
        """
        Segment the downloaded file into .ts chunks inside TMP_DIR.
        Returns True on success.
        """
        if not self._path.exists():
            log.error("Video file not found: %s", self._path)
            return False

        prefix = f"vod_{self._path.stem}"
        seg_pattern = str(config.TMP_DIR / f"{prefix}_%05d.ts")

        cmd = [
            config.FFMPEG_BIN,
            "-hide_banner", "-loglevel", "warning",
            "-y",
            "-i", str(self._path),
            "-c", "copy",
            "-f", "segment",
            "-segment_time", str(config.SEGMENT_DURATION),
            "-segment_format", "mpegts",
            "-reset_timestamps", "1",
            "-strftime", "0",
            seg_pattern,
        ]

        log.info("Segmenting video into %d-second chunks…", config.SEGMENT_DURATION)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                log.error("ffmpeg segmentation failed:\n%s", result.stderr[-2000:])
                return False
        except subprocess.TimeoutExpired:
            log.error("ffmpeg segmentation timed out.")
            return False

        # Collect generated .ts files
        ts_files = sorted(config.TMP_DIR.glob(f"{prefix}_*.ts"))
        if not ts_files:
            log.error("No .ts segments produced.")
            return False

        # Build SegmentInfo list with wall times aligned to the actual VOD
        base_time = time.time() - self._duration
        timeline_offset = 0.0
        with self._lock:
            self.video_start_time = base_time
            for idx, f in enumerate(ts_files):
                segment_duration = _probe_duration(f)
                if segment_duration <= 0:
                    segment_duration = min(
                        float(config.SEGMENT_DURATION),
                        max(0.0, self._duration - timeline_offset),
                    )
                if segment_duration <= 0:
                    continue
                wall_start = base_time + timeline_offset
                seg = SegmentInfo(
                    path=f,
                    index=idx,
                    wall_start=wall_start,
                    duration=segment_duration,
                )
                self._segments.append(seg)
                timeline_offset += segment_duration

        log.info("Prepared %d virtual segments from VOD.", len(self._segments))
        return bool(self._segments)

    # ── StreamRecorder-compatible API ──────────────────────────────────────

    def get_segments_covering(self, seconds: float) -> list[SegmentInfo]:
        with self._lock:
            segs = list(self._segments)
        if not segs:
            return []
        total = 0.0
        result: list[SegmentInfo] = []
        for seg in reversed(segs):
            result.insert(0, seg)
            total += seg.duration
            if total >= seconds:
                break
        return result

    def get_segments_for_window(
        self, spike_time: float, clip_duration: float
    ) -> list[SegmentInfo]:
        """Return source segments covering the clip window around a VOD spike."""
        with self._lock:
            segments = list(self._segments)
        if not segments:
            return []

        video_start = segments[0].wall_start
        video_end = segments[-1].wall_end
        if video_end - video_start < clip_duration:
            return segments

        clip_start = spike_time - config.CLIP_LOOKBACK
        clip_start = max(video_start, min(clip_start, video_end - clip_duration))
        clip_end = clip_start + clip_duration
        return [
            seg for seg in segments
            if seg.wall_end > clip_start and seg.wall_start < clip_end
        ]

    def latest_segment(self) -> Optional[SegmentInfo]:
        with self._lock:
            return self._segments[-1] if self._segments else None

    def segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    # Stub methods to satisfy HypeDetector (which calls recorder.stop() etc.)
    def start(self) -> None:  pass
    def stop(self)  -> None:  pass

    def cleanup(self) -> None:
        """Remove all .ts segments produced by this provider."""
        with self._lock:
            for seg in self._segments:
                try:
                    seg.path.unlink(missing_ok=True)
                except Exception:
                    pass
            self._segments.clear()


# ---------------------------------------------------------------------------
# Silence-based hype detector for VODs (no live chat)
# ---------------------------------------------------------------------------

class VODHypeDetector:
    """
    Analyses a fully-downloaded VOD file to find interesting segments using
    audio loudness peaks — no live chat required.

    Returns `base_wall_time + offset` for each detected moment, where offset
    is the start (in seconds) of the loud ANALYSIS_INTERVAL window on the
    source timeline. Pass base_wall_time=0 to get plain video offsets.
    """

    def find_spikes(
        self,
        video_path: Path,
        base_wall_time: float,
        max_clips: int = 20,
        threshold_db: float | None = None,
        clip_duration: float | None = None,
        duration: float | None = None,
    ) -> list[float]:
        """
        Analyse the VOD for loud audio segments.
        Moments are spaced at least one clip length apart so clips never
        overlap. Videos shorter than the clip length yield (at most) one
        moment; the caller shrinks the clip to the video.
        """
        if max_clips <= 0:
            log.warning("VOD max_clips must be greater than zero.")
            return []

        if not duration or duration <= 0:
            duration = _probe_duration(video_path)
        if duration <= 0:
            log.error("Could not determine the video duration; cannot analyse %s.", video_path)
            return []
        clip_duration = clip_duration or config.CLIP_DURATION
        effective_clip = min(clip_duration, duration)

        log.info("Analysing %.0f-second VOD for hype moments…", duration)

        interval = config.ANALYSIS_INTERVAL
        threshold = (
            config.VOD_LOUDNESS_SPIKE_DB
            if threshold_db is None
            else threshold_db
        )
        samples = self._measure_all_windows(video_path, duration, interval)
        if samples is None:
            log.info("Falling back to per-window loudness measurement.")
            samples = []
            t = 0.0
            while t < duration:
                rms = self._measure_loudness_at(video_path, t, min(interval, duration - t))
                samples.append((t, rms))
                t += interval

        valid = [(offset, rms) for offset, rms in samples if rms > -90.0]
        if not valid:
            log.error("No usable audio readings in VOD; cannot identify clip moments.")
            return []

        candidates: list[tuple[float, float]] = []
        for i, (offset, rms) in enumerate(samples):
            baseline_values = [
                value for _, value in samples[max(0, i - config.ROLLING_WINDOW_SIZE):i]
                if value > -90.0
            ]
            if (
                rms > -90.0
                and len(baseline_values) >= max(1, config.ROLLING_WINDOW_SIZE // 2)
                and rms >= sum(baseline_values) / len(baseline_values) + threshold
            ):
                candidates.append((offset, rms))

        if not candidates and valid:
            log.warning(
                "No VOD windows exceeded the %.1f dB threshold; selecting the loudest windows instead.",
                threshold,
            )
            candidates = valid

        # A cooldown shorter than a clip would produce overlapping outputs.
        clip_spacing = max(config.CLIP_COOLDOWN_SECONDS, effective_clip)
        selected: list[tuple[float, float]] = []
        for candidate in sorted(candidates, key=lambda sample: sample[1], reverse=True):
            if all(
                abs(candidate[0] - chosen[0]) >= clip_spacing
                for chosen in selected
            ):
                selected.append(candidate)
                if len(selected) >= max_clips:
                    break

        selected.sort(key=lambda sample: sample[0])
        spike_times = [base_wall_time + offset for offset, _ in selected]
        for offset, rms in selected:
            log.info("VOD spike at %.0fs (rms=%.1f dB)", offset, rms)
        log.info(
            "VOD loudness scan: %d/%d usable windows; threshold=%.1f dB.",
            len(valid),
            len(samples),
            threshold,
        )
        log.info("Found %d spike(s) in VOD.", len(spike_times))
        return spike_times

    def _measure_all_windows(
        self, path: Path, duration: float, interval: float
    ) -> Optional[list[tuple[float, float]]]:
        """
        Measure the RMS level of every `interval`-second window in a single
        decode pass (much faster than one ffmpeg process per window).
        Returns [(offset, rms_db)], or None so the caller can fall back.
        """
        sample_rate = 16000
        window = max(1, int(interval * sample_rate))
        cmd = [
            config.FFMPEG_BIN, "-hide_banner", "-nostats",
            "-i", str(path),
            "-map", "0:a:0", "-vn",
            "-af",
            f"aresample={sample_rate},asetnsamples=n={window}:p=0,"
            "astats=metadata=1:reset=1,"
            "ametadata=mode=print:key=lavfi.astats.Overall.RMS_level:file=-",
            "-f", "null", "-",
        ]
        samples: list[tuple[float, float]] = []
        pending: list[float] = []
        last_logged = [time.monotonic()]

        def on_line(line: str) -> None:
            if line.startswith("frame:"):
                m = re.search(r"pts_time:([\d.]+)", line)
                if m:
                    pending.append(float(m.group(1)))
                    if time.monotonic() - last_logged[0] > 10:
                        last_logged[0] = time.monotonic()
                        log.info("Hype scan: %.0f%%", min(100.0, 100 * pending[-1] / duration))
            elif line.startswith("lavfi.astats.Overall.RMS_level=") and pending:
                raw = line.split("=", 1)[1].strip()
                try:
                    value = float(raw)
                except ValueError:
                    value = -99.0
                if value != value or value < -200:   # nan / -inf (digital silence)
                    value = -99.0
                samples.append((round(pending.pop(), 3), value))

        try:
            res = run_process(cmd, timeout=max(600, duration * 0.5), on_stdout=on_line)
        except FileNotFoundError:
            log.error("ffmpeg not found while measuring VOD loudness.")
            return None
        if res.returncode != 0 or not samples:
            log.warning(
                "Single-pass loudness scan failed (exit %s): %s",
                res.returncode, res.last_error_line(),
            )
            return None
        log.info("Hype scan: measured %d windows.", len(samples))
        return samples

    def _measure_loudness_at(
        self, path: Path, offset: float, duration: float
    ) -> float:
        """Measure RMS loudness of a [offset, offset+duration] window."""
        cmd = [
            config.FFMPEG_BIN,
            "-hide_banner",
            "-nostats",
            "-ss", f"{offset:.1f}",
            "-t", f"{duration:.1f}",
            "-i", str(path),
            "-vn",
            "-af", "volumedetect",
            "-f", "null",
            "-",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                match = re.search(
                    r"mean_volume:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB",
                    r.stderr,
                    re.IGNORECASE,
                )
                if match:
                    value = float(match.group(1))
                    return value if value > -200 else -99.0
                log.warning("ffmpeg returned no mean-volume reading at %.1f s.", offset)
            else:
                log.warning(
                    "ffmpeg loudness measurement failed at %.1f s: %s",
                    offset,
                    r.stderr[-500:].strip(),
                )
        except subprocess.TimeoutExpired:
            log.warning("ffmpeg loudness measurement timed out at %.1f s.", offset)
        except FileNotFoundError:
            log.error("ffmpeg not found while measuring VOD loudness.")
        return -99.0


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def _validate_cached_download(
    path: Path, expected_duration: float, require_audio: bool = True
) -> Optional[str]:
    """Return None if `path` is a complete, usable download, else why not."""
    media = probe_media(path)
    if media is None:
        return "unreadable"
    if not media.has_video:
        return "no video stream"
    if require_audio and not media.has_audio:
        return "no audio stream"
    if expected_duration > 0 and media.duration + 5 < expected_duration:
        return f"truncated ({media.duration:.0f}s of {expected_duration:.0f}s)"
    return None


def download_video(
    url: str,
    out_dir: Path,
    video_id: str,
    expected_duration: float = 0.0,
) -> Path:
    """
    Download a YouTube video (≤1080p, separate audio+video merged into one
    file) using yt-dlp and return its path. A previous complete download
    of the same video is reused.

    The download runs in a private temp directory and is moved into the
    cache atomically, so concurrent jobs never see a partial file and an
    interrupted download (e.g. a leftover video-only `ID.f399.mp4`) is
    never mistaken for a finished one.

    Raises VideoSourceError with a user-readable reason on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = re.sub(r"[^\w\-]", "_", video_id) or "video"

    for ext in (".mp4", ".mkv", ".webm"):
        cached = out_dir / f"{safe_id}{ext}"
        if cached.exists():
            problem = _validate_cached_download(cached, expected_duration)
            if problem is None:
                log.info("Reusing downloaded file: %s", cached)
                return cached
            log.warning("Ignoring cached file %s (%s).", cached.name, problem)

    free_mb = shutil.disk_usage(out_dir).free / (1024 * 1024)
    if free_mb < config.MIN_FREE_DISK_MB:
        raise VideoSourceError(
            f"Insufficient disk space: only {free_mb:.0f} MB free in {out_dir}."
        )

    job_dir = Path(tempfile.mkdtemp(prefix=f".dl_{safe_id}_", dir=out_dir))
    try:
        cmd = [
            config.YTDLP_BIN,
            *config.ytdlp_js_runtime_args(url),
            "--no-playlist",
            "-f", config.VOD_FORMAT,
            "-S", config.VOD_FORMAT_SORT,
            "--merge-output-format", "mp4/mkv",
            "-o", str(job_dir / f"{safe_id}.%(ext)s"),
            "--print", "after_move:filepath",
            "--progress", "--newline",
            url,
        ]
        log.info("Downloading video: %s", url)
        log.debug("yt-dlp command: %s", format_cmd(cmd))

        last_pct = [-10.0]

        def on_line(line: str) -> None:
            m = re.search(r"\[download\]\s+([\d.]+)%", line)
            if m:
                pct = float(m.group(1))
                if pct >= last_pct[0] + 10 or pct >= 100:
                    last_pct[0] = pct
                    log.info("Download: %s", line.split("]", 1)[1].strip())

        try:
            res = run_process(
                cmd, timeout=config.VOD_DOWNLOAD_TIMEOUT, on_stdout=on_line, on_stderr=on_line,
            )
        except FileNotFoundError:
            raise VideoSourceError("yt-dlp not found. Install with: pip install yt-dlp") from None
        if res.timed_out:
            raise VideoSourceError(
                f"Video download timed out after {config.VOD_DOWNLOAD_TIMEOUT} s."
            )
        if res.returncode != 0:
            reason = describe_ytdlp_error(res.stderr)
            log.error("yt-dlp download failed (exit %d):\n%s", res.returncode, res.stderr[-2000:])
            raise VideoSourceError(f"Video download failed: {reason}")

        candidates = [Path(l.strip()) for l in res.stdout_tail if l.strip()]
        candidates = [p for p in candidates if p.exists()]
        if not candidates:
            candidates = [
                p for p in job_dir.glob(f"{safe_id}.*")
                if p.suffix in (".mp4", ".mkv", ".webm")
            ]
        if not candidates:
            raise VideoSourceError("Video download failed: yt-dlp produced no file.")
        downloaded = candidates[-1]

        problem = _validate_cached_download(downloaded, expected_duration, require_audio=False)
        if problem and problem.startswith("truncated"):
            log.warning("Downloaded file looks %s; continuing with what is available.", problem)
        elif problem:
            raise VideoSourceError(f"Downloaded media is unusable: {problem}.")

        final = out_dir / f"{safe_id}{downloaded.suffix}"
        try:
            os.replace(downloaded, final)
        except OSError as exc:
            if final.exists() and _validate_cached_download(
                final, expected_duration, require_audio=False
            ) is None:
                log.info("Another job finished the same download first; reusing it.")
            else:
                raise VideoSourceError(f"Could not store the download: {exc}") from None
        log.info("Downloaded: %s (%.1f MB)", final.name, final.stat().st_size / 1e6)
        return final
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _probe_duration(video_path: Path) -> float:
    """Return the duration of a local video file in seconds."""
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return float(r.stdout.strip().split()[0])
    except Exception:
        pass
    return 0.0


def _has_video_stream(video_path: Path) -> bool:
    """Return whether a local media file contains a video stream."""
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        log.error("ffprobe not found while validating downloaded video.")
        return False
    except subprocess.TimeoutExpired:
        log.error("Timed out checking for a video stream in %s.", video_path)
        return False
    if result.returncode != 0:
        log.warning("Could not inspect media streams in %s: %s", video_path, result.stderr.strip())
        return False
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Null chat reader (for VODs without live chat)
# ---------------------------------------------------------------------------

class NullChatReader(ChatReader):
    """
    A no-op ChatReader used when processing regular (non-live) videos
    that have no live chat stream available.
    """

    def __init__(self):
        super().__init__()

    def start(self) -> None:
        log.info("NullChatReader: no live chat for this video type.")

    def stop(self) -> None:
        self._stop_event.set()
