"""
video_processor.py — GTAClipper Regular Video Processor
=========================================================
Handles non-live YouTube videos (published VODs, archived streams, podcasts).

Workflow
--------
1. Probe the URL with yt-dlp to detect video type (live / archived / regular).
2. Download the full video (or use an already-downloaded file) using yt-dlp.
3. Split the downloaded video into SEGMENT_DURATION-second virtual segments.
4. Run the same HypeDetector → ClipMaker pipeline used for live streams.
5. After all clips are produced, clean up.

This module is intentionally designed to reuse existing components:
  • yt-dlp for download & metadata
  • ffmpeg for segmentation & encoding (via ClipMaker)
  • HypeDetector for highlight identification
  • ClipMaker for clip rendering
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config
from recorder import SegmentInfo, StreamRecorder
from clip_maker import ClipMaker
from chat_reader import ChatReader, ChatMessage

log = logging.getLogger(__name__)


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


def probe_video(url: str) -> Optional[VideoInfo]:
    """
    Run yt-dlp --dump-json to retrieve video metadata without downloading.

    Returns a VideoInfo on success, None on error (private / unavailable / etc.).
    """
    log.info("Probing video metadata: %s", url)
    cmd = [
        config.YTDLP_BIN,
        "--dump-json",
        "--no-download",
        "--no-playlist",
    ]
    # YouTube requires the deno JS runtime for certain requests
    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        cmd += ["--js-runtimes", "deno"]
    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except FileNotFoundError:
        log.error("yt-dlp not found. Install with: pip install yt-dlp")
        return None
    except subprocess.TimeoutExpired:
        log.error("yt-dlp metadata probe timed out for %s", url)
        return None

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # Provide meaningful diagnostics for common failure modes
        if "Private video" in stderr:
            log.error("Video is private: %s", url)
        elif "age-restricted" in stderr.lower():
            log.error("Video is age-restricted: %s", url)
        elif "not available" in stderr.lower() or "unavailable" in stderr.lower():
            log.error("Video is unavailable: %s", url)
        elif "region" in stderr.lower():
            log.error("Video is region-restricted: %s", url)
        else:
            log.error("yt-dlp metadata probe failed:\n%s", stderr[:1000])
        return None

    try:
        meta = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        log.error("Could not parse yt-dlp JSON output: %s", exc)
        return None

    is_live     = bool(meta.get("is_live", False))
    was_live    = bool(meta.get("was_live", False))
    is_upcoming = bool(meta.get("is_upcoming", False))
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
    )

    log.info(
        "Video info — id=%s | type=%s | duration=%.0fs | title=%r",
        info.video_id, info.source_type, info.duration, info.title,
    )
    return info


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

        # Build SegmentInfo list with synthetic wall_start values
        base_time = time.time() - self._duration
        with self._lock:
            for f in ts_files:
                try:
                    idx = int(f.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue
                wall_start = base_time + idx * config.SEGMENT_DURATION
                seg = SegmentInfo(
                    path=f,
                    index=idx,
                    wall_start=wall_start,
                    duration=float(config.SEGMENT_DURATION),
                )
                self._segments.append(seg)

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
    audio loudness peaks and scene changes — no live chat required.

    Returns a list of wall_start times (seconds offset into the video) where
    hype spikes are detected, spaced by CLIP_COOLDOWN_SECONDS.
    """

    def find_spikes(
        self,
        video_path: Path,
        base_wall_time: float,
        max_clips: int = 20,
    ) -> list[float]:
        """
        Analyse the VOD for loud audio segments.
        Returns a list of absolute wall-clock spike times.
        """
        duration = _probe_duration(video_path)
        if duration <= 0:
            log.warning("Could not determine video duration; using whole video.")
            duration = 3600.0

        log.info("Analysing %.0f-second VOD for hype moments…", duration)

        # Sample one loudness reading per ANALYSIS_INTERVAL seconds
        interval   = config.ANALYSIS_INTERVAL
        window_rms: list[float] = []
        spikes: list[float]     = []
        last_spike_offset       = -config.CLIP_COOLDOWN_SECONDS

        t = 0.0
        while t < duration:
            rms = self._measure_loudness_at(video_path, t, interval)
            window_rms.append(rms)

            if len(window_rms) > config.ROLLING_WINDOW_SIZE:
                avg = sum(window_rms[-config.ROLLING_WINDOW_SIZE:]) / config.ROLLING_WINDOW_SIZE
                if (
                    rms > -90.0
                    and avg > -90.0
                    and rms >= avg + config.LOUDNESS_SPIKE_DB
                    and (t - last_spike_offset) >= config.CLIP_COOLDOWN_SECONDS
                ):
                    spike_wall = base_wall_time + t
                    spikes.append(spike_wall)
                    last_spike_offset = t
                    log.info(
                        "VOD spike at %.0fs (rms=%.1f dB, avg=%.1f dB)",
                        t, rms, avg,
                    )
                    if len(spikes) >= max_clips:
                        break

            t += interval

        log.info("Found %d spike(s) in VOD.", len(spikes))
        return spikes

    def _measure_loudness_at(
        self, path: Path, offset: float, duration: float
    ) -> float:
        """Measure RMS loudness of a [offset, offset+duration] window."""
        cmd = [
            config.FFPROBE_BIN,
            "-v", "error",
            "-ss", f"{offset:.1f}",
            "-t",  f"{duration:.1f}",
            "-f", "lavfi",
            "-i", f"amovie={path.as_posix()},astats=metadata=1:reset=1",
            "-show_entries", "frame_tags=lavfi.astats.Overall.RMS_level",
            "-of", "csv=p=0",
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                vals = []
                for line in r.stdout.strip().splitlines():
                    try:
                        v = float(line)
                        if v > -200:
                            vals.append(v)
                    except ValueError:
                        pass
                if vals:
                    return sum(vals) / len(vals)
        except Exception:
            pass
        return -99.0


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_video(url: str, out_dir: Path, video_id: str) -> Optional[Path]:
    """
    Download a YouTube video using yt-dlp.

    Returns the path to the downloaded file, or None on failure.
    Skips the download if a file with the expected ID already exists.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (avoid re-downloading)
    for existing in out_dir.glob(f"*{video_id}*"):
        if existing.suffix in (".mp4", ".mkv", ".webm"):
            log.info("Using already-downloaded file: %s", existing)
            return existing

    out_tmpl = str(out_dir / "%(id)s.%(ext)s")
    cmd = [
        config.YTDLP_BIN,
        "--no-playlist",
        "-f", config.STREAM_QUALITY,
        "-o", out_tmpl,
        "--merge-output-format", "mp4",
        "--no-part",        # don't leave .part files on failure
    ]
    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        cmd += ["--js-runtimes", "deno"]
    cmd.append(url)

    log.info("Downloading video: %s", url)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            stderr = result.stderr.strip()
            log.error("yt-dlp download failed:\n%s", stderr[:2000])
            return None
    except subprocess.TimeoutExpired:
        log.error("yt-dlp download timed out.")
        return None

    # Locate the downloaded file
    for f in out_dir.glob(f"{video_id}.*"):
        if f.suffix in (".mp4", ".mkv", ".webm"):
            log.info("Downloaded: %s (%.1f MB)", f.name, f.stat().st_size / 1e6)
            return f

    log.error("Downloaded file not found in %s", out_dir)
    return None


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
