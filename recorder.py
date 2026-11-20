"""
recorder.py — GTAClipper Stream Recorder
=========================================
Strategy:
  1. Run yt-dlp with --get-url to resolve the best stream manifest/direct URL.
  2. Pipe that URL into ffmpeg with `-f segment -segment_time N` to produce
     rolling .ts files: tmp/seg_00001.ts, tmp/seg_00002.ts, …
  3. Maintain a deque of completed segments with their wall-clock timestamps.
  4. Expose `get_segments_covering(seconds)` so ClipMaker can retrieve the
     exact contiguous files needed for a 30-second clip.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

class SegmentInfo:
    """Metadata for a completed .ts segment."""

    __slots__ = ("path", "index", "wall_start", "wall_end", "duration")

    def __init__(self, path: Path, index: int, wall_start: float, duration: float):
        self.path       = path
        self.index      = index
        self.wall_start = wall_start               # time.time() when segment started
        self.wall_end   = wall_start + duration    # approximate end wall-clock time
        self.duration   = duration


# ---------------------------------------------------------------------------
# Main recorder class
# ---------------------------------------------------------------------------

class StreamRecorder:
    """
    Records a live stream into rolling SEGMENT_DURATION-second .ts segments.

    Usage::

        recorder = StreamRecorder(stream_url)
        recorder.start()
        # …
        recorder.stop()
    """

    def __init__(self, stream_url: str):
        self.stream_url   = stream_url
        self._segments: deque[SegmentInfo] = deque()
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._watcher_thread: Optional[threading.Thread] = None
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._segment_counter = 0
        self._segment_prefix = f"seg_{uuid.uuid4().hex}"

        # Ensure temp dir exists
        config.TMP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Resolve stream URL then launch ffmpeg segmenter in a background thread."""
        self._thread = threading.Thread(target=self._run, daemon=True, name="Recorder")
        self._thread.start()
        log.info("Recorder thread started.")

    def stop(self) -> None:
        """Signal the recorder to stop and wait for cleanup."""
        self._stop_event.set()
        if self._ffmpeg_proc and self._ffmpeg_proc.poll() is None:
            self._ffmpeg_proc.terminate()
            try:
                self._ffmpeg_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._ffmpeg_proc.kill()
        if self._thread:
            self._thread.join(timeout=15)
        log.info("Recorder stopped.")

    def get_segments_covering(self, seconds: float) -> list[SegmentInfo]:
        """
        Return the most recent completed segments whose combined wall-clock
        duration covers at least `seconds` seconds (oldest first).
        """
        with self._lock:
            segs = list(self._segments)

        if not segs:
            return []

        # Walk backwards from most recent until we have enough coverage
        total = 0.0
        result: list[SegmentInfo] = []
        for seg in reversed(segs):
            result.insert(0, seg)
            total += seg.duration
            if total >= seconds:
                break
        return result

    def latest_segment(self) -> Optional[SegmentInfo]:
        """Return the most recently completed segment, or None."""
        with self._lock:
            return self._segments[-1] if self._segments else None

    def segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    # ── Internal ────────────────────────────────────────────────────────────

    def _resolve_stream_url(self) -> Optional[list[str]]:
        """Use yt-dlp to get the direct video and optional audio stream URLs."""
        log.info("Resolving stream URL via yt-dlp…")
        cmd = [config.YTDLP_BIN]
        if "youtube.com" in self.stream_url.lower() or "youtu.be" in self.stream_url.lower():
            cmd.extend(["--js-runtimes", "deno"])
        cmd.extend([
            "--get-url",
            "-f", config.STREAM_QUALITY,
            "--no-playlist",
            self.stream_url,
        ])
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                log.error("yt-dlp --get-url failed:\n%s", result.stderr)
                return None
            urls = result.stdout.strip().splitlines()
            if not urls:
                log.error("yt-dlp returned no stream URLs.")
                return None
            log.info("Resolved %d stream URL(s).", len(urls))
            return urls
        except FileNotFoundError:
            log.error("yt-dlp not found. Install it with: pip install yt-dlp")
            return None
        except subprocess.TimeoutExpired:
            log.error("yt-dlp timed out resolving the stream URL.")
            return None

    def _run(self) -> None:
        """Main recorder loop — resolves URL then runs ffmpeg."""
        direct_urls = self._resolve_stream_url()
        if not direct_urls:
            log.error("Cannot start recorder: stream URL resolution failed.")
            return

        seg_pattern = str(config.TMP_DIR / f"{self._segment_prefix}_%05d.ts")

        # ffmpeg command: read HLS/DASH live stream → write 30 s segments
        cmd = [
            config.FFMPEG_BIN,
            "-hide_banner",
            "-loglevel", "warning",
        ]
        for url in direct_urls:
            cmd.extend(["-re", "-i", url])
        if len(direct_urls) > 1:
            cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
        cmd.extend([
            "-c", "copy",                        # no re-encode = fast
            "-f", "segment",
            "-segment_time", str(config.SEGMENT_DURATION),
            "-segment_format", "mpegts",
            "-reset_timestamps", "1",
            "-strftime", "0",                    # use numeric index
            seg_pattern,
        ])

        log.info("Launching ffmpeg segmenter…")
        self._ffmpeg_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Start a thread that watches for new segment files
        self._watcher_thread = threading.Thread(
            target=self._watch_segments, daemon=True, name="SegWatcher"
        )
        self._watcher_thread.start()

        # Monitor ffmpeg stderr for warnings
        try:
            assert self._ffmpeg_proc.stderr is not None
            for line in self._ffmpeg_proc.stderr:
                if self._stop_event.is_set():
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip()
                if decoded:
                    log.debug("[ffmpeg] %s", decoded)
        except Exception as exc:
            log.warning("ffmpeg stderr reader error: %s", exc)

        self._ffmpeg_proc.wait()
        log.info("ffmpeg exited with code %d.", self._ffmpeg_proc.returncode)

    def _watch_segments(self) -> None:
        """
        Poll tmp/ for new .ts segment files. When a new file appears AND
        the *next* segment has started (meaning the previous one is complete),
        register it in the deque.
        """
        seen: dict[int, float] = {}   # index → wall_start

        while not self._stop_event.is_set():
            time.sleep(1)
            try:
                ts_files = sorted(config.TMP_DIR.glob(f"{self._segment_prefix}_*.ts"))
            except Exception:
                continue

            for f in ts_files:
                try:
                    idx = int(f.stem.rsplit("_", 1)[1])
                except (IndexError, ValueError):
                    continue

                if idx not in seen:
                    seen[idx] = time.time() - config.SEGMENT_DURATION
                    log.debug("New segment detected: %s", f.name)

            # A segment is "complete" once the next one has appeared
            # (ffmpeg starts writing seg_N+1 when seg_N is sealed)
            complete_indices = []
            idxs = sorted(seen.keys())
            for i, idx in enumerate(idxs[:-1]):  # all but the last (in-progress)
                complete_indices.append(idx)

            for idx in complete_indices:
                if idx not in [s.index for s in self._segments]:
                    path = config.TMP_DIR / f"{self._segment_prefix}_{idx:05d}.ts"
                    if path.exists():
                        wall_start = seen[idx]
                        seg = SegmentInfo(
                            path=path,
                            index=idx,
                            wall_start=wall_start,
                            duration=float(config.SEGMENT_DURATION),
                        )
                        with self._lock:
                            self._segments.append(seg)
                            # Prune old segments beyond buffer limit
                            while len(self._segments) > config.SEGMENT_BUFFER_KEEP:
                                old = self._segments.popleft()
                                try:
                                    old.path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                        log.info(
                            "Registered segment #%05d (wall %.1f)",
                            idx, wall_start,
                        )
