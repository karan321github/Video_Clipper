"""
hype_detector.py — GTAClipper Hype Detector
=============================================
Runs every ANALYSIS_INTERVAL seconds (default 10 s).
On each tick:
  1. Count chat messages in the last 10 s window.
  2. Measure audio RMS loudness of the latest .ts segment using ffprobe.
  3. Compare both against their rolling baselines (last 60 s / 6 windows).
  4. If either signal exceeds its spike threshold, fire the clip callback.
  5. Apply a cooldown to prevent duplicate clips.

The detector is intentionally decoupled from ClipMaker — it fires a
callback (`on_spike`) with the wall-clock time of the spike. The
orchestrator wires up the callback.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from typing import Callable, Optional

import config
from chat_reader import ChatReader
from recorder import StreamRecorder

log = logging.getLogger(__name__)

# Callback type: on_spike(spike_wall_time: float)
SpikeCallback = Callable[[float], None]


class HypeDetector:
    """
    Periodically measures chat activity and audio loudness, fires
    `on_spike` when both exceed their rolling-average thresholds.
    """

    def __init__(
        self,
        chat_reader: ChatReader,
        recorder: StreamRecorder,
        on_spike: SpikeCallback,
    ):
        self.chat_reader = chat_reader
        self.recorder    = recorder
        self.on_spike    = on_spike

        # Rolling history (one entry per analysis tick)
        self._chat_history:     deque[float] = deque(maxlen=config.ROLLING_WINDOW_SIZE)
        self._loudness_history: deque[float] = deque(maxlen=config.ROLLING_WINDOW_SIZE)

        self._last_clip_time: float = 0.0
        self._last_measured_segment = None   # segment behind last_rms_db
        self._stop_event = threading.Event()
        self._timer: Optional[threading.Timer] = None

        # Expose current readings for the status console
        self.last_chat_count:  int   = 0
        self.last_rms_db:      float = -99.0
        self.last_chat_avg:    float = 0.0
        self.last_rms_avg:     float = -99.0
        self.spike_count:      int   = 0

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        log.info("HypeDetector started (interval=%.1fs).", config.ANALYSIS_INTERVAL)
        self._schedule()

    def stop(self) -> None:
        self._stop_event.set()
        if self._timer:
            self._timer.cancel()
        log.info("HypeDetector stopped.")

    # ── Internal ────────────────────────────────────────────────────────────

    def _schedule(self) -> None:
        if not self._stop_event.is_set():
            self._timer = threading.Timer(
                config.ANALYSIS_INTERVAL, self._tick
            )
            self._timer.daemon = True
            self._timer.start()

    def _tick(self) -> None:
        """One analysis cycle."""
        try:
            self._analyse()
        except Exception as exc:
            log.exception("HypeDetector tick error: %s", exc)
        finally:
            self._schedule()   # re-arm

    def _analyse(self) -> None:
        spike_time = time.time()

        # 1. Chat count in the last ANALYSIS_INTERVAL seconds
        recent_msgs = self.chat_reader.recent(config.ANALYSIS_INTERVAL)
        chat_count  = len(recent_msgs)
        self.last_chat_count = chat_count
        self._chat_history.append(float(chat_count))

        # 2. Audio loudness of the latest completed segment
        rms_db = self._measure_loudness()
        self.last_rms_db = rms_db
        if rms_db > -90.0:   # only record if we got a real reading
            self._loudness_history.append(rms_db)

        # 3. Rolling averages
        chat_avg = (
            sum(self._chat_history) / len(self._chat_history)
            if self._chat_history else 0.0
        )
        rms_avg = (
            sum(self._loudness_history) / len(self._loudness_history)
            if self._loudness_history else -99.0
        )
        self.last_chat_avg = chat_avg
        self.last_rms_avg  = rms_avg

        log.debug(
            "Tick — chat: %d (avg %.1f) | rms: %.1f dB (avg %.1f dB)",
            chat_count, chat_avg, rms_db, rms_avg,
        )

        # 4. Spike detection
        chat_spiked = (
            chat_count >= config.CHAT_SPIKE_MIN_MESSAGES
            and chat_avg > 0
            and chat_count >= chat_avg * config.CHAT_SPIKE_MULTIPLIER
        )
        rms_spiked = (
            rms_db > -90.0
            and rms_avg > -90.0
            and rms_db >= rms_avg + config.LOUDNESS_SPIKE_DB
        )

        if chat_spiked or rms_spiked:
            if rms_spiked and not chat_spiked and self._last_measured_segment is not None:
                # The loud audio is in the measured segment, not at "now".
                seg = self._last_measured_segment
                spike_time = seg.wall_start + seg.duration / 2
            # 5. Cooldown check
            since_last = spike_time - self._last_clip_time
            if since_last >= config.CLIP_COOLDOWN_SECONDS:
                signals = []
                if chat_spiked:
                    signals.append("chat %d (%.1f×avg)" % (
                        chat_count, chat_count / max(chat_avg, 1)
                    ))
                if rms_spiked:
                    signals.append("rms %.1f dB (+%.1f dB)" % (
                        rms_db, rms_db - rms_avg
                    ))
                log.info(
                    "🔥 SPIKE DETECTED — %s",
                    " | ".join(signals),
                )
                self._last_clip_time = spike_time
                self.spike_count += 1
                try:
                    self.on_spike(spike_time)
                except Exception as exc:
                    log.error("on_spike callback error: %s", exc)
            else:
                log.debug(
                    "Spike detected but cooldown active (%.0f s remaining).",
                    config.CLIP_COOLDOWN_SECONDS - since_last,
                )

    def _measure_loudness(self) -> float:
        """
        Run ffprobe astats on the latest completed .ts segment.
        Returns the RMS dB level, or -99.0 on failure.
        """
        seg = self.recorder.latest_segment()
        self._last_measured_segment = None
        if seg is None or not seg.path.exists():
            return -99.0
        self._last_measured_segment = seg

        cmd = [
            config.FFPROBE_BIN,
            "-v", "error",
            "-f", "lavfi",
            f"-i", f"amovie={seg.path.as_posix()},astats=metadata=1:reset=1",
            "-show_entries", "frame_tags=lavfi.astats.Overall.RMS_level",
            "-of", "csv=p=0",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode != 0:
                return self._ffmpeg_astats_fallback(seg.path)

            lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
            if not lines:
                return self._ffmpeg_astats_fallback(seg.path)

            # Average over all frames returned
            values = []
            for line in lines:
                try:
                    v = float(line)
                    if v > -200:   # filter -inf
                        values.append(v)
                except ValueError:
                    pass
            if values:
                return sum(values) / len(values)
            return self._ffmpeg_astats_fallback(seg.path)

        except subprocess.TimeoutExpired:
            log.warning("ffprobe loudness measurement timed out.")
            return -99.0
        except FileNotFoundError:
            log.error("ffprobe not found. Ensure ffmpeg is installed and on PATH.")
            return -99.0

    def _ffmpeg_astats_fallback(self, seg_path) -> float:
        """
        Fallback: use `ffmpeg -af astats` to measure RMS dB on the segment.
        Parses the stderr output for 'RMS level dB' lines.
        """
        cmd = [
            config.FFMPEG_BIN,
            "-hide_banner",
            "-i", str(seg_path),
            "-af", "astats",
            "-f", "null",
            "-",
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=20
            )
            # Look for lines like: "RMS level dB:          -18.3"
            import re
            matches = re.findall(r"RMS level dB:\s+([\-\d\.]+|inf)", result.stderr)
            values = []
            for m in matches:
                try:
                    v = float(m)
                    if v > -200:
                        values.append(v)
                except ValueError:
                    pass
            if values:
                return sum(values) / len(values)
        except Exception as exc:
            log.debug("astats fallback failed: %s", exc)
        return -99.0
