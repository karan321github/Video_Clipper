"""
clip_maker.py — GTAClipper Clip Maker
=======================================
Triggered by HypeDetector when a hype spike is detected.
Pipeline for each clip:
  1. Gather the .ts segments that cover [spike_time - CLIP_LOOKBACK,
     spike_time - CLIP_LOOKBACK + CLIP_DURATION].
  2. Concatenate segments into a single temp raw clip (ffmpeg concat demuxer).
  3. Render 16:9 horizontal (1920×1080) output:
       • If source is already 16:9: scale directly.
       • Otherwise: scale source to fit, fill empty bars with a blurred
         & scaled copy (OUTPUT_FILL_MODE="blur") or black bars.
       • If a streamer cam overlay is provided, composite it over the
         bottom-right corner of the frame.
  4. Transcribe audio with faster-whisper → write SRT subtitle file.
  5. Burn subtitles and title drawtext overlay in a single ffmpeg pass.
  6. Save final clip to clips/<timestamp>_<title>.mp4.
  7. Clean up temp files.

Up to MAX_CONCURRENT_CLIPS clips are processed simultaneously via a
thread pool so that back-to-back spikes don't block each other.
"""

from __future__ import annotations

import logging
import json
import os
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import config
from recorder import SegmentInfo, StreamRecorder
from chat_reader import ChatReader
from title_generator import generate_title

log = logging.getLogger(__name__)

MAX_CONCURRENT_CLIPS = 2    # max parallel clip-making threads


class ClipMaker:
    """
    Receives spike events and produces horizontal 16:9 MP4 clips with
    burned-in captions and a title overlay.
    """

    def __init__(self, recorder: StreamRecorder, chat_reader: ChatReader):
        self.recorder    = recorder
        self.chat_reader = chat_reader
        self._executor   = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_CLIPS, thread_name_prefix="ClipMaker"
        )
        self._whisper_lock = threading.Lock()   # one transcription at a time
        self._whisper_model = None              # lazy-loaded

        # Stats
        self.clips_made: int = 0
        self.clips_failed: int = 0

        # Ensure output directories exist
        config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        config.TMP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def on_spike(self, spike_time: float) -> None:
        """Called by HypeDetector. Submits a clip job to the thread pool."""
        # Snapshot the recent chat messages at spike time
        recent_chat = self.chat_reader.recent(seconds=30.0)
        self._executor.submit(self._make_clip, spike_time, recent_chat)
        log.info("Clip job submitted for spike at %.1f.", spike_time)

    def shutdown(self) -> None:
        """Wait for all in-progress clips to finish."""
        self._executor.shutdown(wait=True)
        log.info("ClipMaker: all jobs finished.")

    # ── Whisper model (lazy, shared) ────────────────────────────────────────

    def _get_whisper_model(self):
        if self._whisper_model is None:
            with self._whisper_lock:
                if self._whisper_model is None:
                    try:
                        from faster_whisper import WhisperModel
                        log.info(
                            "Loading faster-whisper model '%s' on %s…",
                            config.WHISPER_MODEL, config.WHISPER_DEVICE,
                        )
                        self._whisper_model = WhisperModel(
                            config.WHISPER_MODEL,
                            device=config.WHISPER_DEVICE,
                            compute_type=config.WHISPER_COMPUTE_TYPE,
                        )
                        log.info("Whisper model loaded.")
                    except ImportError:
                        log.error(
                            "faster-whisper not installed. "
                            "Run: pip install faster-whisper"
                        )
                        raise
        return self._whisper_model

    # ── Main clip pipeline ──────────────────────────────────────────────────

    def _make_clip(self, spike_time: float, recent_chat) -> None:
        clip_id = int(spike_time)
        log.info("=== Making clip %d ===", clip_id)

        try:
            # ── Step 1: gather segments ──────────────────────────────────
            needed_seconds = config.CLIP_LOOKBACK + config.CLIP_DURATION + 5
            get_window = getattr(self.recorder, "get_segments_for_window", None)
            if get_window:
                segments = get_window(spike_time, config.CLIP_DURATION)
            else:
                segments = self.recorder.get_segments_covering(needed_seconds)

            if not segments:
                log.warning("Clip %d: no segments available yet.", clip_id)
                self.clips_failed += 1
                return

            # ── Step 2: write concat list & extract raw clip ─────────────
            concat_list = config.TMP_DIR / f"concat_{clip_id}.txt"
            self._write_concat_list(concat_list, segments)

            raw_clip = config.TMP_DIR / f"raw_{clip_id}.mp4"
            if not self._extract_raw_clip(concat_list, raw_clip, segments, spike_time):
                self.clips_failed += 1
                return
            raw_duration = _probe_media_duration(raw_clip)
            if raw_duration + 2.0 < config.CLIP_DURATION:
                log.error(
                    "Clip %d extraction is only %.1f s; expected %d s. Skipping short output.",
                    clip_id,
                    raw_duration,
                    config.CLIP_DURATION,
                )
                self.clips_failed += 1
                return

            # ── Step 3: generate title from chat ────────────────────────
            title = generate_title(recent_chat)
            log.info("Clip %d title: %r", clip_id, title)

            # ── Step 4: transcribe with faster-whisper ───────────────────
            srt_path = config.TMP_DIR / f"subs_{clip_id}.srt"
            self._transcribe(raw_clip, srt_path)

            # ── Step 5: encode final horizontal 16:9 clip ────────────────
            safe_title = _safe_filename(title)
            ts_str     = time.strftime("%Y%m%d_%H%M%S", time.localtime(spike_time))
            job_suffix = uuid.uuid4().hex[:8]
            out_path   = config.CLIPS_DIR / f"clip_{ts_str}_{safe_title}_{job_suffix}.mp4"

            if not self._encode_final(raw_clip, srt_path, title, out_path):
                self.clips_failed += 1
                return

            self.clips_made += 1
            log.info("✅ Clip saved: %s", out_path)

        except Exception as exc:
            log.exception("Clip %d failed: %s", clip_id, exc)
            self.clips_failed += 1
        finally:
            self._cleanup(clip_id)

    # ── Step 2: concat & extract ────────────────────────────────────────────

    def _write_concat_list(self, path: Path, segments: list[SegmentInfo]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for seg in segments:
                # ffmpeg concat demuxer requires forward slashes
                f.write(f"file '{seg.path.as_posix()}'\n")

    def _extract_raw_clip(
        self,
        concat_list: Path,
        out_path: Path,
        segments: list[SegmentInfo],
        spike_time: float,
    ) -> bool:
        """
        Extracts CLIP_DURATION seconds starting CLIP_LOOKBACK seconds before
        the spike from the concatenated segments.
        """
        if not segments:
            return False

        # Calculate how far from the START of our segment buffer the clip begins
        earliest_wall = segments[0].wall_start
        total_duration = sum(s.duration for s in segments)
        if total_duration < config.CLIP_DURATION:
            log.warning(
                "Clip buffer is only %.1f s; waiting for at least %d s before making a clip.",
                total_duration,
                config.CLIP_DURATION,
            )
            return False
        clip_start_wall = spike_time - config.CLIP_LOOKBACK

        # Offset from beginning of concat buffer
        offset = max(0.0, clip_start_wall - earliest_wall)
        # Don't overshoot the buffer
        offset = min(offset, max(0.0, total_duration - config.CLIP_DURATION))

        log.debug(
            "Extract: offset=%.1f s, duration=%d s from %d segments",
            offset, config.CLIP_DURATION, len(segments),
        )

        cmd = [
            config.FFMPEG_BIN,
            "-hide_banner", "-loglevel", "warning",
            "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-ss", f"{offset:.2f}",
            "-t",  str(config.CLIP_DURATION),
            "-c", "copy",
            str(out_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                log.error("ffmpeg extract failed:\n%s", result.stderr)
                return False
            if not out_path.exists() or out_path.stat().st_size < 1000:
                log.error("ffmpeg extract produced empty file.")
                return False
            return True
        except subprocess.TimeoutExpired:
            log.error("ffmpeg extract timed out.")
            return False

    # ── Step 4: transcription ───────────────────────────────────────────────

    def _transcribe(self, video_path: Path, srt_path: Path) -> None:
        """
        Run faster-whisper on the raw clip and write an SRT file.
        Falls back to an empty SRT if anything fails.
        """
        try:
            model = self._get_whisper_model()
        except Exception:
            _write_empty_srt(srt_path)
            return

        try:
            with self._whisper_lock:
                segments, _info = model.transcribe(
                    str(video_path),
                    language=config.WHISPER_LANGUAGE,
                    vad_filter=True,
                    word_timestamps=False,
                    beam_size=5,
                )
                seg_list = list(segments)   # consume generator while lock held

            _write_srt(seg_list, srt_path)
            log.info("Transcribed %d segments to %s.", len(seg_list), srt_path.name)

        except Exception as exc:
            log.warning("Transcription failed: %s. Writing empty SRT.", exc)
            _write_empty_srt(srt_path)

    # ── Step 5: final encode ────────────────────────────────────────────────

    def _encode_final(
        self,
        raw_clip: Path,
        srt_path: Path,
        title: str,
        out_path: Path,
        cam_clip: Optional[Path] = None,
    ) -> bool:
        """
        Single ffmpeg pass producing a 16:9 horizontal clip (1920×1080):

        Layout (no cam):
          • Probe source dimensions.
          • If fill_mode="blur": [0:v] scaled + blurred background, then source
            scaled-to-fit centered on top → 1920×1080.
          • If fill_mode="black": scale with pad to 1920×1080 (black bars).

        Layout (with cam overlay):
          • Same background as above.
          • Streamer cam composited bottom-right, proportional to output width.

        Then:
          • Burn SRT subtitles.
          • Draw title text at top.
          • Encode H.264 / AAC.
        """
        w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT

        # ── Probe source video dimensions ─────────────────────────────────
        src_w, src_h = _probe_video_dimensions(raw_clip)
        if src_w <= 0 or src_h <= 0:
            # Fallback: assume 16:9
            src_w, src_h = 1920, 1080
            log.warning("Could not probe source dimensions; assuming 1920×1080.")

        # ── Build video filter chain ───────────────────────────────────────
        fill_mode = config.OUTPUT_FILL_MODE.lower()
        inputs = ["-i", str(raw_clip)]
        if (
            config.INCLUDE_STREAMER_CAM.lower() != "off"
            and cam_clip
            and cam_clip.exists()
            and cam_clip.stat().st_size > 1000
        ):
            inputs += ["-i", str(cam_clip)]
            has_cam = True
        else:
            has_cam = False

        if fill_mode == "blur":
            # Background: scale source to fill 1920×1080 (may crop edges), blur it
            bg = (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},"
                f"boxblur=20:5[bg]"
            )
            # Foreground: scale source to fit inside 1920×1080 (letterbox / pillarbox)
            fg = (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease[fg]"
            )
            # Overlay fg centered over bg
            overlay_base = (
                f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base]"
            )
        else:
            # Black bars
            bg = ""
            fg = ""
            overlay_base = (
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[base]"
            )

        # ── Cam overlay (bottom-right) ────────────────────────────────────
        if has_cam:
            cam_w = int(w * config.CAM_OVERLAY_WIDTH_FRAC)
            # Maintain cam aspect ratio by scaling width, height auto
            cam_scale = f"[1:v]scale={cam_w}:-1[cam]"
            margin    = config.CAM_OVERLAY_MARGIN
            cam_x     = f"W-w-{margin}"
            cam_y     = f"H-h-{margin}"
            cam_composite = f"[base][cam]overlay={cam_x}:{cam_y}[composited]"
            final_label   = "composited"
        else:
            cam_scale     = ""
            cam_composite = ""
            final_label   = "base"

        # ── Subtitle filter ───────────────────────────────────────────────
        srt_escaped = _escape_ffmpeg_path(str(srt_path))
        has_subs    = srt_path.exists() and srt_path.stat().st_size > 10
        sub_filter  = ""
        if has_subs:
            sub_filter = (
                f"subtitles='{srt_escaped}'"
                f":force_style='FontSize={config.CAPTION_FONT_SIZE},"
                f"PrimaryColour=&Hffffff,"
                f"OutlineColour=&H000000,"
                f"BorderStyle=3,"
                f"Outline=2,"
                f"Alignment=2'"
            )

        # ── Title drawtext ────────────────────────────────────────────────
        title_escaped = title.replace("'", "\\'").replace(":", "\\:")[:config.TITLE_MAX_CHARS]
        title_filter  = (
            f"drawtext=text='{title_escaped}'"
            f":fontsize={config.TITLE_FONT_SIZE}"
            f":fontcolor={config.TITLE_FONT_COLOR}"
            f":box=1:boxcolor={config.TITLE_BG_COLOR}"
            f":boxborderw={config.CAPTION_BOX_BORDER}"
            f":x=(w-text_w)/2"
            f":y=40"
        )

        # ── Assemble filtergraph ──────────────────────────────────────────
        if fill_mode == "blur":
            filter_parts = [bg, fg, overlay_base]
        else:
            filter_parts = [overlay_base]
        filter_parts.extend(part for part in (cam_scale, cam_composite) if part)
        video_filters = [part for part in (sub_filter, title_filter) if part]
        filter_parts.append(f"[{final_label}]{','.join(video_filters)}")
        filtergraph = ";".join(filter_parts)

        # A long CPU encode can take several minutes, especially when other
        # clipping processes are running.
        encode_timeout = max(900, config.CLIP_DURATION * 10)
        temp_out = out_path.with_name(
            f".{out_path.stem}.{uuid.uuid4().hex[:8]}.partial.mp4"
        )

        cmd = [
            config.FFMPEG_BIN,
            "-hide_banner", "-loglevel", "warning",
            "-y",
        ]
        cmd += inputs
        cmd += [
            "-filter_complex", filtergraph,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            str(temp_out),
        ]

        log.debug("ffmpeg filtergraph: %s", filtergraph)
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=encode_timeout)
            if result.returncode != 0:
                log.error("ffmpeg final encode failed:\n%s", result.stderr[-2000:])
                return False
            if not self._validate_encoded_clip(temp_out):
                log.error("ffmpeg output is incomplete or invalid; not saving %s.", out_path.name)
                return False
            os.replace(temp_out, out_path)
            return True
        except subprocess.TimeoutExpired:
            log.error(
                "ffmpeg final encode timed out after %d s; incomplete output discarded.",
                encode_timeout,
            )
            return False
        except OSError:
            log.exception("Could not publish encoded clip %s.", out_path)
            return False
        finally:
            try:
                temp_out.unlink(missing_ok=True)
            except OSError:
                log.warning("Could not remove incomplete encode %s.", temp_out)

    def _validate_encoded_clip(self, path: Path) -> bool:
        """Check that ffmpeg produced a playable, full-duration H.264 MP4."""
        if not path.exists() or path.stat().st_size < 1000:
            log.error("Encoded output is missing or too small: %s", path)
            return False

        cmd = [
            config.FFPROBE_BIN,
            "-v", "error",
            "-show_entries", "format=format_name,duration",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json",
            str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            log.error("ffprobe not found; cannot validate encoded clip.")
            return False
        except subprocess.TimeoutExpired:
            log.error("Timed out validating encoded clip %s.", path)
            return False

        if result.returncode != 0:
            log.error("ffprobe rejected encoded clip %s: %s", path, result.stderr.strip())
            return False
        try:
            metadata = json.loads(result.stdout)
            duration = float(metadata["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            log.error("ffprobe returned invalid clip metadata for %s.", path)
            return False

        format_name = metadata.get("format", {}).get("format_name", "")
        streams = metadata.get("streams", [])
        has_h264_video = any(
            stream.get("codec_type") == "video"
            and stream.get("codec_name") == "h264"
            for stream in streams
        )
        if (
            "mp4" not in format_name.split(",")
            or not has_h264_video
            or duration < config.CLIP_DURATION - 2
        ):
            log.error(
                "Encoded clip validation failed: format=%s, h264_video=%s, duration=%.1f s.",
                format_name,
                has_h264_video,
                duration,
            )
            return False
        return True

    # ── Cleanup ─────────────────────────────────────────────────────────────

    def _cleanup(self, clip_id: int) -> None:
        for pattern in (f"raw_{clip_id}.mp4", f"subs_{clip_id}.srt",
                        f"concat_{clip_id}.txt"):
            try:
                (config.TMP_DIR / pattern).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Video dimension probe helper
# ---------------------------------------------------------------------------

def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    """
    Use ffprobe to get the (width, height) of the first video stream.
    Returns (0, 0) on failure.
    """
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
            parts = line.split(",")
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
    except Exception as exc:
        log.debug("_probe_video_dimensions failed: %s", exc)
    return 0, 0


def _probe_media_duration(video_path: Path) -> float:
    """Return media duration in seconds, or zero when ffprobe cannot read it."""
    cmd = [
        config.FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        log.error("ffprobe not found while validating extracted clip duration.")
        return 0.0
    except subprocess.TimeoutExpired:
        log.error("ffprobe timed out while validating %s.", video_path)
        return 0.0
    if result.returncode != 0:
        log.error("ffprobe could not read extracted clip duration: %s", result.stderr.strip())
        return 0.0
    try:
        return float(result.stdout.strip())
    except ValueError:
        log.error("ffprobe returned an invalid duration for %s.", video_path)
        return 0.0


# ---------------------------------------------------------------------------
# SRT helpers
# ---------------------------------------------------------------------------

def _write_srt(segments, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = _srt_time(seg.start)
            end   = _srt_time(seg.end)
            text  = seg.text.strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def _write_empty_srt(path: Path) -> None:
    path.write_text("", encoding="utf-8")


def _srt_time(seconds: float) -> str:
    ms  = int((seconds % 1) * 1000)
    s   = int(seconds) % 60
    m   = (int(seconds) // 60) % 60
    h   = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Path / filename helpers
# ---------------------------------------------------------------------------

def _safe_filename(text: str) -> str:
    """Convert a title string to a filesystem-safe filename component."""
    safe = re.sub(r"[^\w\s\-]", "", text)
    safe = re.sub(r"\s+", "_", safe.strip())
    return safe[:40] or "clip"


def _escape_ffmpeg_path(path_str: str) -> str:
    """
    Escape a path for use inside an ffmpeg -vf filter value.
    On Windows: forward-slash, escape colons after drive letter.
    """
    path_str = path_str.replace("\\", "/")
    # Escape the colon in Windows drive letters (C:/ → C\\:/)
    path_str = re.sub(r"^([A-Za-z]):/", r"\1\\:/", path_str)
    return path_str
