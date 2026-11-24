"""
clip_maker.py — GTAClipper Clip Maker
=======================================
Renders clips from either:

  • a local source file (regular videos / livestream replays / manual
    timestamps): `submit_source_clip()` / `render_source_clip()`
  • the live recorder's rolling .ts segments (live streams): `on_spike()`

Both paths share one FFmpeg render pipeline:
  1. Validate the requested [start, end] range against the source duration.
  2. Check that the source has a video stream (audio is optional).
  3. Create a unique per-job temp directory (no collisions between jobs).
  4. Optionally transcribe the clip's audio with faster-whisper → SRT.
  5. Encode in a single ffmpeg pass using *input* seeking (-ss before -i)
     with a full re-encode, which gives frame-accurate cuts, keeps audio
     and video in sync, and works with any input codec (H.264, VP9, AV1,
     AAC, Opus, 10-bit…). Output: 1920×1080 H.264 (yuv420p) + AAC MP4.
  6. Verify the output exists, is non-empty, has the right streams and
     duration, and decodes without errors — only then publish it to
     clips/ with a unique filename.
  7. Clean up the job's temp directory.

Failures raise ClipError internally and are reported as a ClipResult with
the stage and a user-readable message; failed clips are never counted as
successful.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import config
from chat_reader import ChatReader
from media_tools import MediaInfo, format_cmd, probe_media, run_ffmpeg
from recorder import SegmentInfo, StreamRecorder
from timestamps import (
    ClipRange,
    TimestampError,
    filename_timestamp,
    format_timestamp,
    validate_range,
)
from title_generator import generate_title

log = logging.getLogger(__name__)

MAX_CONCURRENT_CLIPS = 2    # max parallel clip-making threads

# User-facing message prefix for each pipeline stage.
STAGE_MESSAGES = {
    "validate": "Invalid clip timestamps",
    "source": "Unable to read source media",
    "streams": "Unsupported source media",
    "segments": "Recorded footage unavailable",
    "disk": "Insufficient disk space",
    "encode": "FFmpeg processing failed",
    "timeout": "FFmpeg processing timed out",
    "cancelled": "Clip generation cancelled",
    "verify": "Generated video failed validation",
    "publish": "Could not save the generated clip",
    "internal": "Unexpected error",
}


class ClipError(Exception):
    """A clip failed at a specific pipeline stage."""

    def __init__(self, stage: str, message: str, detail: str = ""):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.detail = detail

    def user_message(self) -> str:
        prefix = STAGE_MESSAGES.get(self.stage, "Clip failed")
        return f"{prefix}: {self.message}"


@dataclass
class ClipResult:
    index: int
    start: float
    end: float
    hype_time: Optional[float] = None
    status: str = "pending"            # pending | running | ok | failed
    path: Optional[Path] = None
    error: str = ""
    stage: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def label(self) -> str:
        return f"{format_timestamp(self.start)}-{format_timestamp(self.end)}"


ProgressCallback = Callable[[ClipResult, str, float], None]


def captions_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


class ClipMaker:
    """
    Receives spike events / clip requests and produces horizontal 16:9 MP4
    clips with optional burned-in captions and a title overlay.
    """

    def __init__(
        self,
        recorder: Optional[StreamRecorder],
        chat_reader: Optional[ChatReader],
        captions: bool = True,
        max_workers: int = MAX_CONCURRENT_CLIPS,
    ):
        self.recorder    = recorder
        self.chat_reader = chat_reader
        self._executor   = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="ClipMaker"
        )
        self._whisper_lock = threading.Lock()   # one transcription at a time
        self._whisper_model = None              # lazy-loaded
        self._captions = captions and captions_available()
        if captions and not self._captions:
            log.warning("faster-whisper is not installed — clips will have no captions.")

        self._stats_lock = threading.Lock()
        self._media_cache: dict[Path, MediaInfo] = {}
        self._media_lock = threading.Lock()
        self._stop_event = threading.Event()     # cancels running ffmpeg jobs
        self._live_stop = threading.Event()      # live: stop waiting for footage
        self._live_counter = 0

        # Stats
        self.clips_made: int = 0
        self.clips_failed: int = 0

        config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
        config.TMP_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def on_spike(self, spike_time: float) -> None:
        """Called by HypeDetector (live mode). Submits a clip job."""
        recent_chat = self.chat_reader.recent(seconds=30.0) if self.chat_reader else []
        self._executor.submit(self._make_clip, spike_time, recent_chat)
        log.info("Clip job submitted for spike at %.1f.", spike_time)

    def submit_source_clip(
        self,
        source: Path,
        clip_range: ClipRange,
        *,
        index: int = 1,
        title: str = "",
        video_label: str = "",
        min_duration: float = config.MANUAL_CLIP_MIN_DURATION,
        progress: Optional[ProgressCallback] = None,
    ) -> "Future[ClipResult]":
        """Queue a clip of `source` between clip_range.start and clip_range.end."""
        return self._executor.submit(
            self.render_source_clip,
            source,
            clip_range,
            index=index,
            title=title,
            video_label=video_label,
            min_duration=min_duration,
            progress=progress,
        )

    def render_source_clip(
        self,
        source: Path,
        clip_range: ClipRange,
        *,
        index: int = 1,
        title: str = "",
        video_label: str = "",
        min_duration: float = config.MANUAL_CLIP_MIN_DURATION,
        progress: Optional[ProgressCallback] = None,
    ) -> ClipResult:
        """Render one clip from a local source file (blocking)."""
        result = ClipResult(
            index=index,
            start=clip_range.start,
            end=clip_range.end,
            hype_time=clip_range.hype_time,
            status="running",
        )
        media: Optional[MediaInfo] = None
        try:
            media = self._source_media(source)
            try:
                validated = validate_range(
                    clip_range.start, clip_range.end, media.duration, min_duration
                )
            except TimestampError as exc:
                raise ClipError("validate", str(exc)) from None
            result.start, result.end = validated.start, validated.end

            video_id = _safe_filename(video_label or source.stem)
            out_name = (
                f"clip_{video_id}_{filename_timestamp(result.start)}-"
                f"{filename_timestamp(result.end)}_{uuid.uuid4().hex[:8]}.mp4"
            )
            result.path = self._render(
                source=source,
                input_args=(),
                start=result.start,
                duration=result.end - result.start,
                has_audio=media.has_audio,
                source_size=(media.width, media.height),
                title=title or "Hype Moment",
                out_path=config.CLIPS_DIR / out_name,
                result=result,
                progress=progress,
            )
            result.status = "ok"
            self._count(ok=True)
            log.info("✅ Clip %d saved: %s", index, result.path)
        except ClipError as exc:
            self._fail(result, exc, video_label, media)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("Clip %d crashed", index)
            self._fail(result, ClipError("internal", str(exc)), video_label, media)
        if progress:
            progress(result, result.status, 1.0)
        return result

    def shutdown(self, cancel: bool = False) -> None:
        """
        Wait for queued clips. Live clips still waiting for footage are
        rendered with what has been recorded. With cancel=True, pending jobs
        are dropped and running ffmpeg processes are killed.
        """
        if cancel:
            self._stop_event.set()
        self._live_stop.set()
        self._executor.shutdown(wait=True, cancel_futures=cancel)
        log.info("ClipMaker: all jobs finished.")

    # ── Whisper model (lazy, shared) ────────────────────────────────────────

    def _get_whisper_model(self):
        if self._whisper_model is None:
            with self._whisper_lock:
                if self._whisper_model is None:
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
        return self._whisper_model

    # ── Live clip pipeline ──────────────────────────────────────────────────

    def _make_clip(self, spike_time: float, recent_chat) -> None:
        with self._stats_lock:
            self._live_counter += 1
            index = self._live_counter
        clip_start = spike_time - config.CLIP_LOOKBACK
        clip_end = clip_start + config.CLIP_DURATION
        result = ClipResult(index=index, start=clip_start, end=clip_end,
                            hype_time=spike_time, status="running")
        log.info("=== Making live clip %d (spike at %.1f) ===", index, spike_time)
        job_dir: Optional[Path] = None
        try:
            segments = self._wait_for_live_window(clip_start, clip_end)
            if not segments:
                raise ClipError("segments", "no recorded segments cover this moment.")

            win_start = max(clip_start, segments[0].wall_start)
            win_end = min(clip_end, segments[-1].wall_end)
            duration = win_end - win_start
            if duration < config.CLIP_MIN_DURATION:
                raise ClipError(
                    "segments",
                    f"only {duration:.1f} s of footage recorded around the spike "
                    f"(minimum {config.CLIP_MIN_DURATION} s).",
                )

            # Hard-link (or copy) the segments into the job dir so the
            # recorder can prune its buffer while we encode.
            job_dir = Path(tempfile.mkdtemp(prefix="live_", dir=config.TMP_DIR))
            local_names = []
            for seg in segments:
                local = job_dir / seg.path.name
                try:
                    os.link(seg.path, local)
                except OSError:
                    shutil.copy2(seg.path, local)
                local_names.append(local.name)
            concat_list = job_dir / "concat.txt"
            concat_list.write_text(
                "".join(f"file '{name}'\n" for name in local_names), encoding="utf-8"
            )

            title = generate_title(recent_chat)
            log.info("Clip %d title: %r", index, title)
            ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(spike_time))
            out_path = config.CLIPS_DIR / (
                f"clip_{ts_str}_{_safe_filename(title)}_{uuid.uuid4().hex[:8]}.mp4"
            )
            media = probe_media(job_dir / local_names[0])
            result.path = self._render(
                source=concat_list,
                input_args=("-f", "concat", "-safe", "0"),
                start=win_start - segments[0].wall_start,
                duration=duration,
                has_audio=bool(media and media.has_audio),
                source_size=(media.width, media.height) if media else None,
                title=title,
                out_path=out_path,
                result=result,
            )
            result.status = "ok"
            self._count(ok=True)
            log.info("✅ Clip saved: %s", result.path)
        except ClipError as exc:
            self._fail(result, exc, "live", None)
        except Exception as exc:
            log.exception("Clip %d failed: %s", index, exc)
            self._fail(result, ClipError("internal", str(exc)), "live", None)
        finally:
            if job_dir:
                shutil.rmtree(job_dir, ignore_errors=True)

    def _wait_for_live_window(self, clip_start: float, clip_end: float) -> list[SegmentInfo]:
        """
        Wait until the recorder has *completed* footage up to clip_end (the
        spike happens before most of the clip has been recorded), then
        return the segments overlapping [clip_start, clip_end].
        """
        recorder = self.recorder
        if recorder is None:
            return []
        get_between = getattr(recorder, "get_segments_between", None)
        if get_between is None:
            get_window = getattr(recorder, "get_segments_for_window", None)
            if get_window:
                return get_window(clip_start + config.CLIP_LOOKBACK, config.CLIP_DURATION)
            return recorder.get_segments_covering(config.CLIP_LOOKBACK + config.CLIP_DURATION)

        deadline = time.time() + max(0.0, clip_end - time.time()) + 2 * config.SEGMENT_DURATION + 30
        while True:
            latest = recorder.latest_segment()
            if latest is not None and latest.wall_end >= clip_end:
                break
            if self._live_stop.is_set() or self._stop_event.is_set():
                log.info("Stopping early; rendering the footage recorded so far.")
                break
            if time.time() > deadline:
                log.warning("Timed out waiting for footage up to %.1f.", clip_end)
                break
            self._live_stop.wait(2.0)
        return get_between(clip_start, clip_end)

    # ── Shared render pipeline ──────────────────────────────────────────────

    def _render(
        self,
        *,
        source: Path,
        input_args: Sequence[str],
        start: float,
        duration: float,
        has_audio: bool,
        title: str,
        out_path: Path,
        result: ClipResult,
        progress: Optional[ProgressCallback] = None,
        source_size: Optional[tuple[int, int]] = None,
    ) -> Path:
        _ensure_disk_space(config.CLIPS_DIR)
        job_dir = Path(tempfile.mkdtemp(prefix=f"clip{result.index}_", dir=config.TMP_DIR))
        try:
            srt_path = job_dir / "subs.srt"
            _write_empty_srt(srt_path)
            if self._captions and has_audio:
                if progress:
                    progress(result, "transcribing", 0.0)
                wav = job_dir / "audio.wav"
                if self._extract_audio(source, input_args, start, duration, wav):
                    self._transcribe(wav, srt_path)

            def _on_progress(fraction: float) -> None:
                if progress:
                    progress(result, "encoding", fraction)

            self._encode_final(
                source, srt_path, title, out_path,
                input_args=input_args, start=start, duration=duration,
                has_audio=has_audio, on_progress=_on_progress,
                source_size=source_size,
            )
            return out_path
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    def _extract_audio(
        self, source: Path, input_args: Sequence[str], start: float, duration: float, wav: Path
    ) -> bool:
        cmd = [
            config.FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            *input_args, "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", str(wav),
        ]
        res = run_ffmpeg(cmd, timeout=max(120, duration * 2), cancel_event=self._stop_event)
        if res.returncode != 0 or not wav.exists():
            log.warning("Audio extraction for captions failed (exit %s): %s",
                        res.returncode, res.last_error_line())
            return False
        return True

    def _transcribe(self, audio_path: Path, srt_path: Path) -> None:
        """Run faster-whisper and write an SRT file (empty on failure)."""
        try:
            model = self._get_whisper_model()
            with self._whisper_lock:
                segments, _info = model.transcribe(
                    str(audio_path),
                    language=config.WHISPER_LANGUAGE,
                    vad_filter=True,
                    word_timestamps=False,
                    beam_size=5,
                )
                seg_list = list(segments)   # consume generator while lock held
            _write_srt(seg_list, srt_path)
            log.info("Transcribed %d segments to %s.", len(seg_list), srt_path.name)
        except Exception as exc:
            log.warning("Transcription failed: %s. Clip will have no captions.", exc)
            _write_empty_srt(srt_path)

    def _encode_final(
        self,
        raw_clip: Path,
        srt_path: Path,
        title: str,
        out_path: Path,
        cam_clip: Optional[Path] = None,
        *,
        input_args: Sequence[str] = (),
        start: float = 0.0,
        duration: Optional[float] = None,
        has_audio: bool = True,
        on_progress: Optional[Callable[[float], None]] = None,
        source_size: Optional[tuple[int, int]] = None,
    ) -> bool:
        """
        Single ffmpeg pass producing a verified 16:9 1920×1080 H.264/AAC MP4.

        `raw_clip` is the input (a media file or a concat list when
        input_args selects the concat demuxer); `start`/`duration` select
        the range using accurate input seeking. Subtitle and title files
        are referenced relative to their job directory (ffmpeg runs there),
        so no filtergraph path/text escaping is needed.

        Returns True on success; raises ClipError otherwise.
        """
        w, h = config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT
        work_dir = srt_path.parent
        if duration is None:
            media = probe_media(raw_clip)
            duration = max(0.0, (media.duration if media else 0.0) - start)
        if duration <= 0:
            raise ClipError("validate", "clip duration must be greater than zero.")

        has_cam = bool(
            config.INCLUDE_STREAMER_CAM.lower() != "off"
            and cam_clip and cam_clip.exists() and cam_clip.stat().st_size > 1000
        )

        # ── Background / fit ──────────────────────────────────────────────
        if source_size is None:
            source_size = _probe_video_dimensions(raw_clip)
        src_w, src_h = source_size
        same_aspect = src_w > 0 and src_h > 0 and abs(src_w / src_h - w / h) < 0.01
        if same_aspect:
            # Already 16:9 — the background would be fully covered anyway.
            filter_parts = [f"[0:v]scale={w}:{h}[base]"]
        elif config.OUTPUT_FILL_MODE.lower() == "blur":
            # Blur at quarter resolution (~20× faster, visually identical).
            bw, bh = max(2, w // 4 // 2 * 2), max(2, h // 4 // 2 * 2)
            filter_parts = [
                f"[0:v]split=2[bgsrc][fgsrc]",
                f"[bgsrc]scale={bw}:{bh}:force_original_aspect_ratio=increase,"
                f"crop={bw}:{bh},boxblur=5:2,scale={w}:{h}[bg]",
                f"[fgsrc]scale={w}:{h}:force_original_aspect_ratio=decrease[fg]",
                "[bg][fg]overlay=(W-w)/2:(H-h)/2[base]",
            ]
        else:
            filter_parts = [
                f"[0:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black[base]"
            ]
        final_label = "base"

        if has_cam:
            cam_w = int(w * config.CAM_OVERLAY_WIDTH_FRAC)
            margin = config.CAM_OVERLAY_MARGIN
            filter_parts.append(f"[1:v]scale={cam_w}:-2[cam]")
            filter_parts.append(f"[base][cam]overlay=W-w-{margin}:H-h-{margin}[composited]")
            final_label = "composited"

        # ── Captions + title ──────────────────────────────────────────────
        video_filters = []
        if srt_path.exists() and srt_path.stat().st_size > 10:
            video_filters.append(
                f"subtitles={srt_path.name}"
                f":force_style='FontSize={config.CAPTION_FONT_SIZE},"
                f"PrimaryColour=&Hffffff,OutlineColour=&H000000,"
                f"BorderStyle=3,Outline=2,Alignment=2'"
            )
        title = (title or "").strip()[: config.TITLE_MAX_CHARS]
        if title:
            title_file = work_dir / f"title_{uuid.uuid4().hex[:6]}.txt"
            title_file.write_text(title, encoding="utf-8")
            video_filters.append(
                f"drawtext=textfile={title_file.name}:expansion=none"
                f":fontsize={config.TITLE_FONT_SIZE}"
                f":fontcolor={config.TITLE_FONT_COLOR}"
                f":box=1:boxcolor={config.TITLE_BG_COLOR}"
                f":boxborderw={config.CAPTION_BOX_BORDER}"
                f":x=(w-text_w)/2:y=40"
            )
        video_filters.append("format=yuv420p")   # 10-bit/HDR sources → widely playable
        filter_parts.append(f"[{final_label}]{','.join(video_filters)}[vout]")
        filtergraph = ";".join(filter_parts)

        temp_out = out_path.with_name(f".{out_path.stem}.{uuid.uuid4().hex[:8]}.partial.mp4")
        seek = ["-ss", f"{start:.3f}"] if start > 0 else []
        cmd = [
            config.FFMPEG_BIN, "-hide_banner", "-loglevel", "warning", "-y",
            *input_args, *seek, "-t", f"{duration:.3f}", "-i", str(raw_clip),
        ]
        if has_cam:
            cmd += [*seek, "-t", f"{duration:.3f}", "-i", str(cam_clip)]
        cmd += ["-filter_complex", filtergraph, "-map", "[vout]"]
        if has_audio:
            cmd += [
                "-map", "0:a:0",
                "-af", "aresample=async=1:first_pts=0",
                "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
            ]
        else:
            cmd += ["-an"]
        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-max_muxing_queue_size", "4096",
            "-movflags", "+faststart",
            str(temp_out),
        ]

        # Long CPU encodes can take several minutes, especially in parallel.
        encode_timeout = max(900, duration * 10)
        log.debug("ffmpeg filtergraph: %s", filtergraph)
        try:
            res = run_ffmpeg(
                cmd,
                timeout=encode_timeout,
                expected_duration=duration,
                on_progress=on_progress,
                cancel_event=self._stop_event,
                cwd=work_dir,
            )
            if res.cancelled:
                raise ClipError("cancelled", "the job was cancelled.")
            if res.timed_out:
                raise ClipError(
                    "timeout",
                    f"ffmpeg did not finish within {encode_timeout:.0f} s.",
                    f"command: {format_cmd(cmd)}",
                )
            if res.returncode != 0:
                raise ClipError(
                    "encode",
                    f"exit code {res.signed_returncode}: {res.last_error_line() or 'no error output'}",
                    f"command: {format_cmd(cmd)}\nstderr:\n{res.stderr[-3000:]}",
                )
            reason = self._verify_output(temp_out, duration, has_audio)
            if reason:
                raise ClipError("verify", reason, f"file: {temp_out}")
            try:
                os.replace(temp_out, out_path)
            except OSError as exc:
                raise ClipError("publish", str(exc)) from None
            return True
        finally:
            try:
                temp_out.unlink(missing_ok=True)
            except OSError:
                log.warning("Could not remove incomplete encode %s.", temp_out)

    # ── Output verification ────────────────────────────────────────────────

    def _verify_output(
        self, path: Path, expected_duration: Optional[float] = None, expect_audio: bool = False
    ) -> Optional[str]:
        """Return None if `path` is a playable MP4, else a reason string."""
        if not path.exists() or path.stat().st_size < 1000:
            return "output file is missing or empty."

        cmd = [
            config.FFPROBE_BIN, "-v", "error",
            "-show_entries", "format=format_name,duration",
            "-show_entries", "stream=codec_type,codec_name",
            "-of", "json", str(path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except FileNotFoundError:
            return "ffprobe not found; cannot validate output."
        except subprocess.TimeoutExpired:
            return "timed out inspecting the output."
        if result.returncode != 0:
            return f"ffprobe could not read the output ({result.stderr.strip()[-300:]})."
        try:
            metadata = json.loads(result.stdout)
            duration = float(metadata["format"]["duration"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return "ffprobe returned invalid metadata."

        streams = metadata.get("streams", [])
        if "mp4" not in metadata.get("format", {}).get("format_name", "").split(","):
            return "output is not an MP4 file."
        if not any(s.get("codec_type") == "video" and s.get("codec_name") == "h264" for s in streams):
            return "output has no H.264 video stream."
        if expect_audio and not any(s.get("codec_type") == "audio" for s in streams):
            return "output has no audio stream."
        if expected_duration and abs(duration - expected_duration) > config.CLIP_DURATION_TOLERANCE:
            return (
                f"output is {duration:.1f} s long but {expected_duration:.1f} s was requested "
                "(source may be truncated)."
            )

        # Full decode test: catches corrupt frames that ffprobe accepts.
        decode = [config.FFMPEG_BIN, "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"]
        try:
            dres = subprocess.run(
                decode, capture_output=True, text=True,
                timeout=max(120, (expected_duration or duration) * 3),
            )
        except subprocess.TimeoutExpired:
            return "timed out while test-decoding the output."
        if dres.returncode != 0:
            return f"output could not be decoded: {dres.stderr.strip()[-300:]}"
        return None

    def _validate_encoded_clip(
        self, path: Path, expected_duration: Optional[float] = None, expect_audio: bool = False
    ) -> bool:
        reason = self._verify_output(path, expected_duration, expect_audio)
        if reason:
            log.error("Encoded clip validation failed for %s: %s", path, reason)
        return reason is None

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _source_media(self, source: Path) -> MediaInfo:
        with self._media_lock:
            cached = self._media_cache.get(source)
        if cached:
            return cached
        if not source.exists():
            raise ClipError("source", f"file not found: {source}")
        media = probe_media(source)
        if media is None:
            raise ClipError("source", f"ffprobe could not read {source.name}.")
        if not media.has_video:
            raise ClipError("streams", "no video stream found in the source.")
        if media.duration <= 0:
            raise ClipError("source", "could not determine the source duration.")
        if not media.has_audio:
            log.warning("Source %s has no audio stream; clips will be silent.", source.name)
        with self._media_lock:
            self._media_cache[source] = media
        return media

    def _count(self, ok: bool) -> None:
        with self._stats_lock:
            if ok:
                self.clips_made += 1
            else:
                self.clips_failed += 1

    def _fail(
        self, result: ClipResult, exc: ClipError, video_label: str, media: Optional[MediaInfo]
    ) -> None:
        result.status = "failed"
        result.stage = exc.stage
        result.error = exc.user_message()
        result.path = None
        self._count(ok=False)
        log.error(
            "❌ Clip %d failed at stage '%s': %s | video=%s | video_duration=%s | "
            "hype_time=%s | requested=%s-%s",
            result.index, exc.stage, exc.message, video_label or "?",
            f"{media.duration:.2f}s" if media else "?",
            f"{result.hype_time:.2f}s" if result.hype_time is not None else "n/a",
            f"{result.start:.3f}", f"{result.end:.3f}",
        )
        if exc.detail:
            log.error("Clip %d failure details:\n%s", result.index, exc.detail)


# ---------------------------------------------------------------------------
# Probe helpers (kept for backwards compatibility)
# ---------------------------------------------------------------------------

def _probe_video_dimensions(video_path: Path) -> tuple[int, int]:
    media = probe_media(video_path)
    return (media.width, media.height) if media else (0, 0)


def _probe_media_duration(video_path: Path) -> float:
    media = probe_media(video_path)
    return media.duration if media else 0.0


def _ensure_disk_space(directory: Path) -> None:
    try:
        free_mb = shutil.disk_usage(directory).free / (1024 * 1024)
    except OSError:
        return
    if free_mb < config.MIN_FREE_DISK_MB:
        raise ClipError(
            "disk",
            f"only {free_mb:.0f} MB free in {directory} (need {config.MIN_FREE_DISK_MB} MB).",
        )


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
