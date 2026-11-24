"""
clipper.py — GTAClipper Main Orchestrator
==========================================
Entry point. Wires together Recorder → ChatReader → HypeDetector → ClipMaker.
Supports both live streams and regular (published) YouTube videos.
Displays a live status console in the terminal while the stream is live.

Usage
-----
  python clipper.py <url> [options]

Options
-------
  --platform   youtube | twitch   (default: auto-detect from URL)
  --channel    twitch channel name (required for Twitch)
  --model      faster-whisper model: tiny|base|small|medium (overrides config)
  --threshold  chat spike multiplier override (float, default from config)
  --vod-threshold  VOD audio spike threshold in dB above its baseline
  --max-clips  maximum number of clips to generate from a VOD (default: 20)
  --debug      enable DEBUG logging

Examples
--------
  # Live stream (YouTube)
  python clipper.py https://www.youtube.com/watch?v=LIVE_ID

  # Regular published YouTube video
  python clipper.py https://www.youtube.com/watch?v=VIDEO_ID

  # Short URL
  python clipper.py https://youtu.be/VIDEO_ID

  # YouTube Live URL
  python clipper.py https://www.youtube.com/live/VIDEO_ID

  # Twitch
  python clipper.py https://www.twitch.tv/xqc --platform twitch --channel xqc
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import config
from recorder      import StreamRecorder
from chat_reader   import make_chat_reader
from hype_detector import HypeDetector
from clip_maker    import ClipMaker, ClipResult, captions_available
from media_tools   import probe_media
from timestamps    import (
    ClipRange,
    TimestampError,
    build_hype_range,
    format_timestamp,
    parse_range,
    parse_timestamp,
    validate_range,
)
from video_processor import (
    probe_video_detailed,
    normalize_youtube_url,
    download_video,
    VODHypeDetector,
    VideoInfo,
    VideoSourceError,
)

# Force UTF-8 output on Windows so Unicode chars don't crash
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    # File handler (always DEBUG)
    fh = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(fmt, datefmt))

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt, datefmt))

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(fh)
    root.addHandler(ch)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="GTAClipper — Auto live stream & video clipper",
    )
    parser.add_argument(
        "url",
        help="YouTube URL (live stream, archived stream, or regular video) or Twitch URL.",
    )
    parser.add_argument(
        "--platform",
        choices=["youtube", "twitch"],
        default=None,
        help="Platform override. Auto-detected from URL if omitted.",
    )
    parser.add_argument(
        "--channel",
        default=None,
        help="Twitch channel name (required when --platform twitch).",
    )
    parser.add_argument(
        "--model",
        default=None,
        dest="whisper_model",
        help=f"faster-whisper model size (default: {config.WHISPER_MODEL})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        dest="chat_threshold",
        help=f"Chat spike multiplier (default: {config.CHAT_SPIKE_MULTIPLIER})",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=20,
        dest="max_clips",
        help="Maximum number of clips to generate from a VOD (default: 20).",
    )
    parser.add_argument(
        "--vod-threshold",
        type=float,
        default=None,
        help=(
            "VOD audio-spike threshold in dB above its rolling baseline "
            f"(default: {config.VOD_LOUDNESS_SPIKE_DB})."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    manual = parser.add_argument_group(
        "manual clipping (regular videos & replays)",
        "Clip exact timestamps instead of (or after) automatic hype detection. "
        "Timestamps accept HH:MM:SS, MM:SS or seconds.",
    )
    manual.add_argument("--start", help="Manual clip start, e.g. 00:02:15.")
    manual.add_argument("--end", help="Manual clip end, e.g. 00:02:55.")
    manual.add_argument(
        "--clip", action="append", default=[], metavar="START-END",
        help="Manual clip range, e.g. 00:02:15-00:02:55. Repeatable.",
    )
    manual.add_argument(
        "--duration", default=None,
        help=(
            "Clip length. Manual mode: used with --start when --end is omitted. "
            f"Automatic mode: length of each hype clip (default {config.CLIP_DURATION}s)."
        ),
    )
    manual.add_argument(
        "--padding-before", default=None,
        help=f"Automatic mode: time kept before each hype moment (default {config.CLIP_LOOKBACK}s).",
    )
    manual.add_argument(
        "--padding-after", default=None,
        help="Automatic mode: time kept after each hype moment (default: duration - padding-before).",
    )
    manual.add_argument(
        "--review", action="store_true",
        help="Automatic mode: review/edit the detected timestamps before clips are generated.",
    )
    manual.add_argument(
        "--detect-only", action="store_true",
        help="Automatic mode: only print detected clip ranges (re-use them with --clip).",
    )
    manual.add_argument(
        "--info", action="store_true",
        help="Only load and print the video information (type, duration).",
    )
    manual.add_argument(
        "--no-captions", action="store_true",
        help="Skip faster-whisper transcription.",
    )
    manual.add_argument(
        "--open", action="store_true",
        help="Open each finished clip in the default video player for preview.",
    )
    return parser.parse_args()


def _detect_platform(url: str) -> str:
    url_lower = url.lower()
    if "twitch.tv" in url_lower:
        return "twitch"
    if "youtube.com" in url_lower or "youtu.be" in url_lower:
        return "youtube"
    # Default
    return "youtube"


def _extract_twitch_channel(url: str) -> str | None:
    """Try to extract the channel name from a Twitch URL."""
    import re
    m = re.search(r"twitch\.tv/([A-Za-z0-9_]+)", url)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Status console (live stream mode)
# ---------------------------------------------------------------------------

CLEAR_LINE = "\r\033[K"
BOLD       = "\033[1m"
GREEN      = "\033[92m"
YELLOW     = "\033[93m"
RED        = "\033[91m"
CYAN       = "\033[96m"
RESET      = "\033[0m"


class StatusConsole:
    """
    Prints a one-line rolling status update every second.
    Works on both Windows (no ANSI) and Unix terminals.
    """

    _ANSI_SUPPORTED: bool = (
        sys.platform != "win32"
        or os.environ.get("WT_SESSION")     # Windows Terminal
        or os.environ.get("TERM_PROGRAM")
        or "ANSICON" in os.environ
    )

    def __init__(
        self,
        detector: HypeDetector,
        clip_maker: ClipMaker,
        chat_reader,
        platform: str,
    ):
        self._detector   = detector
        self._clip_maker = clip_maker
        self._chat_reader = chat_reader
        self._platform   = platform
        self._stop       = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = time.time()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="StatusConsole"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        print()  # newline after last status line

    def _run(self) -> None:
        while not self._stop.is_set():
            self._print_status()
            time.sleep(2)

    def _print_status(self) -> None:
        elapsed   = int(time.time() - self._start_time)
        h, rem    = divmod(elapsed, 3600)
        m, s      = divmod(rem, 60)
        uptime    = f"{h:02d}:{m:02d}:{s:02d}"

        d = self._detector
        cm = self._clip_maker

        chat_now  = d.last_chat_count
        chat_avg  = d.last_chat_avg
        rms_now   = d.last_rms_db
        rms_avg   = d.last_rms_avg
        clips     = cm.clips_made
        failed    = cm.clips_failed
        spikes    = d.spike_count
        total_msg = len(self._chat_reader.messages)

        # Compute spike indicators
        chat_spike = chat_avg > 0 and chat_now >= chat_avg * config.CHAT_SPIKE_MULTIPLIER
        rms_spike  = rms_avg > -90 and rms_now >= rms_avg + config.LOUDNESS_SPIKE_DB

        chat_indicator = "^" if chat_spike else " "
        rms_indicator  = "^" if rms_spike  else " "

        line = (
            f"[{uptime}] "
            f"Platform: {self._platform.upper()}  |  "
            f"Chat: {chat_now:3d} msg/10s {chat_indicator}(avg {chat_avg:.1f})  |  "
            f"Vol: {rms_now:5.1f} dBFS {rms_indicator}(avg {rms_avg:.1f})  |  "
            f"Clips: {clips}ok {failed}fail  |  "
            f"Spikes: {spikes}  |  "
            f"Chat msgs: {total_msg}"
        )

        try:
            if self._ANSI_SUPPORTED:
                print(f"{CLEAR_LINE}{line}", end="", flush=True)
            else:
                print(f"\r{line}", end="", flush=True)
        except UnicodeEncodeError:
            print(f"\r{line.encode('ascii', errors='replace').decode()}", end="", flush=True)


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_dependencies() -> bool:
    ok = True

    def _bin_exists(path: str) -> bool:
        """Works for both bare names on PATH and absolute paths."""
        import pathlib
        p = pathlib.Path(path)
        if p.is_absolute():
            return p.exists()
        return shutil.which(path) is not None

    # ffmpeg
    if not _bin_exists(config.FFMPEG_BIN):
        print(
            f"[ERROR] ffmpeg not found: {config.FFMPEG_BIN}\n"
            f"        Download from https://ffmpeg.org/download.html",
            file=sys.stderr,
        )
        ok = False

    # ffprobe
    if not _bin_exists(config.FFPROBE_BIN):
        print(
            f"[ERROR] ffprobe not found: {config.FFPROBE_BIN}",
            file=sys.stderr,
        )
        ok = False

    # yt-dlp
    if not _bin_exists(config.YTDLP_BIN):
        print(
            f"[ERROR] yt-dlp not found: {config.YTDLP_BIN}\n"
            f"        Install with: pip install yt-dlp",
            file=sys.stderr,
        )
        ok = False

    # faster-whisper (optional: only needed for burned-in captions)
    if not captions_available():
        print(
            "[WARN]  faster-whisper not installed — clips will have no captions.\n"
            "        Install with: pip install faster-whisper",
            file=sys.stderr,
        )

    return ok


# ---------------------------------------------------------------------------
# Live stream mode
# ---------------------------------------------------------------------------

def _run_live(args: argparse.Namespace, platform: str) -> int:
    """
    Original live-stream pipeline:
    StreamRecorder → ChatReader → HypeDetector → ClipMaker
    """
    twitch_channel = args.channel
    if platform == "twitch" and not twitch_channel:
        twitch_channel = _extract_twitch_channel(args.url)
        if not twitch_channel:
            print(
                "❌  Could not detect Twitch channel from URL. "
                "Use --channel <name>.",
                file=sys.stderr,
            )
            return 1

    log.info("Stream URL : %s", args.url)
    log.info("Platform   : %s", platform)
    log.info("Whisper    : %s on %s (%s)",
             config.WHISPER_MODEL, config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE)
    log.info("Output dir : %s", config.CLIPS_DIR)
    log.info("Output     : %dx%d (%s fill)", config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT,
             config.OUTPUT_FILL_MODE)

    # Build component graph
    recorder    = StreamRecorder(args.url)
    chat_reader = make_chat_reader(platform, args.url, twitch_channel)
    clip_maker  = ClipMaker(recorder, chat_reader, captions=not args.no_captions)
    detector    = HypeDetector(
        chat_reader=chat_reader,
        recorder=recorder,
        on_spike=clip_maker.on_spike,
    )
    console     = StatusConsole(detector, clip_maker, chat_reader, platform)

    # Graceful shutdown on SIGINT / SIGTERM
    _shutdown_called = threading.Event()

    def _shutdown(sig=None, frame=None) -> None:
        if _shutdown_called.is_set():
            return
        _shutdown_called.set()
        print(f"\n\nShutting down...")
        console.stop()
        detector.stop()
        recorder.stop()
        chat_reader.stop()
        print("Waiting for in-progress clips to finish...")
        clip_maker.shutdown()
        log.info(
            "Session complete. Clips made: %d, failed: %d.",
            clip_maker.clips_made, clip_maker.clips_failed,
        )
        print(
            f"\nDone! "
            f"{clip_maker.clips_made} clip(s) saved to {config.CLIPS_DIR}"
        )

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start everything
    recorder.start()

    # Give the recorder a moment to start accumulating segments
    log.info("Waiting 5 s for recorder to initialise...")
    time.sleep(5)

    chat_reader.start()
    detector.start()
    console.start()

    print(
        f"\n{CYAN}Recording...{RESET}  "
        f"Chat: {platform.upper()}  |  "
        f"Model: {config.WHISPER_MODEL}  |  "
        f"Clips -> {config.CLIPS_DIR}\n"
        f"{'─' * 70}"
    )

    # Main thread just waits
    try:
        while not _shutdown_called.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()

    return 0


# ---------------------------------------------------------------------------
# VOD / regular video / manual clipping mode
# ---------------------------------------------------------------------------

def _parse_seconds(value: str | None, flag: str, default: float) -> float:
    if value is None:
        return default
    seconds = parse_timestamp(value)
    if seconds <= 0:
        raise TimestampError(f"{flag} must be greater than zero.")
    return seconds


def _manual_ranges(args: argparse.Namespace) -> list[tuple[float, float]]:
    """Parse manual timestamps from the CLI (shape only; duration is checked later)."""
    ranges = [parse_range(text) for text in args.clip]
    if args.start is not None or args.end is not None:
        if args.start is None:
            raise TimestampError("--end requires --start.")
        start = parse_timestamp(args.start)
        if args.end is not None:
            end = parse_timestamp(args.end)
        elif args.duration is not None:
            end = start + _parse_seconds(args.duration, "--duration", 0)
        else:
            raise TimestampError("--start requires --end or --duration.")
        ranges.append((start, end))
    for start, end in ranges:
        if end <= start:
            raise TimestampError(
                f"End time ({format_timestamp(end)}) must be after start time "
                f"({format_timestamp(start)})."
            )
    return ranges


def _hype_window(args: argparse.Namespace) -> tuple[float, float]:
    """(seconds before, seconds after) each detected hype moment."""
    clip_len = _parse_seconds(args.duration, "--duration", float(config.CLIP_DURATION))
    before = (
        parse_timestamp(args.padding_before)
        if args.padding_before is not None
        else min(float(config.CLIP_LOOKBACK), clip_len)
    )
    after = (
        parse_timestamp(args.padding_after)
        if args.padding_after is not None
        else max(0.0, clip_len - before)
    )
    if before + after <= 0:
        raise TimestampError("Padding before + after must be greater than zero.")
    return before, after


def _print_ranges(ranges: list[ClipRange]) -> None:
    for i, r in enumerate(ranges, start=1):
        hype = f"  (hype at {format_timestamp(r.hype_time)})" if r.hype_time is not None else ""
        print(f"  {i:>2}. {r.label()}  [{r.duration:.0f}s]{hype}")


def _review_ranges(ranges: list[ClipRange], video_duration: float) -> list[ClipRange]:
    """Let the user keep, skip or edit each detected range interactively."""
    if not sys.stdin.isatty():
        log.warning("--review needs an interactive terminal; using detected ranges as-is.")
        return ranges
    print(
        "\nReview detected clips — Enter = keep, s = skip, "
        "or type a new START-END (e.g. 00:01:00-00:01:40):"
    )
    reviewed: list[ClipRange] = []
    for i, r in enumerate(ranges, start=1):
        while True:
            hype = format_timestamp(r.hype_time) if r.hype_time is not None else "-"
            answer = input(f"  Clip {i} [{r.label()}] (hype at {hype}): ").strip()
            if not answer:
                reviewed.append(r)
                break
            if answer.lower() in ("s", "skip"):
                break
            try:
                start, end = parse_range(answer)
                edited = validate_range(start, end, video_duration, config.MANUAL_CLIP_MIN_DURATION)
                edited.hype_time = r.hype_time
                reviewed.append(edited)
                break
            except TimestampError as exc:
                print(f"    ❌ {exc}")
    return reviewed


def _open_file(path: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(path)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:
        log.warning("Could not open %s for preview: %s", path, exc)


def _render_ranges(
    args: argparse.Namespace,
    info: VideoInfo,
    video_path: Path,
    ranges: list[ClipRange],
    min_duration: float,
) -> list[ClipResult]:
    """Generate one clip per range, printing per-clip progress."""
    total = len(ranges)
    print_lock = threading.Lock()
    last_report: dict[int, tuple[str, int]] = {}

    def progress(result: ClipResult, stage: str, fraction: float) -> None:
        step = int(fraction * 10)
        with print_lock:
            if last_report.get(result.index) == (stage, step):
                return
            last_report[result.index] = (stage, step)
            prefix = f"  [clip {result.index}/{total} {result.label()}]"
            if stage == "encoding":
                print(f"{prefix} encoding {step * 10:3d}%", flush=True)
            elif stage == "ok":
                print(f"{prefix} ✅ done → {result.path}", flush=True)
            elif stage == "failed":
                print(f"{prefix} ❌ {result.error}", flush=True)
            else:
                print(f"{prefix} {stage}…", flush=True)

    maker = ClipMaker(None, None, captions=not args.no_captions)
    title = (info.title or "Hype Moment")[: config.TITLE_MAX_CHARS]
    futures = [
        maker.submit_source_clip(
            video_path, r, index=i, title=title, video_label=info.video_id,
            min_duration=min_duration, progress=progress,
        )
        for i, r in enumerate(ranges, start=1)
    ]
    try:
        results = [f.result() for f in futures]
    except KeyboardInterrupt:
        print("\nCancelling clip generation…")
        maker.shutdown(cancel=True)
        raise
    maker.shutdown()
    if args.open:
        for r in results:
            if r.ok and r.path:
                _open_file(r.path)
    return results


def _summarise(args: argparse.Namespace, results: list[ClipResult]) -> None:
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]
    print(f"\n{'─' * 70}\nResults: {len(ok)} succeeded, {len(failed)} failed")
    for r in results:
        if r.ok:
            print(f"  ✅ {r.index:>2}. {r.label()}  {r.path}")
        else:
            print(f"  ❌ {r.index:>2}. {r.label()}  {r.error}")
    if failed:
        clips = " ".join(f"--clip {r.label()}" for r in failed)
        print(f"\nRetry failed clips with:\n  python clipper.py {args.url} {clips}")
    print(f"Log file: {config.LOG_FILE}")


def _run_vod(
    args: argparse.Namespace,
    info: VideoInfo,
    manual: list[tuple[float, float]],
) -> int:
    """
    Pipeline for regular videos and completed livestream replays:
    download once → (detect hype moments | manual ranges) → clip from source.
    """
    log.info("VOD mode — %s (%s, %.0fs)", info.title, info.source_type, info.duration)

    # Fail fast on timestamps that can't fit the reported duration.
    if manual and info.duration > 0:
        try:
            for start, end in manual:
                validate_range(start, end, info.duration, config.MANUAL_CLIP_MIN_DURATION)
        except TimestampError as exc:
            print(f"\n❌ Invalid clip timestamps: {exc}", file=sys.stderr)
            return 2

    try:
        video_path = download_video(
            args.url, config.TMP_DIR / "downloads", info.video_id, info.duration
        )
    except VideoSourceError as exc:
        print(f"\n❌ {exc}", file=sys.stderr)
        return 1

    media = probe_media(video_path)
    if media is None or not media.has_video or media.duration <= 0:
        print("\n❌ Unable to read the downloaded video (no video stream or unknown duration).",
              file=sys.stderr)
        return 1
    duration = media.duration
    if info.duration and abs(info.duration - duration) > 5:
        log.warning(
            "Downloaded duration %.1fs differs from YouTube's %.1fs; "
            "timestamps are validated against the downloaded file.",
            duration, info.duration,
        )
    print(
        f"  Source: {video_path.name} | {media.width}x{media.height} "
        f"{media.video_codec}/{media.audio_codec or 'no audio'} | "
        f"{format_timestamp(duration)}"
    )

    if manual:
        try:
            ranges = [
                validate_range(s, e, duration, config.MANUAL_CLIP_MIN_DURATION)
                for s, e in manual
            ]
        except TimestampError as exc:
            print(f"\n❌ Invalid clip timestamps: {exc}", file=sys.stderr)
            return 2
        min_duration = config.MANUAL_CLIP_MIN_DURATION
        print(f"\nManual clip(s):")
        _print_ranges(ranges)
    else:
        if not media.has_audio:
            print(
                "\n❌ No audio stream found — automatic hype detection needs audio.\n"
                "   Use --start/--end (manual clipping) instead.",
                file=sys.stderr,
            )
            return 1
        before, after = _hype_window(args)
        spikes = VODHypeDetector().find_spikes(
            video_path=video_path,
            base_wall_time=0.0,
            max_clips=args.max_clips,
            threshold_db=args.vod_threshold,
            clip_duration=before + after,
            duration=duration,
        )
        if not spikes:
            print("\n⚠️  No hype moments found. Try a lower --vod-threshold, "
                  "or clip manually with --start/--end.")
            return 0
        ranges = [build_hype_range(t, before, after, duration) for t in spikes]
        min_duration = min(float(config.CLIP_MIN_DURATION), duration)
        print(f"\nDetected {len(ranges)} hype moment(s):")
        _print_ranges(ranges)
        if args.detect_only:
            clips = " ".join(f"--clip {r.label()}" for r in ranges)
            print(f"\nGenerate (or edit) these clips with:\n  python clipper.py {args.url} {clips}")
            return 0
        if args.review:
            ranges = _review_ranges(ranges, duration)
            if not ranges:
                print("No clips selected.")
                return 0

    print(f"\nGenerating {len(ranges)} clip(s)… (Ctrl+C to cancel)")
    results = _render_ranges(args, info, video_path, ranges, min_duration)

    # Offer to retry failures interactively.
    for _attempt in range(3):
        failed = [r for r in results if not r.ok]
        if not failed or not sys.stdin.isatty():
            break
        answer = input(f"\nRetry {len(failed)} failed clip(s)? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            break
        retried = _render_ranges(
            args, info, video_path,
            [ClipRange(r.start, r.end, r.hype_time) for r in failed],
            min_duration,
        )
        by_index = {r.index: new for r, new in zip(failed, retried)}
        results = [
            _reindexed(by_index[r.index], r.index) if r.index in by_index else r
            for r in results
        ]

    _summarise(args, results)
    return 0 if all(r.ok for r in results) else 1


def _reindexed(result: ClipResult, index: int) -> ClipResult:
    result.index = index
    return result

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    _setup_logging(debug=args.debug)

    # Apply CLI overrides to config
    if args.whisper_model:
        config.WHISPER_MODEL = args.whisper_model
    if args.chat_threshold:
        config.CHAT_SPIKE_MULTIPLIER = args.chat_threshold

    print(
        f"\n{BOLD}{CYAN}"
        "  _____ _____ _____   _____ _ _                       \n"
        " |  __ \\_   _|  __ \\ / ____| (_)                      \n"
        " | |  | || | | |__) | |    | |_ _ __  _ __   ___ _ __ \n"
        " | |  | || | |  ___/| |    | | | '_ \\| '_ \\ / _ \\ '__|\\n"
        " | |__| || |_| |    | |____| | | |_) | |_) |  __/ |   \n"
        " |_____/_____|_|     \\_____|_|_| .__/| .__/ \\___|_|   \n"
        "                               | |   | |               \n"
        "                               |_|   |_|               \n"
        f"{RESET}"
    )
    print(f"  Live Stream & Video Auto-Clipper  |  free & local  |  Ctrl+C to stop\n")

    # Dependency check
    if not _check_dependencies():
        return 1

    # Validate manual timestamps before doing any network work.
    try:
        manual = _manual_ranges(args)
        if not manual:
            _hype_window(args)
    except TimestampError as exc:
        print(f"❌ Invalid clip timestamps: {exc}", file=sys.stderr)
        return 2

    # Platform detection
    platform = args.platform or _detect_platform(args.url)

    if platform != "youtube":
        if manual or args.info:
            print("❌ Manual clipping and --info are only supported for YouTube videos.",
                  file=sys.stderr)
            return 2
        # Twitch — always live stream mode
        return _run_live(args, platform)

    # For YouTube: normalize URL and probe video metadata to decide mode
    url = normalize_youtube_url(args.url)
    if url != args.url:
        log.info("Normalized URL: %s → %s", args.url, url)
        args.url = url

    print(f"Loading video info: {args.url} …")
    try:
        info = probe_video_detailed(args.url)
    except VideoSourceError as exc:
        print(f"\n❌ Unable to retrieve video information: {exc}", file=sys.stderr)
        return 1

    type_label = {
        "live":     "LIVE STREAM",
        "archived": "COMPLETED LIVESTREAM (replay)",
        "video":    "REGULAR VIDEO",
    }.get(info.source_type, info.source_type.upper())
    if info.is_upcoming:
        type_label = "UPCOMING / SCHEDULED"
    print(
        f"  Title    : {info.title}\n"
        f"  Channel  : {info.uploader}\n"
        f"  Type     : {type_label}\n"
        f"  Duration : {format_timestamp(info.duration) if info.duration else 'unknown (live)'}"
    )
    if args.info:
        return 0

    if info.is_upcoming:
        print(f"\n❌ This video is scheduled but not yet live: {info.title}", file=sys.stderr)
        return 1

    if info.is_live:
        if manual:
            print(
                "\n❌ Manual timestamp clipping needs a finished video; this stream is "
                "still live. Run it again on the replay once the stream ends.",
                file=sys.stderr,
            )
            return 2
        print(f"  → Live stream — switching to live mode.\n")
        return _run_live(args, platform)

    print(f"  → Processing as {'manual clip' if manual else 'hype detection'} on the full video.\n")
    try:
        return _run_vod(args, info, manual)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except TimestampError as exc:
        print(f"\n❌ Invalid clip timestamps: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    sys.exit(main())
