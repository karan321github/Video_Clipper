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
from clip_maker    import ClipMaker
from video_processor import (
    probe_video,
    normalize_youtube_url,
    download_video,
    OfflineSegmentProvider,
    VODHypeDetector,
    NullChatReader,
    VideoInfo,
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

    # faster-whisper
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        print(
            "[ERROR] faster-whisper not installed.\n"
            "        Install with: pip install faster-whisper",
            file=sys.stderr,
        )
        ok = False

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
    clip_maker  = ClipMaker(recorder, chat_reader)
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
# VOD / regular video mode
# ---------------------------------------------------------------------------

def _run_vod(args: argparse.Namespace, info: VideoInfo) -> int:
    """
    Offline video pipeline for published VODs and archived streams.
    Downloads the full video, segments it, runs VODHypeDetector,
    then passes each spike to ClipMaker.
    """
    log.info("VOD mode — %s (%s, %.0fs)", info.title, info.source_type, info.duration)
    log.info("Output: %dx%d (%s fill)", config.OUTPUT_WIDTH, config.OUTPUT_HEIGHT,
             config.OUTPUT_FILL_MODE)

    # Download
    dl_dir     = config.TMP_DIR / "downloads"
    video_path = download_video(args.url, dl_dir, info.video_id)
    if not video_path:
        log.error("Video download failed. Cannot continue.")
        return 1

    # Segment into rolling .ts chunks
    provider = OfflineSegmentProvider(video_path, info.duration or _get_local_duration(video_path))
    if not provider.prepare():
        log.error("Segmentation failed. Cannot continue.")
        return 1

    # Use NullChatReader (no live chat for VODs)
    chat_reader = NullChatReader()

    # Create ClipMaker using the OfflineSegmentProvider as the recorder
    clip_maker = ClipMaker(provider, chat_reader)  # type: ignore[arg-type]

    # Detect spikes in the VOD
    base_wall = provider.video_start_time
    vod_detector = VODHypeDetector()
    spikes = vod_detector.find_spikes(
        video_path=video_path,
        base_wall_time=base_wall,
        max_clips=args.max_clips,
        threshold_db=args.vod_threshold,
    )

    if not spikes:
        log.warning(
            "No hype spikes found. Check the audio track or lower --vod-threshold."
        )
        provider.cleanup()
        return 0

    log.info("Submitting %d clip jobs…", len(spikes))
    for spike_time in spikes:
        clip_maker.on_spike(spike_time)

    # Wait for all clips to finish
    print(f"\nProcessing {len(spikes)} clip(s)… (this may take a while)")
    clip_maker.shutdown()

    log.info(
        "VOD complete. Clips made: %d, failed: %d.",
        clip_maker.clips_made, clip_maker.clips_failed,
    )
    print(
        f"\nDone! "
        f"{clip_maker.clips_made} clip(s) saved to {config.CLIPS_DIR}"
    )

    # Clean up segments
    provider.cleanup()
    return 0


def _get_local_duration(video_path: Path) -> float:
    """Probe duration of a local file."""
    from video_processor import _probe_duration
    return _probe_duration(video_path)


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

    # Platform detection
    platform = args.platform or _detect_platform(args.url)

    # For YouTube: normalize URL and probe video metadata to decide mode
    if platform == "youtube":
        url = normalize_youtube_url(args.url)
        if url != args.url:
            log.info("Normalized URL: %s → %s", args.url, url)
            args.url = url

        print(f"Probing video: {args.url} …")
        info = probe_video(args.url)

        if info is None:
            # probe_video already logged the specific error
            print(
                "\n❌ Could not retrieve video info. "
                "The video may be private, age-restricted, region-locked, or deleted.",
                file=sys.stderr,
            )
            return 1

        if info.is_upcoming:
            print(
                f"\n❌ This video is scheduled but not yet live: {info.title}",
                file=sys.stderr,
            )
            return 1

        log.info(
            "Video: %r | type=%s | duration=%.0fs",
            info.title, info.source_type, info.duration,
        )

        if info.is_live:
            # Currently live → use original live-stream pipeline
            print(f"  → Detected: LIVE STREAM — switching to live mode.\n")
            return _run_live(args, platform)
        else:
            # Archived stream or regular published video → VOD pipeline
            type_label = {
                "archived": "ARCHIVED LIVESTREAM",
                "video":    "REGULAR VIDEO",
            }.get(info.source_type, info.source_type.upper())
            print(f"  → Detected: {type_label} — processing as VOD.\n")
            return _run_vod(args, info)

    else:
        # Twitch — always live stream mode
        return _run_live(args, platform)


if __name__ == "__main__":
    sys.exit(main())
