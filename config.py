"""
config.py — GTAClipper Configuration
All user-tuneable constants live here. Edit freely before running.
"""

# ── Recording ──────────────────────────────────────────────────────────────
SEGMENT_DURATION: int = 30          # seconds per .ts segment written by ffmpeg
# Buffer must cover CLIP_LOOKBACK + CLIP_DURATION + margin.
# Default 240 s clip + 20 s lookback + 10 s margin ≈ 270 s → 9 segments of 30 s.
# Keep 12 for safety (12 × 30 = 360 s).
SEGMENT_BUFFER_KEEP: int = 12       # how many past segments to keep on disk
STREAM_QUALITY: str = "bestvideo+bestaudio/best"  # Separate live tracks, with muxed fallback
# Regular videos / replays: output is 1080p, so don't download 4K/8K sources.
# Any codec (H.264, VP9, AV1, AAC, Opus…) is accepted — clips are re-encoded.
VOD_FORMAT: str = "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"
# Among allowed formats prefer ≤1080p, then H.264/AAC: they decode far faster
# than AV1/VP9 (clips are re-encoded either way, so quality is unaffected).
VOD_FORMAT_SORT: str = "res:1080,vcodec:h264,acodec:aac"
VOD_DOWNLOAD_TIMEOUT: int = 3 * 3600   # seconds

# ── Analysis cadence ──────────────────────────────────────────────────────
ANALYSIS_INTERVAL: float = 10.0     # seconds between hype checks

# ── Clip geometry ─────────────────────────────────────────────────────────
CLIP_DURATION: int = 240            # max length of the output clip (seconds) — configurable
CLIP_MIN_DURATION: int = 15         # minimum clip length (seconds)
CLIP_LOOKBACK: int = 20             # seconds before spike where clip starts
MANUAL_CLIP_MIN_DURATION: float = 1.0  # shortest clip allowed in manual mode
# Generated clips must be within this many seconds of the requested length.
CLIP_DURATION_TOLERANCE: float = 1.5
# Refuse to start downloads/encodes when less free space than this remains.
MIN_FREE_DISK_MB: int = 500

# Output: 16:9 horizontal (YouTube-ready)
OUTPUT_WIDTH:  int = 1920
OUTPUT_HEIGHT: int = 1080           # 16:9 horizontal

# When source aspect ratio differs from 16:9, fill empty bars with:
#   "blur"  — blurred & scaled copy of the source (cinematic letterbox feel)
#   "black" — plain black bars
OUTPUT_FILL_MODE: str = "blur"      # "blur" | "black"

# ── Video source / layout ────────────────────────────────────────────────
# Source type auto-detection: "auto" | "live" | "archived" | "video"
VIDEO_SOURCE_TYPE: str = "auto"

# Streamer webcam/face-cam overlay.
# "auto"  — include if detected in the source stream
# "off"   — never overlay
INCLUDE_STREAMER_CAM: str = "auto"  # "auto" | "off"

# Streamer cam overlay geometry (fraction of output width, anchored bottom-right)
CAM_OVERLAY_WIDTH_FRAC: float = 0.28   # 28 % of output width
CAM_OVERLAY_MARGIN: int = 20           # px margin from edges

# ── Spike thresholds ──────────────────────────────────────────────────────
# Chat: messages-per-window must exceed rolling_avg * multiplier
CHAT_SPIKE_MULTIPLIER: float = 3.0   # Tuned UP for very busy streams
# Chat: also require at least this many absolute messages (avoids spike on 2→5)
CHAT_SPIKE_MIN_MESSAGES: int = 15   # Minimum 15 messages in 10s window to count

# Audio: RMS loudness must exceed rolling_avg_rms + this many dB.
# Chat is optional: an audio spike alone can trigger a clip.
LOUDNESS_SPIKE_DB: float = 5.0
VOD_LOUDNESS_SPIKE_DB: float = 3.0  # lower relative threshold for prerecorded audio
# How many analysis windows to average for the rolling baseline
ROLLING_WINDOW_SIZE: int = 6        # 6 × 10 s = 60 s baseline

# ── Cooldown ─────────────────────────────────────────────────────────────
CLIP_COOLDOWN_SECONDS: int = 45     # 45 s cooldown — get more clips from busy stream

# ── Faster-Whisper ────────────────────────────────────────────────────────
WHISPER_MODEL: str = "base"         # tiny | base | small | medium | large-v2
WHISPER_DEVICE: str = "cpu"         # cpu | cuda
WHISPER_COMPUTE_TYPE: str = "int8"  # int8 (CPU) | float16 (GPU)
WHISPER_LANGUAGE: str | None = None # None = auto-detect

# ── Title generator ───────────────────────────────────────────────────────
TITLE_MAX_CHARS: int = 50
# How many recent chat messages to scan when deriving the title
TITLE_CHAT_WINDOW: int = 60         # messages

# ── Caption style (ffmpeg drawtext / subtitles filter) ───────────────────
CAPTION_FONT_SIZE: int = 36         # slightly smaller for wider 16:9 frame
CAPTION_FONT_COLOR: str = "white"
CAPTION_BOX_COLOR: str = "black@0.6"
CAPTION_BOX_BORDER: int = 10

TITLE_FONT_SIZE: int = 48
TITLE_FONT_COLOR: str = "white"
TITLE_BG_COLOR: str = "black@0.75"

# ── Paths ─────────────────────────────────────────────────────────────────
import pathlib
BASE_DIR   = pathlib.Path(__file__).parent.resolve()
TMP_DIR    = BASE_DIR / "tmp"
CLIPS_DIR  = BASE_DIR / "clips"
LOG_FILE   = BASE_DIR / "clipper.log"

# ── ffmpeg / yt-dlp executables ───────────────────────────────────────────
# Change to full paths if they are not on your system PATH.
# Auto-detects yt-dlp from the Python Scripts folder when not on PATH.
import shutil as _shutil
import sys as _sys

def _find_bin(name: str) -> str:
    """Return the full path to `name` if found, else just `name` (rely on PATH)."""
    import os
    # 1. Check PATH first
    found = _shutil.which(name)
    if found:
        return found
    # 2. Candidate dirs (Python Scripts, user Scripts, AppData Roaming user install)
    candidates_dirs = [
        pathlib.Path(_sys.executable).parent / "Scripts",
        pathlib.Path(_sys.prefix) / "Scripts",
        # pip --user on Windows installs here
        pathlib.Path(os.environ.get("APPDATA", "")) / "Python" / f"Python{_sys.version_info.major}{_sys.version_info.minor}" / "Scripts",
        pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python" / f"Python{_sys.version_info.major}{_sys.version_info.minor}" / "Scripts",
    ]
    for d in candidates_dirs:
        for candidate in (name, name + ".exe"):
            p = d / candidate
            if p.exists():
                return str(p)
    # 3. Also check the project-local ffmpeg/ folder (auto-downloaded)
    local_ffmpeg = pathlib.Path(__file__).parent / "ffmpeg"
    if local_ffmpeg.is_dir():
        for candidate in (name, name + ".exe"):
            p = local_ffmpeg / candidate
            if p.exists():
                return str(p)
    return name   # fallback — will fail with a clear error at runtime

FFMPEG_BIN  = _find_bin("ffmpeg")
FFPROBE_BIN = _find_bin("ffprobe")
YTDLP_BIN   = _find_bin("yt-dlp")
DENO_BIN    = _find_bin("deno")     # JS runtime yt-dlp uses for YouTube


def ytdlp_js_runtime_args(url: str) -> list[str]:
    """`--js-runtimes` arguments for YouTube URLs, only if Deno is installed."""
    lowered = url.lower()
    if "youtube.com" not in lowered and "youtu.be" not in lowered:
        return []
    if pathlib.Path(DENO_BIN).is_absolute() and pathlib.Path(DENO_BIN).exists():
        return ["--js-runtimes", f"deno:{DENO_BIN}"]
    return []
