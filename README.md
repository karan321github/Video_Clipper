# GTAClipper 🎬

**Free, local, zero-API-key live stream auto-clipper.**

Watches YouTube or Twitch live streams and processes regular YouTube videos and completed livestream replays. It detects chat or audio spikes (hype moments) and generates horizontal 16:9 clips with an optional burned-in captions track and a title. You can also clip exact timestamps manually.

---

## How It Works

### Regular videos & completed livestream replays
```
YouTube URL
   │
   ▼
yt-dlp probes the video type (regular / replay / live / upcoming)
   │
   ▼
Download once (≤1080p, H.264/AAC preferred, separate A/V merged) → tmp/downloads/
   │                 (reused on later runs if complete & valid)
   ├── Automatic: single-pass audio scan → hype moments (dB above rolling baseline)
   │              → each moment becomes [moment − padding-before, moment + padding-after]
   └── Manual:    your --start/--end or --clip ranges
   │
   ▼
Validate every range against the real file duration
   │
   ▼
ffmpeg: accurate input-seek + re-encode → 1920×1080 H.264/AAC MP4
   │
   ▼
Verify output (exists, streams, duration, full decode) → clips/
```

### Live streams
```
Stream URL
   │
   ▼
yt-dlp resolves best stream URL
   │
   ▼
ffmpeg writes rolling 30-second .ts segments → tmp/
   │
   ├──► Chat reader (YouTube JSONL / Twitch IRC) → message deque
   │
   └──► Every 10 s: HypeDetector checks both signals
              • Chat: messages/window > rolling_avg × 3.0 AND ≥ 15 msgs
              • Audio: RMS dBFS > rolling_avg + 5 dB
              • Either signal spikes? → fire ClipMaker
                    │
                    ▼
              Wait until the footage after the spike is recorded
              Cut 240 s starting 20 s before the spike
              Render to 1920×1080 (16:9 horizontal)
              faster-whisper → SRT captions (optional)
              Chat n-gram → title overlay
              → clips/clip_<timestamp>_<title>_<id>.mp4
```

---

## Requirements

| Dependency | Install |
|-----------|---------|
| **Python 3.11+** | [python.org](https://www.python.org/downloads/) |
| **ffmpeg + ffprobe** | [ffmpeg.org/download](https://ffmpeg.org/download.html) — must be on PATH |
| **yt-dlp** | `pip install yt-dlp` (included in requirements.txt), or drop `yt-dlp.exe` into a project-local `ffmpeg/` folder |
| **Deno** | Required by yt-dlp for YouTube stream extraction; install it and make sure `deno` is on PATH |
| **faster-whisper** *(optional)* | `pip install faster-whisper` — only needed for burned-in captions; clips are generated without captions if missing |

> **Windows users**: After installing ffmpeg, add its `bin/` folder to your system PATH.

---

## Installation

```bash
# Clone or download the project
cd GTAClipper

# Install Python dependencies
pip install -r requirements.txt
```

The first time you run a clip, faster-whisper will download the Whisper `base` model (~145 MB) automatically.

---

## Usage

### YouTube live stream
```bash
python clipper.py https://www.youtube.com/watch?v=LIVE_VIDEO_ID
```

### Regular or archived YouTube video (automatic hype detection)
```bash
python clipper.py https://youtu.be/VIDEO_ID --max-clips 5

# 60-second clips, 10 s before / 50 s after each hype moment
python clipper.py https://youtu.be/VIDEO_ID --duration 60 --padding-before 10 --padding-after 50

# Review/edit the detected timestamps before clips are generated
python clipper.py https://youtu.be/VIDEO_ID --review

# Only print the detected ranges (edit them and pass them back with --clip)
python clipper.py https://youtu.be/VIDEO_ID --detect-only
```

### Manual timestamp clipping
Timestamps accept `HH:MM:SS`, `MM:SS` or seconds.
```bash
# Load video information (type, duration) only
python clipper.py https://youtu.be/VIDEO_ID --info

# 40-second clip from 00:02:15 to 00:02:55
python clipper.py https://youtu.be/VIDEO_ID --start 00:02:15 --end 00:02:55

# Start + duration
python clipper.py https://youtu.be/VIDEO_ID --start 2:15 --duration 40

# Several ranges at once, then open each finished clip for preview
python clipper.py https://youtu.be/VIDEO_ID --clip 00:02:15-00:02:55 --clip 600-645 --open
```
Invalid input (negative times, end before start, times beyond the video duration) is rejected with an explanation before anything is downloaded.
After every run a summary lists each clip as ✅ (with its file path) or ❌ (with the reason). For failed clips you are offered a retry, and a ready-to-run `--clip` retry command is printed.

### Twitch stream
```bash
python clipper.py https://www.twitch.tv/xqc --platform twitch --channel xqc
```

### All options
```
python clipper.py <url> [options]

Options:
  --platform   youtube | twitch     (auto-detected from URL if omitted)
  --channel    <name>               Twitch channel name (required for Twitch)
  --model      tiny|base|small|medium|large-v2   Whisper model (default: base)
  --threshold  <float>              Chat spike multiplier (default: 3.0)
  --max-clips  <int>                Maximum VOD clips (default: 20)
  --vod-threshold <float>           VOD audio threshold in dB (default: 3.0)
  --debug                           Enable verbose debug logging

Manual clipping / timestamp editing (regular videos & replays):
  --start <time> --end <time>       Clip an exact range
  --clip START-END                  Clip a range (repeatable)
  --duration <time>                 Manual: length after --start. Automatic: clip length (default 240 s)
  --padding-before <time>           Automatic: kept before each hype moment (default 20 s)
  --padding-after <time>            Automatic: kept after each hype moment (default duration − before)
  --review                          Automatic: edit/skip detected ranges interactively
  --detect-only                     Automatic: print detected ranges and exit
  --info                            Print video information and exit
  --no-captions                     Skip transcription
  --open                            Open finished clips in the default player
```

Exit codes: `0` all clips succeeded, `1` a clip/download/probe failed, `2` invalid timestamps.

### Examples
```bash
# Higher quality captions (slower, needs more RAM)
python clipper.py https://www.youtube.com/watch?v=LIVE_ID --model small

# More sensitive chat spike detection
python clipper.py https://www.twitch.tv/shroud --platform twitch --channel shroud --threshold 1.5

# Process a published video with a lower audio-spike threshold
python clipper.py https://youtu.be/VIDEO_ID --max-clips 5 --vod-threshold 2.5

# Debug mode
python clipper.py https://www.youtube.com/watch?v=LIVE_ID --debug
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `SEGMENT_DURATION` | `30` | Seconds per .ts segment |
| `ANALYSIS_INTERVAL` | `10` | Seconds between hype checks |
| `CLIP_DURATION` | `240` | Output clip length (seconds) |
| `CLIP_LOOKBACK` | `20` | Seconds before spike where clip starts |
| `CHAT_SPIKE_MULTIPLIER` | `3.0` | Chat must be this × its avg to spike |
| `CHAT_SPIKE_MIN_MESSAGES` | `15` | Minimum absolute messages to count as spike |
| `LOUDNESS_SPIKE_DB` | `5.0` | dB above rolling avg to count as loud |
| `VOD_LOUDNESS_SPIKE_DB` | `3.0` | VOD audio threshold above rolling baseline |
| `CLIP_COOLDOWN_SECONDS` | `45` | Minimum gap between live spike detections |
| `OUTPUT_WIDTH` × `OUTPUT_HEIGHT` | `1920×1080` | Horizontal 16:9 output resolution |
| `WHISPER_MODEL` | `"base"` | Whisper model size |
| `WHISPER_DEVICE` | `"cpu"` | `"cpu"` or `"cuda"` (NVIDIA GPU) |
| `VOD_FORMAT` / `VOD_FORMAT_SORT` | ≤1080p, prefer H.264/AAC | yt-dlp format selection for regular videos & replays |
| `CLIP_DURATION_TOLERANCE` | `1.5` | Max difference (s) between requested and produced clip length |
| `MIN_FREE_DISK_MB` | `500` | Downloads/encodes stop with a clear error below this free space |

---

## Output

Clips are saved to the `clips/` folder:
```
clips/clip_<videoId>_00h02m15s-00h02m55s_<id>.mp4      ← regular videos / replays / manual
clips/clip_20241120_143022_Clip_It_<id>.mp4             ← live streams (timestamp + chat title)
```
The short random `<id>` guarantees that parallel jobs never overwrite each other.

Each clip is:
- **1920×1080** (16:9 horizontal, YouTube-ready)
- **H.264 (yuv420p) video + AAC audio**, `+faststart` MP4
- **Verified** before it is saved: correct streams, correct duration, decodes without errors
- **Burned-in captions** when faster-whisper is installed
- **Title overlay** at the top (video title for VODs, derived from chat for live streams)

---

## File Structure

```
GTAClipper/
├── clipper.py          ← Main entry point (run this)
├── recorder.py         ← Live stream recorder (yt-dlp + ffmpeg)
├── chat_reader.py      ← YouTube live chat + Twitch IRC
├── hype_detector.py    ← Live spike detection engine
├── video_processor.py  ← Video probing, download, VOD hype detection
├── clip_maker.py       ← Shared FFmpeg render + verification pipeline
├── media_tools.py      ← Subprocess runner, ffprobe helpers, log redaction
├── timestamps.py       ← Timestamp parsing & clip-range validation
├── title_generator.py  ← Chat → title algorithm
├── config.py           ← All settings in one place
├── requirements.txt    ← Python dependencies
├── clips/              ← Saved clips (auto-created)
└── tmp/                ← Working files & download cache (auto-created & cleaned)
```

---

## Tips

- **Low-end PC**: Use `--model tiny` for fastest transcription, or omit captions entirely by setting `WHISPER_MODEL = "tiny"` in config.py
- **GPU**: Change `WHISPER_DEVICE = "cuda"` and `WHISPER_COMPUTE_TYPE = "float16"` in config.py for 5–10× faster transcription
- **Quiet streams**: Lower `LOUDNESS_SPIKE_DB` to `3.0` or `2.0` in config.py
- **Busy chat**: Raise `CHAT_SPIKE_MULTIPLIER` to `3.0` to reduce false positives
- **Log file**: All events are written to `clipper.log`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ffmpeg not found` | Install ffmpeg and add to PATH |
| `yt-dlp not found` | `pip install yt-dlp` |
| `Unable to retrieve video information: …` | The reason is shown (private, age-restricted, members-only, region-locked, not started, removed…) |
| `Sign in to confirm you're not a bot` | YouTube rate-limited this machine; wait and retry |
| No live chat on YouTube | Confirm the stream is live with active chat |
| `No audio stream found` | Automatic detection needs audio; use `--start/--end` instead |
| `Invalid clip timestamps: …` | Check the format (`HH:MM:SS`, `MM:SS`, seconds) and the video duration shown by `--info` |
| `FFmpeg processing failed: exit code …: <cause>` | The root-cause line from ffmpeg is shown; full command and stderr are in `clipper.log` |
| `Generated video failed validation: …` | The output was discarded; the source may be truncated (e.g. a stream replay still processing) |
| `Insufficient disk space` | Free space or lower `MIN_FREE_DISK_MB` |
| Too many / too few clips | Adjust thresholds in `config.py` or `--vod-threshold` |

---

## License

Free to use for personal use. `yt-dlp`, `ffmpeg`, and `faster-whisper` are each governed by their own open-source licenses.
