# GTAClipper 🎬

**Free, local, zero-API-key live stream auto-clipper.**

Watches any YouTube or Twitch live stream, detects hype moments from a chat spike or an audio loudness spike, and automatically generates vertical 9:16 short clips with burned-in captions and a chat-derived title — all while the stream is live.

---

## How It Works

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
              • Chat: messages/window > rolling_avg × 2.0 AND ≥ 5 msgs
              • Audio: RMS dBFS > rolling_avg + 5 dB
              • Either signal spikes? → fire ClipMaker
                    │
                    ▼
              Extract 30 s from 20 s before spike
              Crop to 1080×1920 (9:16 vertical)
              faster-whisper → SRT captions
              Chat n-gram → title overlay
              → clips/clip_<timestamp>_<title>.mp4
```

---

## Requirements

| Dependency | Install |
|-----------|---------|
| **Python 3.11+** | [python.org](https://www.python.org/downloads/) |
| **ffmpeg + ffprobe** | [ffmpeg.org/download](https://ffmpeg.org/download.html) — must be on PATH |
| **yt-dlp** | `pip install yt-dlp` (included in requirements.txt) |
| **Deno** | Required by yt-dlp for YouTube stream extraction; install it and make sure `deno` is on PATH |
| **faster-whisper** | `pip install faster-whisper` (included in requirements.txt) |

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
  --threshold  <float>              Chat spike multiplier (default: 2.0)
  --debug                           Enable verbose debug logging
```

### Examples
```bash
# Higher quality captions (slower, needs more RAM)
python clipper.py https://www.youtube.com/watch?v=LIVE_ID --model small

# More sensitive chat spike detection
python clipper.py https://www.twitch.tv/shroud --platform twitch --channel shroud --threshold 1.5

# Debug mode
python clipper.py https://www.youtube.com/watch?v=LIVE_ID --debug
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `SEGMENT_DURATION` | `30` | Seconds per .ts segment |
| `ANALYSIS_INTERVAL` | `10` | Seconds between hype checks |
| `CLIP_DURATION` | `30` | Output clip length (seconds) |
| `CLIP_LOOKBACK` | `20` | Seconds before spike where clip starts |
| `CHAT_SPIKE_MULTIPLIER` | `2.0` | Chat must be this × its avg to spike |
| `CHAT_SPIKE_MIN_MESSAGES` | `5` | Minimum absolute messages to count as spike |
| `LOUDNESS_SPIKE_DB` | `5.0` | dB above rolling avg to count as loud |
| `CLIP_COOLDOWN_SECONDS` | `60` | Minimum gap between clips |
| `WHISPER_MODEL` | `"base"` | Whisper model size |
| `WHISPER_DEVICE` | `"cpu"` | `"cpu"` or `"cuda"` (NVIDIA GPU) |

---

## Output

Clips are saved to the `clips/` folder in the format:
```
clips/clip_20241120_143022_Clip_It.mp4
       └── timestamp ─┘ └── chat title ─┘
```

Each clip is:
- **1080×1920** (9:16 vertical, TikTok/Reels/Shorts ready)
- **H.264 video + AAC audio**
- **Burned-in captions** (white text, black outline)
- **Title overlay** at the top (derived from chat messages)

---

## File Structure

```
GTAClipper/
├── clipper.py          ← Main entry point (run this)
├── recorder.py         ← Stream recorder (yt-dlp + ffmpeg)
├── chat_reader.py      ← YouTube live chat + Twitch IRC
├── hype_detector.py    ← Spike detection engine
├── clip_maker.py       ← FFmpeg + faster-whisper pipeline
├── title_generator.py  ← Chat → title algorithm
├── config.py           ← All settings in one place
├── requirements.txt    ← Python dependencies
├── clips/              ← Saved clips (auto-created)
└── tmp/                ← Working buffer (auto-created & cleaned)
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
| No live chat on YouTube | Confirm the stream is live with active chat |
| Clips are silent | Use `-c:a aac` instead of `-c:a copy` in clip_maker.py |
| Too many / too few clips | Adjust thresholds in `config.py` |

---

## License

Free to use for personal use. `yt-dlp`, `ffmpeg`, and `faster-whisper` are each governed by their own open-source licenses.
