"""
chat_reader.py — GTAClipper Chat Reader
=========================================
Reads live chat from YouTube or Twitch and feeds messages into a shared deque.

YouTube strategy
----------------
Run yt-dlp as a subprocess with --write-subs --sub-lang live_chat pointing
to a temporary file, then tail and parse the JSONL lines it appends.
Each JSONL line is a JSON object; chat actions are nested inside.

Twitch strategy
---------------
Anonymous IRC connection (justinfanXXXXX / no OAuth) over raw socket.
Reconnects automatically on drop.
"""

from __future__ import annotations

import json
import logging
import random
import re
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import config

log = logging.getLogger(__name__)

TWITCH_HOST = "irc.chat.twitch.tv"
TWITCH_PORT = 6667

# Regex for Twitch PRIVMSG lines
_PRIVMSG_RE = re.compile(
    r"^:([^!]+)![^@]+@[^\s]+ PRIVMSG (#\S+) :(.+)\r?$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Shared message type
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    timestamp: float          # time.time()
    platform:  str            # "youtube" | "twitch"
    username:  str
    text:      str

    def __repr__(self) -> str:
        return f"[{self.platform}] {self.username}: {self.text!r}"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class ChatReader:
    """
    Abstract base. Subclasses push ChatMessage objects into `self.messages`.
    External code (HypeDetector, TitleGenerator) reads from this deque.
    """

    def __init__(self, maxlen: int = 2000):
        self.messages: deque[ChatMessage] = deque(maxlen=maxlen)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=10)

    def recent(self, seconds: float) -> list[ChatMessage]:
        """Return messages from the last `seconds` seconds."""
        cutoff = time.time() - seconds
        with_lock = list(self.messages)   # snapshot (deque is thread-safe for iteration)
        return [m for m in with_lock if m.timestamp >= cutoff]


# ---------------------------------------------------------------------------
# YouTube live chat
# ---------------------------------------------------------------------------

class YouTubeChatReader(ChatReader):
    """
    Uses yt-dlp to download the live_chat subtitle stream into a JSONL file,
    then tails that file for new events in real time.
    """

    def __init__(self, stream_url: str):
        super().__init__()
        self.stream_url = stream_url
        self._chat_file: Optional[Path] = None
        self._ytdlp_proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="YTChat"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._ytdlp_proc and self._ytdlp_proc.poll() is None:
            self._ytdlp_proc.terminate()
            try:
                self._ytdlp_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._ytdlp_proc.kill()
        if self._thread:
            self._thread.join(timeout=15)

    def _run(self) -> None:
        # yt-dlp writes the live chat to a .live_chat.json file
        chat_path = config.TMP_DIR / f"livechat_{uuid.uuid4().hex}.live_chat.json"
        out_tmpl = str(chat_path).removesuffix(".live_chat.json") + ".%(ext)s"

        cmd = [
            config.YTDLP_BIN,
            "--js-runtimes", "deno",
            "--skip-download",
            "--write-subs",
            "--sub-lang", "live_chat",
            "--no-playlist",
            "-o", out_tmpl,
            self.stream_url,
        ]

        log.info("Starting YouTube live chat reader…")
        try:
            self._ytdlp_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            log.error("yt-dlp not found. Install with: pip install yt-dlp")
            return

        # Wait briefly for the chat file to appear
        for _ in range(30):
            if self._stop_event.is_set():
                return
            if chat_path.exists():
                break
            time.sleep(1)
        else:
            log.warning(
                "Live chat file did not appear within 30 s. "
                "The stream may not have live chat enabled."
            )
            return

        self._chat_file = chat_path
        log.info("Tailing live chat file: %s", chat_path)
        self._tail_chat_file(chat_path)

    def _tail_chat_file(self, path: Path) -> None:
        """Read the JSONL file line by line, blocking when at EOF."""
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            while not self._stop_event.is_set():
                line = fh.readline()
                if not line:
                    time.sleep(0.5)
                    continue
                self._parse_jsonl_line(line.strip())

    def _parse_jsonl_line(self, line: str) -> None:
        """Parse a single JSONL line from yt-dlp's live_chat output."""
        if not line:
            return
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            return

        # yt-dlp wraps each chat action in an array of "replayChatItemAction"
        actions = obj.get("replayChatItemAction", {}).get("actions", [])
        if not actions:
            # Try direct actions key (live format)
            actions = obj.get("actions", [])

        for action in actions:
            item = (
                action
                .get("addChatItemAction", {})
                .get("item", {})
            )
            # Regular text message
            msg_renderer = item.get("liveChatTextMessageRenderer", {})
            if not msg_renderer:
                # Superchat / membership — also capture
                msg_renderer = item.get("liveChatPaidMessageRenderer", {})
            if not msg_renderer:
                continue

            author = msg_renderer.get("authorName", {}).get("simpleText", "unknown")
            runs   = msg_renderer.get("message", {}).get("runs", [])
            text   = "".join(r.get("text", "") for r in runs).strip()

            if text:
                self.messages.append(ChatMessage(
                    timestamp=time.time(),
                    platform="youtube",
                    username=author,
                    text=text,
                ))
                log.debug("[YT chat] %s: %s", author, text)


# ---------------------------------------------------------------------------
# Twitch IRC anonymous reader
# ---------------------------------------------------------------------------

class TwitchChatReader(ChatReader):
    """
    Connects to Twitch IRC anonymously (justinfanXXXXX) over a raw socket.
    No OAuth token required — read-only access.
    Automatically reconnects on disconnection.
    """

    def __init__(self, channel: str):
        super().__init__()
        # Normalise: strip leading # and lowercase
        self.channel = "#" + channel.lstrip("#").lower()
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="TwitchChat"
        )
        self._thread.start()

    def _run(self) -> None:
        log.info("Starting Twitch IRC reader for %s…", self.channel)
        while not self._stop_event.is_set():
            try:
                self._connect_and_read()
            except Exception as exc:
                if not self._stop_event.is_set():
                    log.warning("Twitch IRC error (%s). Reconnecting in 5 s…", exc)
                    time.sleep(5)

    def _connect_and_read(self) -> None:
        nick = f"justinfan{random.randint(10000, 99999)}"

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(300)  # 5-minute read timeout before we consider it dead
        sock.connect((TWITCH_HOST, TWITCH_PORT))
        self._sock = sock

        def send(msg: str) -> None:
            sock.send((msg + "\r\n").encode("utf-8"))

        send(f"PASS SCHMOOPIIE")
        send(f"NICK {nick}")
        send(f"JOIN {self.channel}")
        log.info("Twitch IRC: connected as %s, joined %s", nick, self.channel)

        buffer = ""
        while not self._stop_event.is_set():
            try:
                data = sock.recv(4096).decode("utf-8", errors="ignore")
            except socket.timeout:
                log.warning("Twitch IRC: receive timeout — reconnecting")
                break
            if not data:
                log.warning("Twitch IRC: connection closed by server")
                break

            buffer += data
            lines = buffer.split("\r\n")
            buffer = lines.pop()   # incomplete last line

            for line in lines:
                if line.startswith("PING"):
                    send("PONG :tmi.twitch.tv")
                    continue
                match = _PRIVMSG_RE.match(line)
                if match:
                    username = match.group(1)
                    text     = match.group(3).strip()
                    self.messages.append(ChatMessage(
                        timestamp=time.time(),
                        platform="twitch",
                        username=username,
                        text=text,
                    ))
                    log.debug("[Twitch] %s: %s", username, text)

        sock.close()
        self._sock = None

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=10)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_chat_reader(
    platform: str,
    stream_url: str,
    twitch_channel: Optional[str] = None,
) -> ChatReader:
    """
    Returns the appropriate ChatReader for the given platform.

    Parameters
    ----------
    platform       : "youtube" | "twitch"
    stream_url     : Full stream URL (used for YouTube)
    twitch_channel : Twitch channel name (required when platform="twitch")
    """
    if platform == "youtube":
        return YouTubeChatReader(stream_url)
    elif platform == "twitch":
        if not twitch_channel:
            raise ValueError("twitch_channel is required for Twitch streams.")
        return TwitchChatReader(twitch_channel)
    else:
        raise ValueError(f"Unknown platform: {platform!r}")
