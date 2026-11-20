"""
title_generator.py — GTAClipper Title Generator
=================================================
Derives a short clip title from recent chat messages without any API key.

Algorithm
---------
1. Collect the last TITLE_CHAT_WINDOW messages at the spike moment.
2. Clean each message: strip URLs, bot commands (!command), emoji-only lines,
   and very short noise words.
3. Score candidates in two passes:
   a. Single-word hype words: short, ALL-CAPS tokens (e.g. "WHAT", "PogChamp",
      "LETSGO", "W", "OMG") — weighted by frequency.
   b. Repeated short phrases (2–5 word n-grams) — scored by occurrence count.
4. Return the highest-scoring candidate, capped at TITLE_MAX_CHARS.
   Fall back to "Hype Moment" if nothing passes minimum thresholds.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Sequence

import config
from chat_reader import ChatMessage

# Tokens to skip entirely
_SKIP_TOKENS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "that", "this", "is", "it", "be", "was", "he", "she",
    "they", "we", "i", "my", "you", "your", "just", "like", "so", "get",
    "got", "lol", "lmao", "lmfao", "xd", "d", "s", "t", "m", "ll", "ve",
    "re", "im", "its",
})

# Known high-value hype tokens (boost their score)
_HYPE_BOOST: frozenset[str] = frozenset({
    "pog", "pogchamp", "poggers", "omg", "wtf", "letsgo", "lets go",
    "what", "no way", "clip it", "clip", "fire", "goat", "gg", "w",
    "based", "diff", "insane", "bro", "holy", "crazy", "aim",
    "highlight", "clutch", "one shot", "ez",
})

_URL_RE     = re.compile(r"https?://\S+")
_CMD_RE     = re.compile(r"^!\w+")          # bot commands
_EMOJI_RE   = re.compile(
    r"[\U0001F600-\U0001FFFF"
    r"\U00002702-\U000027B0"
    r"\U0001F1E0-\U0001F1FF]+",
    re.UNICODE,
)


def _clean(text: str) -> str:
    """Remove URLs, bot commands, and strip punctuation."""
    text = _URL_RE.sub("", text)
    text = _CMD_RE.sub("", text)
    # Keep emojis but remove other non-printable
    text = text.strip()
    return text


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, return word tokens."""
    text = text.lower()
    # Remove punctuation except apostrophes inside words
    text = re.sub(r"[^\w\s']", " ", text)
    return [t for t in text.split() if t not in _SKIP_TOKENS and len(t) > 1]


def _ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def generate_title(messages: Sequence[ChatMessage]) -> str:
    """
    Derive a clip title from a sequence of ChatMessage objects.
    Returns a string of at most config.TITLE_MAX_CHARS characters.
    """
    if not messages:
        return "Hype Moment"

    # Take the last TITLE_CHAT_WINDOW messages
    subset = list(messages)[-config.TITLE_CHAT_WINDOW:]

    cleaned_texts: list[str] = []
    for m in subset:
        text = _clean(m.text)
        # Skip emoji-only lines
        if not _EMOJI_RE.sub("", text).strip():
            continue
        # Skip very short messages (single char reactions like "!")
        if len(text.strip()) < 2:
            continue
        cleaned_texts.append(text)

    if not cleaned_texts:
        return "Hype Moment"

    # ── Pass 1: single-word hype scoring ──────────────────────────────────
    all_tokens: list[str] = []
    for t in cleaned_texts:
        all_tokens.extend(_tokenise(t))

    word_counts = Counter(all_tokens)

    # Score words: frequency × hype boost × length bonus for short caps words
    def word_score(word: str, count: int) -> float:
        score = float(count)
        if word in _HYPE_BOOST:
            score *= 3.0
        # Raw chat often uses ALL CAPS for hype
        raw_occurrences = sum(
            1 for t in cleaned_texts
            if re.search(rf"\b{re.escape(word)}\b", t, re.IGNORECASE)
            and word.upper() in t
        )
        score += raw_occurrences * 1.5
        return score

    scored_words = {
        w: word_score(w, c)
        for w, c in word_counts.items()
        if c >= 2 and len(w) >= 2
    }

    # ── Pass 2: n-gram phrase scoring ─────────────────────────────────────
    phrase_counts: Counter[str] = Counter()
    for t in cleaned_texts:
        toks = _tokenise(t)
        for n in (2, 3, 4):
            phrase_counts.update(_ngrams(toks, n))

    scored_phrases = {
        phrase: count * (len(phrase.split()) * 0.8)
        for phrase, count in phrase_counts.items()
        if count >= 2
    }

    # ── Combine & pick best ───────────────────────────────────────────────
    all_candidates: dict[str, float] = {}
    all_candidates.update(scored_words)
    all_candidates.update(scored_phrases)

    if not all_candidates:
        return "Hype Moment"

    best = max(all_candidates, key=lambda k: all_candidates[k])

    # Title-case the result
    title = " ".join(w.capitalize() for w in best.split())
    title = title[: config.TITLE_MAX_CHARS]
    return title or "Hype Moment"


# ── Quick self-test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    from chat_reader import ChatMessage
    import time

    sample_messages = [
        ChatMessage(time.time(), "twitch", "user1", "CLIP IT CLIP IT"),
        ChatMessage(time.time(), "twitch", "user2", "clip it bro"),
        ChatMessage(time.time(), "twitch", "user3", "WHAT WAS THAT"),
        ChatMessage(time.time(), "twitch", "user4", "clip it"),
        ChatMessage(time.time(), "twitch", "user5", "POG"),
        ChatMessage(time.time(), "twitch", "user6", "WHAT"),
        ChatMessage(time.time(), "twitch", "user7", "POG POG POG"),
        ChatMessage(time.time(), "twitch", "user8", "insane aim bro"),
        ChatMessage(time.time(), "twitch", "user9", "clip it right now"),
        ChatMessage(time.time(), "twitch", "user10", "CLIP IT"),
    ]
    print("Title:", generate_title(sample_messages))
