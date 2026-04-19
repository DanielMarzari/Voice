"""
Python port of Reader's src/lib/ttsChunker.ts.

Same heuristics: 180-250 char target (30-47 words), 320 hard max, split
priority = paragraph break → sentence end → clause break → whitespace →
hard cut. Chunks carry absolute char offsets into the original document
text so Reader's player can re-derive word boundaries using its own
tokenizer — no need to duplicate tokenization on the Python side.

Keep this file logically in sync with ttsChunker.ts. If one side diverges,
sentence boundaries may drift and Reader's highlight won't align with the
synthesized audio.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


TARGET_MIN_CHARS = 180
TARGET_MAX_CHARS = 250
HARD_MAX_CHARS = 320

_TRAILING_QUOTE_RE = re.compile(r"[\"')\]\u201d\u2019\s]")
_SOFT_WHITESPACE_RE = re.compile(r"\s")
_CONSOLIDATE_WS_RE = re.compile(r"\s+")


@dataclass
class Chunk:
    """One synthesis unit. `index` is 1-based (matches the on-disk
    chunk_NNNN.mp3 naming and the manifest). `speak_text` is the trimmed,
    whitespace-normalized text sent to the TTS engine. `char_start`/`char_end`
    are absolute offsets into the original document text — Reader's player
    maps those to word indices for per-word highlight."""
    index: int
    text: str
    speak_text: str
    char_start: int
    char_end: int


def _is_sentence_end(ch: str) -> bool:
    return ch in ".!?"


def _is_clause_break(ch: str) -> bool:
    return ch in ",;:"


def _find_split(source: str, from_: int, to: int) -> int:
    """Return the exclusive end offset for the first chunk in [from, to).
    Mirrors ttsChunker.ts `findSplit`."""
    remaining = to - from_
    if remaining <= TARGET_MAX_CHARS:
        return to

    window_end = min(to, from_ + HARD_MAX_CHARS)

    # 1. Paragraph break after the min threshold.
    para_idx = source.find("\n\n", from_ + TARGET_MIN_CHARS)
    if para_idx != -1 and para_idx < window_end:
        return para_idx + 2

    soft_end = min(to, from_ + TARGET_MAX_CHARS)

    # 2. Sentence end within preferred window.
    for i in range(soft_end - 1, from_ + TARGET_MIN_CHARS - 1, -1):
        if _is_sentence_end(source[i]):
            j = i + 1
            while j < to and _TRAILING_QUOTE_RE.match(source[j]):
                j += 1
            return j

    # 3. Sentence end within hard window.
    for i in range(window_end - 1, from_ + TARGET_MIN_CHARS - 1, -1):
        if _is_sentence_end(source[i]):
            j = i + 1
            while j < to and _TRAILING_QUOTE_RE.match(source[j]):
                j += 1
            return j

    # 4. Clause break within hard window.
    for i in range(window_end - 1, from_ + TARGET_MIN_CHARS - 1, -1):
        if _is_clause_break(source[i]):
            j = i + 1
            while j < to and _SOFT_WHITESPACE_RE.match(source[j]):
                j += 1
            return j

    # 5. Whitespace boundary near softEnd.
    for i in range(soft_end, window_end):
        if _SOFT_WHITESPACE_RE.match(source[i]):
            return i + 1
    for i in range(soft_end - 1, from_, -1):
        if _SOFT_WHITESPACE_RE.match(source[i]):
            return i + 1

    # 6. Hard cut.
    return window_end


def chunk_document(content: str, start_char_offset: int = 0) -> list[Chunk]:
    """Split the document into sequential chunks starting at
    `start_char_offset`. Paragraph-aware (no chunk crosses a blank line).
    """
    content_len = len(content)
    clamped_start = max(0, min(content_len, int(start_char_offset)))
    if clamped_start >= content_len:
        return []

    # Paragraph spans — non-empty runs separated by \n\n+.
    paras: list[tuple[int, int]] = []
    for m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", content):
        paras.append((m.start(), m.end()))
    if not paras:
        paras.append((0, content_len))

    # Trim to the start offset.
    active: list[tuple[int, int]] = []
    for p_start, p_end in paras:
        if p_end <= clamped_start:
            continue
        active.append((max(p_start, clamped_start), p_end))

    chunks: list[Chunk] = []
    idx = 1
    for p_start, p_end in active:
        cursor = p_start
        while cursor < p_end:
            split_at = _find_split(content, cursor, p_end)
            char_start = cursor
            char_end = min(split_at, p_end)
            slice_text = content[char_start:char_end]
            speak_text = _CONSOLIDATE_WS_RE.sub(" ", slice_text).strip()
            if speak_text:
                chunks.append(
                    Chunk(
                        index=idx,
                        text=slice_text,
                        speak_text=speak_text,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
                idx += 1
            cursor = char_end

    return chunks


def previous_context(content: str, chunk: Chunk, max_chars: int = 500) -> str:
    """Text immediately before a chunk, for the TTS model's prosody
    continuity. Whitespace-normalized, length-capped."""
    start = max(0, chunk.char_start - max_chars)
    prev = content[start:chunk.char_start]
    return _CONSOLIDATE_WS_RE.sub(" ", prev).strip()


def forward_context(chunks: list[Chunk], current_idx: int, max_chars: int = 500) -> str:
    """Speak-text of the NEXT chunk (1-based `current_idx` lookup), for
    trailing intonation planning."""
    # current_idx is the 1-based position; the list is 0-indexed.
    nxt = chunks[current_idx] if current_idx < len(chunks) else None
    if nxt is None:
        return ""
    return nxt.speak_text[:max_chars]
