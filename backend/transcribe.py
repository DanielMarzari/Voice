"""Shared Whisper + audio-probe helpers.

Factored out of backend/training.py so both Deep Clone (segment-level
transcription) and the Clone tab's zero-shot flow (single-file
transcription + duration probe) can share the same faster-whisper
model instance. Loading the model is the expensive part; both flows
go through this module so we only pay it once per process.

Why faster-whisper:
- ~4x faster than openai-whisper at equal accuracy
- Works on Apple Silicon via CTranslate2 CPU kernels (no MPS required)
- int8 compute_type halves memory + doubles throughput for speech <=30s
  with no measurable quality loss

Duration probe intentionally uses ffprobe via subprocess rather than
loading audio into Python — this keeps the Clone tab's upload-validate
path fast (tens of ms) even on a 500 MB audio file.
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Module-level cache; shared with training.py when both code paths
# run in the same FastAPI process. "base" is the sweet spot: ~150 MB
# on disk, 1–2 s per 6 s clip on CPU, equivalent quality to larger
# models on clean-speech prompts.
_WHISPER_MODEL: object | None = None
_WHISPER_MODEL_SIZE = "base"


class TranscribeError(RuntimeError):
    """Raised when Whisper is unavailable or the transcription fails."""


def probe_duration(path: str | Path) -> float:
    """Return the duration of an audio file in seconds.

    Uses ffprobe via subprocess so we don't have to load the file into
    Python. ffprobe ships with ffmpeg, which the Voice Studio install
    docs already require for MP3 encoding of samples.

    Raises:
        FileNotFoundError: if the path doesn't exist
        RuntimeError: if ffprobe is missing or fails
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(p),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "ffprobe not found on PATH — install ffmpeg (brew install ffmpeg)"
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffprobe timed out on {p}") from e
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[:200]}")
    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"Could not parse ffprobe output: {e}") from e


def _ensure_model():
    """Load the faster-whisper model once per process.

    Returns the model on success, None if faster-whisper isn't installed
    (in which case callers should fall back gracefully — a voice can
    still be created with a hand-entered transcript).
    """
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        log.warning(
            "faster-whisper not installed (%s); skipping auto-transcribe.", e
        )
        return None
    log.info("Loading faster-whisper (%s) for transcription…", _WHISPER_MODEL_SIZE)
    _WHISPER_MODEL = WhisperModel(
        _WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
    )
    return _WHISPER_MODEL


def transcribe_file(
    path: str | Path,
    language: str = "en",
    vad_filter: bool = True,
) -> Optional[str]:
    """Transcribe a short (<=30 s) audio file. Returns the stripped text,
    or None if the transcriber is unavailable.

    Raises TranscribeError if Whisper is installed but errors during
    transcription — caller decides whether to surface or fall back.
    """
    model = _ensure_model()
    if model is None:
        return None
    try:
        segs, _info = model.transcribe(  # type: ignore[union-attr]
            str(path), language=language, vad_filter=vad_filter
        )
        return " ".join(s.text.strip() for s in segs).strip()
    except Exception as e:
        raise TranscribeError(f"Whisper failed: {e}") from e
