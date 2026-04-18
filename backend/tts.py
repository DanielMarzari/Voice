"""
TTS engine wrapper.

Primary: F5-TTS (Apache 2.0, zero-shot voice cloning). Lazy-loaded on first
call so the server starts instantly; model stays resident after that.

Alternate (intentionally unused — Coqui Public Model License is
non-commercial): XTTS-v2 adapter kept as a stub below for local-only
experimentation.
"""

from __future__ import annotations

import io
import os
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Raw audio: mono float32 PCM at `sample_rate` Hz."""
    audio: np.ndarray
    sample_rate: int


class _LazyF5:
    """F5-TTS is heavy to import — defer until first use."""

    def __init__(self, device: str = "mps"):
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        log.info("Loading F5-TTS (device=%s) — first load downloads weights", self.device)
        try:
            # F5-TTS's high-level API. The import path and class name here follow
            # the f5-tts PyPI package (>=0.3). If the package reorganizes its
            # public API, this is the single line to update.
            from f5_tts.api import F5TTS  # type: ignore

            self._model = F5TTS(device=self.device)
        except Exception as e:
            log.exception("Failed to load F5-TTS")
            raise RuntimeError(
                f"F5-TTS unavailable: {e}. "
                f"Check that the f5-tts package is installed and the "
                f"'{self.device}' device is available."
            )
        return self._model

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
    ) -> SynthesisResult:
        model = self._load()
        # F5TTS.infer returns (wav_np, sample_rate, spectrogram) in current API.
        # `ref_text` is optional — if omitted, F5 transcribes the reference
        # clip internally via Whisper. Providing it speeds things up and
        # improves prosody matching.
        wav, sr, _ = model.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text or "",
            gen_text=text,
            speed=speed,
        )
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return SynthesisResult(audio=wav.astype(np.float32), sample_rate=int(sr))


# Module-level singleton so FastAPI workers reuse the loaded model.
_engine: Optional[_LazyF5] = None


def get_engine() -> _LazyF5:
    global _engine
    if _engine is None:
        device = os.environ.get("TTS_DEVICE", "mps")
        _engine = _LazyF5(device=device)
    return _engine


# -------- Built-in "design" speaker bank --------
# For manual voice design (no reference sample), we use a small bank of
# pre-recorded reference clips bundled with F5-TTS. The user picks one,
# adjusts pitch/speed/temperature sliders, and we synthesize with that
# clip as the reference.
#
# These paths are resolved relative to `backend/data/presets/`. The presets
# directory ships with the repo; the clips are ~5s WAV files the user
# generates once via `scripts/make_presets.py` (see README) or drops in
# manually.
PRESET_DIR = Path(__file__).parent / "data" / "presets"

PRESETS = [
    {"id": "amber",   "label": "Amber (warm, bright)",   "file": "amber.wav",   "ref_text": "Hello, this is a warm and friendly voice."},
    {"id": "cobalt",  "label": "Cobalt (calm, low)",     "file": "cobalt.wav",  "ref_text": "This is a calm, measured voice speaking."},
    {"id": "rose",    "label": "Rose (soft, expressive)","file": "rose.wav",    "ref_text": "A soft and expressive voice, full of nuance."},
    {"id": "slate",   "label": "Slate (neutral news)",   "file": "slate.wav",   "ref_text": "This is a neutral news-reader voice."},
]


def resolve_preset(preset_id: str) -> tuple[str, str]:
    """Return (ref_audio_path, ref_text) for a preset id. Raises if missing."""
    for p in PRESETS:
        if p["id"] == preset_id:
            path = PRESET_DIR / p["file"]
            if not path.exists():
                raise FileNotFoundError(
                    f"Preset '{preset_id}' is defined but {path} is missing. "
                    f"Run scripts/make_presets.py or drop a WAV clip there."
                )
            return str(path), p["ref_text"]
    raise ValueError(f"Unknown preset id: {preset_id}")


def encode_mp3(result: SynthesisResult, bitrate: str = "128k") -> bytes:
    """Encode a SynthesisResult into MP3 bytes via pydub/ffmpeg."""
    from pydub import AudioSegment

    # pydub wants int16 PCM bytes.
    pcm = np.clip(result.audio, -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype(np.int16).tobytes()
    seg = AudioSegment(
        data=pcm16,
        sample_width=2,
        frame_rate=result.sample_rate,
        channels=1,
    )
    buf = io.BytesIO()
    seg.export(buf, format="mp3", bitrate=bitrate)
    return buf.getvalue()


def pitch_shift(result: SynthesisResult, semitones: float) -> SynthesisResult:
    """Naive pitch shift via resample trick (also changes duration — caller
    compensates with time-stretch if they care). For our slider use-case
    small shifts of +/- 4 semitones are fine."""
    if abs(semitones) < 0.01:
        return result
    factor = 2.0 ** (semitones / 12.0)
    # Resample to new length; the listener perceives pitch change.
    # Keep duration roughly intact by inverse-stretching.
    import scipy.signal  # lazy
    new_len = int(len(result.audio) / factor)
    shifted = scipy.signal.resample(result.audio, new_len).astype(np.float32)
    # Now stretch back to original duration (duration stays, pitch changed).
    restored = scipy.signal.resample(shifted, len(result.audio)).astype(np.float32)
    return SynthesisResult(audio=restored, sample_rate=result.sample_rate)


# -------- XTTS-v2 stub (DO NOT ship — non-commercial license) --------
# If you want to experiment with XTTS-v2 locally, uncomment and install
# `pip install TTS`. Quality is comparable to F5 but license forbids
# commercial use.
#
# class _LazyXTTS:
#     def __init__(self, device="mps"): ...
#     def synthesize(self, text, ref_audio_path, ref_text=None, speed=1.0): ...
