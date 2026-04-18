"""
TTS engine registry.

Multiple zero-shot cloning engines are supported so the user can pick per
voice. All engines expose a common `synthesize(...) -> SynthesisResult`
interface and are lazy-loaded on first use — importing this module is
cheap. Loaded engines stay resident in the worker process.

Currently shipped:
  - "f5"   — F5-TTS (Apache 2.0, English + Chinese, 2024)
  - "xtts" — XTTS-v2 via Coqui `TTS` package (CPML non-commercial, 17 langs)

Adding a new engine = a new subclass of _Engine registered in ENGINES.
"""

from __future__ import annotations

import io
import os
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class SynthesisResult:
    """Raw audio: mono float32 PCM at `sample_rate` Hz."""
    audio: np.ndarray
    sample_rate: int


# ---------------- Engine base class ----------------


class _Engine(ABC):
    """Base class for a lazy-loaded TTS engine."""

    id: str = ""
    label: str = ""
    license: str = ""
    languages: list[str] = []
    default_device: str = "mps"

    def __init__(self, device: Optional[str] = None):
        self.device = device or os.environ.get("TTS_DEVICE", self.default_device)
        self._model = None
        self._available: Optional[bool] = None  # cached importability check

    def is_loaded(self) -> bool:
        return self._model is not None

    def is_available(self) -> bool:
        """Return True if the underlying python package is importable.
        Cheap — does not load weights."""
        if self._available is not None:
            return self._available
        try:
            self._check_importable()
            self._available = True
        except Exception as e:
            log.warning("Engine %s not available: %s", self.id, e)
            self._available = False
        return self._available

    @abstractmethod
    def _check_importable(self):
        """Raise ImportError if the engine's dependencies aren't installed."""

    @abstractmethod
    def _load(self):
        """Load model weights into self._model. Called once."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        language: str = "en",
    ) -> SynthesisResult:
        """Generate audio for `text`, conditioned on `ref_audio_path`."""

    def info(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "license": self.license,
            "languages": self.languages,
            "loaded": self.is_loaded(),
            "available": self.is_available(),
            "device": self.device,
        }


# ---------------- F5-TTS ----------------


class _F5Engine(_Engine):
    id = "f5"
    label = "F5-TTS"
    license = "Apache 2.0"
    languages = ["en", "zh"]

    def _check_importable(self):
        import f5_tts.api  # noqa: F401

    def _load(self):
        if self._model is not None:
            return
        log.info("Loading F5-TTS (device=%s) — first load downloads weights", self.device)
        try:
            from f5_tts.api import F5TTS  # type: ignore
            self._model = F5TTS(device=self.device)
        except Exception as e:
            log.exception("F5-TTS load failed")
            raise RuntimeError(f"F5-TTS unavailable: {e}")

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        language: str = "en",
    ) -> SynthesisResult:
        self._load()
        wav, sr, _ = self._model.infer(  # type: ignore[union-attr]
            ref_file=ref_audio_path,
            ref_text=ref_text or "",
            gen_text=text,
            speed=speed,
        )
        if wav.ndim > 1:
            wav = wav.mean(axis=-1)
        return SynthesisResult(audio=wav.astype(np.float32), sample_rate=int(sr))


# ---------------- XTTS-v2 (Coqui) ----------------
# Non-commercial license (CPML). Fine for personal use.


class _XTTSEngine(_Engine):
    id = "xtts"
    label = "XTTS-v2 (Coqui)"
    license = "CPML (non-commercial)"
    languages = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
        "cs", "ar", "zh", "ja", "hu", "ko", "hi",
    ]

    def _check_importable(self):
        import TTS.api  # noqa: F401

    def _load(self):
        if self._model is not None:
            return
        log.info("Loading XTTS-v2 (device=%s) — first load downloads ~1.8GB", self.device)

        # Coqui TTS needs this env var to auto-accept their license on download.
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        try:
            from TTS.api import TTS  # type: ignore
        except Exception as e:
            raise RuntimeError(
                f"XTTS unavailable: couldn't import Coqui `TTS` package ({e}). "
                f"Install with: pip install TTS>=0.22"
            )

        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        try:
            tts = TTS(model_name)
            # MPS on Apple Silicon is flaky for some ops in XTTS — fall back to CPU.
            try:
                tts = tts.to(self.device)
            except Exception as e:
                log.warning(
                    "XTTS .to(%s) failed (%s) — falling back to CPU",
                    self.device, e,
                )
                tts = tts.to("cpu")
                self.device = "cpu"
            self._model = tts
        except Exception as e:
            log.exception("XTTS load failed")
            raise RuntimeError(f"XTTS-v2 load failed: {e}")

    def synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
        language: str = "en",
    ) -> SynthesisResult:
        self._load()
        # XTTS accepts speaker_wav (path or list), language, and returns float32 numpy.
        wav = self._model.tts(  # type: ignore[union-attr]
            text=text,
            speaker_wav=ref_audio_path,
            language=language,
            speed=speed,
        )
        wav_np = np.asarray(wav, dtype=np.float32)
        if wav_np.ndim > 1:
            wav_np = wav_np.mean(axis=-1)
        # XTTS always outputs 24 kHz.
        return SynthesisResult(audio=wav_np, sample_rate=24000)


# ---------------- Registry ----------------

ENGINES: dict[str, _Engine] = {
    "f5":   _F5Engine(),
    "xtts": _XTTSEngine(),
}

DEFAULT_ENGINE = os.environ.get("TTS_ENGINE_DEFAULT", "xtts")


def get_engine(engine_id: Optional[str] = None) -> _Engine:
    key = (engine_id or DEFAULT_ENGINE).lower()
    if key not in ENGINES:
        raise ValueError(
            f"Unknown engine '{key}'. Available: {', '.join(ENGINES.keys())}"
        )
    return ENGINES[key]


def list_engines() -> list[dict]:
    return [e.info() for e in ENGINES.values()]


# ---------------- Design preset speakers ----------------

PRESET_DIR = Path(__file__).parent / "data" / "presets"

PRESETS = [
    {"id": "amber",  "label": "Amber (warm, bright)",    "file": "amber.wav",  "ref_text": "Hello, this is a warm and friendly voice."},
    {"id": "cobalt", "label": "Cobalt (calm, low)",      "file": "cobalt.wav", "ref_text": "This is a calm, measured voice speaking."},
    {"id": "rose",   "label": "Rose (soft, expressive)", "file": "rose.wav",   "ref_text": "A soft and expressive voice, full of nuance."},
    {"id": "slate",  "label": "Slate (neutral news)",    "file": "slate.wav",  "ref_text": "This is a neutral news-reader voice."},
]


def resolve_preset(preset_id: str) -> tuple[str, str]:
    for p in PRESETS:
        if p["id"] == preset_id:
            path = PRESET_DIR / p["file"]
            if not path.exists():
                raise FileNotFoundError(
                    f"Preset '{preset_id}' is defined but {path} is missing. "
                    f"Drop a WAV clip there (see backend/data/presets/README.md)."
                )
            return str(path), p["ref_text"]
    raise ValueError(f"Unknown preset id: {preset_id}")


# ---------------- Audio helpers ----------------


def encode_mp3(result: SynthesisResult, bitrate: str = "128k") -> bytes:
    """Encode a SynthesisResult into MP3 bytes via pydub/ffmpeg."""
    from pydub import AudioSegment

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
    """Simple pitch shift that preserves duration. Used for manual voice
    design where the slider nudges the preset's natural pitch."""
    if abs(semitones) < 0.01:
        return result
    factor = 2.0 ** (semitones / 12.0)
    import scipy.signal  # lazy
    new_len = int(len(result.audio) / factor)
    shifted = scipy.signal.resample(result.audio, new_len).astype(np.float32)
    restored = scipy.signal.resample(shifted, len(result.audio)).astype(np.float32)
    return SynthesisResult(audio=restored, sample_rate=result.sample_rate)
