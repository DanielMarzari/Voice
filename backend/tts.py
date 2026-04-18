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
        speaker_name: Optional[str] = None,
    ) -> SynthesisResult:
        """Generate audio for `text`.

        XTTS uses `speaker_name` (a built-in speaker from its catalog) if
        provided; otherwise clones from `ref_audio_path`. F5 always uses
        `ref_audio_path` and ignores `speaker_name`.
        """

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
        speaker_name: Optional[str] = None,  # F5 ignores this
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

    # XTTS ships ~58 built-in speakers, explicitly gendered — we expose
    # them for the Design tab so users can pick a female or male voice
    # without having to bring their own reference clip.
    def list_speakers(self) -> list[dict]:
        self._load()
        speakers: list[str] = list(getattr(self._model, "speakers", []) or [])
        # Best-effort gender inference. XTTS doesn't ship gender labels
        # machine-readable; these are known manually. We fall back to
        # "unknown" for anything not in the table.
        known_female = {
            "Claribel Dervla", "Daisy Studious", "Gracie Wise", "Tammie Ema",
            "Alison Dietlinde", "Ana Florence", "Annmarie Nele", "Asya Anara",
            "Brenda Stern", "Gitta Nikolina", "Henriette Usha", "Sofia Hellen",
            "Tammy Grit", "Tanja Adelina", "Vjollca Johnnie", "Nova Hogarth",
            "Maja Ruoho", "Uta Obando", "Lidiya Szekeres", "Chandra MacFarland",
            "Szofi Granger", "Camilla Holmström", "Lilya Stainthorpe",
            "Zofija Kendrick", "Narelle Moon", "Barbora MacLean", "Alexandra Hisakawa",
            "Alma María", "Rosemary Okafor", "Ige Behringer", "Filip Traverse",
        }
        known_male = {
            "Damien Black", "Ferran Simen", "Viktor Eka", "Luis Moray",
            "Baldur Sanjin", "Craig Gutsy", "Marcos Rudaski", "Wulf Carlevaro",
            "Eugenio Mataracı", "Kumar Dahl", "Dionisio Schuyler", "Royston Min",
            "Abrahan Mack", "Adde Michal", "Badr Odhiambo", "Dionisio Schuyler",
            "Gilberto Mathias", "Ilkin Urbano", "Kazuhiko Atallah", "Ludvig Milivoj",
            "Suad Qasim", "Torcull Diarmuid", "Viktor Menelaos", "Zacharie Aimilios",
            "Nova Hogarth", "Maja Ruoho", "Andrew Chipper", "Aaron Dreschner",
        }
        out = []
        for name in speakers:
            if name in known_female:
                g = "female"
            elif name in known_male:
                g = "male"
            else:
                g = "unknown"
            out.append({"name": name, "gender": g})
        return out

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
        speaker_name: Optional[str] = None,
    ) -> SynthesisResult:
        self._load()
        # Prefer a named built-in speaker when given — that way we get
        # actual different voices (male/female, etc.) instead of the same
        # cloned reference every time. Falls back to ref clip cloning.
        kwargs = {
            "text": text,
            "language": language,
            "speed": speed,
        }
        if speaker_name:
            kwargs["speaker"] = speaker_name
        else:
            kwargs["speaker_wav"] = ref_audio_path
        wav = self._model.tts(**kwargs)  # type: ignore[union-attr]
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
#
# Zero-shot cloning needs a reference clip. Rather than asking the user
# to supply 4 distinct WAVs by hand (the Phase-1 design that bit us), we
# now use a single shared "neutral" reference clip and let each preset
# differ only in its default slider suggestions. The reference is resolved
# in priority order:
#
#   1. Custom override at data/presets/base.wav  (user can swap in their own)
#   2. F5-TTS's bundled example clip (ships with `pip install f5-tts`)
#   3. Auto-downloaded from the F5-TTS GitHub repo (Apache 2.0)
#
# Each preset's "flavor" is entirely driven by its default_pitch / default_speed
# on the frontend sliders — the user can always nudge from there.

PRESET_DIR = Path(__file__).parent / "data" / "presets"
PRESET_DIR.mkdir(parents=True, exist_ok=True)

_FALLBACK_REF_URL = (
    "https://raw.githubusercontent.com/SWivid/F5-TTS/main/"
    "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
)
_FALLBACK_REF_TEXT = (
    "Some call me nature. Others call me mother nature."
)

PRESETS = [
    {
        "id": "amber",
        "label": "Amber (warm, bright)",
        "default_pitch": 1.0,
        "default_speed": 1.0,
        "default_temperature": 0.8,
    },
    {
        "id": "cobalt",
        "label": "Cobalt (calm, low)",
        "default_pitch": -2.0,
        "default_speed": 0.95,
        "default_temperature": 0.6,
    },
    {
        "id": "rose",
        "label": "Rose (soft, expressive)",
        "default_pitch": 2.0,
        "default_speed": 0.95,
        "default_temperature": 0.95,
    },
    {
        "id": "slate",
        "label": "Slate (neutral news)",
        "default_pitch": 0.0,
        "default_speed": 1.05,
        "default_temperature": 0.5,
    },
]


def _f5_bundled_example() -> Optional[Path]:
    """Try to find F5-TTS's shipped reference WAV inside the installed package."""
    try:
        import f5_tts  # type: ignore
        from importlib.resources import files as pkg_files

        # F5's canonical example. Path changed across versions — try a few.
        candidates = [
            pkg_files(f5_tts) / "infer" / "examples" / "basic" / "basic_ref_en.wav",
            pkg_files(f5_tts) / "infer" / "examples" / "basic_ref_en.wav",
        ]
        for c in candidates:
            try:
                if c.is_file():
                    return Path(str(c))
            except Exception:
                continue
    except Exception:
        pass
    return None


def _download_fallback_ref(dest: Path) -> None:
    """Download a public-domain reference clip from the F5-TTS repo (Apache 2.0)."""
    import urllib.request

    log.info("Downloading fallback reference clip → %s", dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_FALLBACK_REF_URL, timeout=30) as r:
        data = r.read()
    dest.write_bytes(data)
    log.info("Saved %d bytes to %s", len(data), dest)


def _resolve_base_reference() -> tuple[str, str]:
    """Return (path, ref_text) for the shared base reference clip."""
    # 1. User-provided override.
    custom = PRESET_DIR / "base.wav"
    if custom.exists():
        # Honor a per-file transcript if the user dropped `base.txt` alongside.
        txt = PRESET_DIR / "base.txt"
        ref_text = txt.read_text().strip() if txt.exists() else ""
        return str(custom), ref_text

    # 2. F5-TTS's bundled example.
    bundled = _f5_bundled_example()
    if bundled is not None:
        return str(bundled), _FALLBACK_REF_TEXT

    # 3. Download once and cache.
    cached = PRESET_DIR / "fallback_ref_en.wav"
    if not cached.exists():
        try:
            _download_fallback_ref(cached)
        except Exception as e:
            raise FileNotFoundError(
                f"No reference clip available. Tried custom ({custom}), "
                f"F5-TTS bundled example, and download from {_FALLBACK_REF_URL}. "
                f"Error: {e}. Drop a ~5s WAV at {custom} as a workaround."
            )
    return str(cached), _FALLBACK_REF_TEXT


def resolve_preset(preset_id: str) -> tuple[str, str]:
    """Return (ref_audio_path, ref_text). All presets share one reference
    clip — their 'personality' is expressed via the slider defaults the
    frontend applies. Keeps setup friction to zero."""
    if not any(p["id"] == preset_id for p in PRESETS):
        raise ValueError(f"Unknown preset id: {preset_id}")
    return _resolve_base_reference()


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
