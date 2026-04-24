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
        self._install_wandb_stub()
        import f5_tts.api  # noqa: F401

    @staticmethod
    def _install_wandb_stub():
        """Stub out `wandb` before f5_tts imports it.

        f5_tts/model/trainer.py runs `import wandb` at module load time,
        and `f5_tts/model/__init__.py` imports the trainer eagerly — so
        even inference-only use drags wandb in. Our venv has wandb 0.26.x
        whose generated `.pb2` files can't be parsed by the protobuf
        3.20.2 that Coqui TTS pins, which bricks F5 with:
            ImportError: cannot import name 'Imports' from
            'wandb.proto.wandb_telemetry_pb2'
        Since F5 inference never calls any wandb API (only the trainer
        does, and we don't train in-process), a no-op stub lets the
        import succeed. If someone tries to train via f5_tts on this
        interpreter later, they'll get clear AttributeErrors from the
        stub rather than silent garbage runs.
        """
        import sys
        if "wandb" in sys.modules and not getattr(
            sys.modules["wandb"], "_voice_studio_stub", False
        ):
            return  # real wandb already imported OK, don't clobber it

        class _WandbStub:
            _voice_studio_stub = True
            def __getattr__(self, name):
                return _WandbStub()
            def __call__(self, *args, **kwargs):
                return _WandbStub()

        sys.modules["wandb"] = _WandbStub()  # type: ignore[assignment]

    def _load(self):
        if self._model is not None:
            return
        log.info("Loading F5-TTS (device=%s) — first load downloads weights", self.device)
        self._install_wandb_stub()
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

        # PyTorch 2.6+ flipped `torch.load`'s default to `weights_only=True`,
        # which refuses to unpickle arbitrary objects. Coqui TTS 0.22.0 still
        # calls `torch.load` without the flag, so the XTTS checkpoint load
        # raises `_pickle.UnpicklingError: Weights only load failed`.
        #
        # The "proper" fix is allowlisting each custom class via
        # `torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig,
        # XttsArgs, BaseDatasetConfig, ...])`, but the exact class list
        # shifts across XTTS model revisions and every miss means another
        # round-trip. Since the checkpoints come from the Coqui HuggingFace
        # mirror (trusted) rather than untrusted user uploads, it's safe to
        # disable the gate globally for `torch.load` — the safe-loading
        # default exists to protect servers accepting arbitrary .pt uploads,
        # not apps loading their own model cache. See discussion:
        # github.com/coqui-ai/TTS/issues/3877
        import torch
        if not getattr(torch.load, "_xtts_weights_only_patched", False):
            _orig_load = torch.load

            def _patched_load(*args, **kwargs):  # type: ignore[no-redef]
                kwargs.setdefault("weights_only", False)
                return _orig_load(*args, **kwargs)

            _patched_load._xtts_weights_only_patched = True  # type: ignore[attr-defined]
            torch.load = _patched_load  # type: ignore[assignment]

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

# Default to F5 — zero-shot identity transfer is noticeably closer to the
# reference clip than XTTS's (especially on short 5–10 s prompts), and F5 is
# Apache-2.0 vs XTTS's non-commercial CPML license. XTTS stays available in
# the picker for the 15 non-EN/ZH languages it supports. Override with
# `TTS_ENGINE_DEFAULT=xtts` if you want the old behavior.
DEFAULT_ENGINE = os.environ.get("TTS_ENGINE_DEFAULT", "f5")


def get_engine(engine_id: Optional[str] = None) -> _Engine:
    key = (engine_id or DEFAULT_ENGINE).lower()
    if key not in ENGINES:
        raise ValueError(
            f"Unknown engine '{key}'. Available: {', '.join(ENGINES.keys())}"
        )
    return ENGINES[key]


def list_engines() -> list[dict]:
    return [e.info() for e in ENGINES.values()]


# ---------------- Voice-profile → synthesis inputs ----------------
#
# Shared by the /api/synthesize route (not yet used by Reader) and the
# render worker. Resolves a saved profile to (ref_path, ref_text,
# speaker_name, engine_id) — everything `engine.synthesize(...)` needs.


def resolve_voice_for_synth(profile) -> tuple[str, Optional[str], Optional[str], str]:
    """Look up a VoiceProfile and return the inputs needed to synthesize
    arbitrary text in that voice. Supports 'cloned' and 'designed' kinds.
    'uploaded' kinds have no engine attached — caller must handle before
    calling this helper.

    Returns (ref_audio_path, ref_text, speaker_name, engine_id).
    """
    if profile.kind == "uploaded":
        raise ValueError(
            "Uploaded voices are static audio only — they can't synthesize "
            "new text. Use Clone or Design to get a synthesizable voice."
        )

    # Normalize legacy engine strings (e.g. 'f5-tts' → 'f5').
    engine_id = (profile.engine or DEFAULT_ENGINE)
    engine_id = {"f5-tts": "f5"}.get(engine_id, engine_id)

    if profile.kind == "cloned":
        # Cloned voices store source.<ext> in their profile dir. Accept any ext.
        dir_ = profile.dir()
        sources = [p for p in dir_.iterdir() if p.stem == "source" and p.is_file()]
        if not sources:
            raise FileNotFoundError(
                f"Cloned profile {profile.id} is missing its source reference clip"
            )
        return str(sources[0]), None, None, engine_id

    if profile.kind == "designed":
        design = profile.design or {}
        speaker_name = design.get("speaker_name")  # XTTS-only, optional
        base_voice = design.get("base_voice")
        if not base_voice:
            raise ValueError(
                f"Designed profile {profile.id} missing design.base_voice"
            )
        ref_path, ref_text = resolve_preset(base_voice)
        return ref_path, ref_text, speaker_name, engine_id

    raise ValueError(f"Unknown voice kind: {profile.kind}")


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
USER_REF_DIR = PRESET_DIR / "user"
PRESET_DIR.mkdir(parents=True, exist_ok=True)
USER_REF_DIR.mkdir(parents=True, exist_ok=True)

_FALLBACK_REF_URL = (
    "https://raw.githubusercontent.com/SWivid/F5-TTS/main/"
    "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
)
_FALLBACK_REF_TEXT = (
    "Some call me nature. Others call me mother nature."
)

# Built-in mood presets — slider defaults only. The underlying reference
# clip is shared (auto-downloaded fallback) so these are purely about
# starting-slider positions. For distinct voices (female / male / other
# character), users drop WAVs into data/presets/user/ via the Design tab's
# "Upload reference" button.
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
    """Return (ref_audio_path, ref_text) for a base-voice dropdown entry.

    Accepts four namespaces:
      * "amber" / "cobalt" / "rose" / "slate" — built-in mood presets.
        All share a single reference clip (auto-downloaded / F5 bundled);
        they differ only in slider defaults. Call _resolve_base_reference.
      * "user:<slug>" — user-uploaded reference under data/presets/user/.
        Looks up backend/data/presets/user/<slug>.wav (+ optional .txt).
      * "deep:<slug>-<shortid>" — a Deep Clone fine-tuned voice. We route
        through training.resolve_deep_clone() which currently returns the
        first training segment as the reference clip (until the engines
        grow a checkpoint-loading hook).
    """
    if preset_id.startswith("deep:"):
        # Deferred import — training.py imports nothing heavy, but keeping
        # this lazy avoids any chance of circular-import headaches.
        import training  # type: ignore
        return training.resolve_deep_clone(preset_id)

    if preset_id.startswith("user:"):
        slug = preset_id.split(":", 1)[1]
        wav = USER_REF_DIR / f"{slug}.wav"
        if not wav.exists():
            raise FileNotFoundError(
                f"User reference '{slug}' not found at {wav}. "
                f"Re-upload it from the Design tab."
            )
        txt_path = USER_REF_DIR / f"{slug}.txt"
        ref_text = txt_path.read_text().strip() if txt_path.exists() else ""
        return str(wav), ref_text

    if not any(p["id"] == preset_id for p in PRESETS):
        raise ValueError(f"Unknown preset id: {preset_id}")
    return _resolve_base_reference()


def _read_ref_meta(wav_path: Path) -> dict:
    """Read the sidecar .json for a user reference, if present.
    Holds human-readable name + description (since the filesystem slug
    is lowercase/hyphenated)."""
    meta_path = wav_path.with_suffix(".json")
    if meta_path.exists():
        try:
            import json as _json
            return _json.loads(meta_path.read_text())
        except Exception:
            pass
    # Fallback: derive name from filename.
    return {
        "name": wav_path.stem.replace("_", " ").replace("-", " ").strip(),
        "description": "",
    }


def list_user_references() -> list[dict]:
    """List WAVs dropped into data/presets/user/, for the Design tab dropdown."""
    if not USER_REF_DIR.exists():
        return []
    out = []
    for p in sorted(USER_REF_DIR.iterdir()):
        if not p.is_file() or p.suffix.lower() != ".wav":
            continue
        meta = _read_ref_meta(p)
        # Display label: "Alex (friendly, inviting and balanced)" when a
        # description is present; otherwise just the name.
        name = meta.get("name") or p.stem.replace("_", " ").replace("-", " ")
        desc = (meta.get("description") or "").strip()
        label = f"{name} ({desc})" if desc else name
        out.append({
            "id": f"user:{p.stem}",
            "label": label,
            "name": name,
            "description": desc,
            "path": str(p),
        })
    return out


def save_user_reference(
    name: str,
    audio_bytes: bytes,
    description: str = "",
    transcript: str = "",
) -> dict:
    """Save an uploaded audio clip as a new user reference.

    Converts whatever format the user gave us (WAV / MP3 / M4A / ...) to
    24 kHz mono WAV via pydub/ffmpeg — F5 and XTTS both want WAV at that
    rate. Returns the preset-descriptor dict matching list_user_references().
    """
    from pydub import AudioSegment
    import io as _io
    import json as _json
    import re as _re

    slug = _re.sub(r"[^a-z0-9_-]+", "-", name.lower().strip()).strip("-") or "reference"
    # Avoid clobber — add a numeric suffix if the slug already exists.
    target = USER_REF_DIR / f"{slug}.wav"
    i = 2
    while target.exists():
        target = USER_REF_DIR / f"{slug}-{i}.wav"
        i += 1

    seg = AudioSegment.from_file(_io.BytesIO(audio_bytes))
    seg = seg.set_frame_rate(24000).set_channels(1)
    USER_REF_DIR.mkdir(parents=True, exist_ok=True)
    seg.export(target, format="wav")

    # Sidecar JSON preserves the exact name + description the user typed.
    meta = {"name": name.strip(), "description": description.strip()}
    (USER_REF_DIR / f"{target.stem}.json").write_text(_json.dumps(meta, indent=2))

    if transcript.strip():
        (USER_REF_DIR / f"{target.stem}.txt").write_text(transcript.strip())

    display_label = (
        f"{meta['name']} ({meta['description']})" if meta["description"] else meta["name"]
    )
    return {
        "id": f"user:{target.stem}",
        "label": display_label,
        "name": meta["name"],
        "description": meta["description"],
        "path": str(target),
    }


def delete_user_reference(slug: str) -> bool:
    """Remove a user reference clip + its sidecars."""
    wav = USER_REF_DIR / f"{slug}.wav"
    txt = USER_REF_DIR / f"{slug}.txt"
    meta = USER_REF_DIR / f"{slug}.json"
    ok = False
    for p in (wav, txt, meta):
        if p.exists():
            p.unlink()
            ok = True
    return ok


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
