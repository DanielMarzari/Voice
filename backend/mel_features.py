"""Mel feature extraction for ZipVoice prompt conditioning.

Shared between scripts/compute_prompt_mel.py (the one-off CLI) and
the /api/clone route, which computes prompt_mel inline when a new
voice is created.

Exactly matches ZipVoice's VocosFbank (zipvoice/utils/feature.py):

  torchaudio.transforms.MelSpectrogram(
      sample_rate=24000, n_fft=1024, hop_length=256,
      n_mels=100, center=True, power=1,
  )
  clamp(min=1e-7).log()

Plus rms_norm of the input wav to target_rms=0.1 (from
zipvoice/utils/infer.py).

The extracted log-mel is written as two files alongside the reference
audio in the voice's profile dir:

  prompt_mel.f32       — raw Float32 little-endian, (num_frames, 100)
                         row-major (time-major)
  prompt_mel_meta.json — {num_frames, n_mels, sample_rate, hop_length,
                          feat_scale, target_rms, byte_size, sha256,
                          source_audio}

These are served to the Reader browser client so it can feed Vocos's
`speech_condition` input without re-computing mel in JS.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np

log = logging.getLogger(__name__)

# ZipVoice's VocosFbank constants — must match exactly for the ONNX
# model to behave correctly. See zipvoice/utils/feature.py + infer.py.
SAMPLING_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 100
TARGET_RMS = 0.1
# Scale applied to prompt_mel going INTO fm_decoder, and inverse applied
# to the generated mel going OUT. Keeps flow-matching residuals in a
# numerically stable range during training.
FEAT_SCALE = 0.1


class PromptMelResult(NamedTuple):
    """What compute_prompt_mel() produces. The .f32 and .json have
    already been written to disk; the caller can also read raw bytes
    + meta dict for direct upload without re-reading."""
    f32_path: Path
    meta_path: Path
    num_frames: int
    byte_size: int
    sha256: str
    meta: dict


def _compute(
    audio_path: Path,
) -> tuple[np.ndarray, float, int]:
    """Load audio, rms_norm, extract log-mel. Returns
    (time_major_float32, post_norm_rms, duration_s).

    Imports torchaudio lazily — it's a heavy dep, and the /api/clone
    route only needs it when save=True.
    """
    import torch
    import torchaudio

    wav, sr = torchaudio.load(str(audio_path))
    if sr != SAMPLING_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(wav)
    if wav.shape[0] > 1:
        # downmix to mono — match load_prompt_wav's behavior
        wav = wav.mean(dim=0, keepdim=True)

    duration_s = wav.shape[-1] / SAMPLING_RATE

    # rms_norm — boost quiet clips to target. Matches ZipVoice
    # utils/infer.py rms_norm (boosts only, never attenuates).
    rms = torch.sqrt(torch.mean(torch.square(wav)))
    if rms < TARGET_RMS:
        wav = wav * TARGET_RMS / rms
    post_rms = float(torch.sqrt(torch.mean(torch.square(wav))))

    # MelSpectrogram with power=1 (NOT default power=2), clamp+log
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=True,
        power=1,
    )
    mel = fbank(wav)                # (1, n_mels, T)
    log_mel = mel.clamp(min=1e-7).log()
    log_mel = log_mel.squeeze(0)    # (n_mels, T)

    # Transpose to (T, 100) — row-major so the browser can cast the
    # fetched ArrayBuffer straight to Float32Array without re-strides.
    time_major = (
        log_mel.transpose(0, 1).contiguous().numpy().astype(np.float32)
    )
    return time_major, post_rms, duration_s


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def compute_prompt_mel(
    audio_path: Path,
    out_dir: Path | None = None,
) -> PromptMelResult:
    """Compute log-mel from a reference audio file and write it +
    metadata JSON to disk. Returns paths + summary.

    Args:
        audio_path: WAV or MP3 (or anything torchaudio can decode).
        out_dir:    where to write prompt_mel.{f32,meta.json}.
                    Defaults to the audio file's parent directory.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))
    out_dir = Path(out_dir or audio_path.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    mel, post_rms, duration_s = _compute(audio_path)
    num_frames, _ = mel.shape
    assert mel.shape == (num_frames, N_MELS)

    raw = mel.tobytes()
    f32_path = out_dir / "prompt_mel.f32"
    meta_path = out_dir / "prompt_mel_meta.json"

    f32_path.write_bytes(raw)
    meta = {
        "num_frames": int(num_frames),
        "n_mels": N_MELS,
        "sample_rate": SAMPLING_RATE,
        "hop_length": HOP_LENGTH,
        "feat_scale": FEAT_SCALE,
        "target_rms": TARGET_RMS,
        "n_fft": N_FFT,
        "dtype": "float32",
        "layout": "time-major (T, n_mels)",
        "byte_size": len(raw),
        "sha256": _sha256(raw),
        "source_audio": audio_path.name,
        "source_duration_s": round(duration_s, 3),
        "post_norm_rms": round(post_rms, 4),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info(
        "prompt_mel: %s (%.2fs, %d frames, %.1f KB)",
        audio_path.name, duration_s, num_frames, len(raw) / 1024,
    )
    return PromptMelResult(
        f32_path=f32_path,
        meta_path=meta_path,
        num_frames=int(num_frames),
        byte_size=len(raw),
        sha256=meta["sha256"],
        meta=meta,
    )
