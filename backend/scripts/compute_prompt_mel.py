#!/usr/bin/env python3.11
"""Compute ZipVoice prompt_mel for a voice's reference audio.

Takes an audio file (WAV or MP3) and writes the log-mel spectrogram as
a packed binary file suitable for fetching + reinterpreting as a
Float32Array in the browser. Matches ZipVoice's VocosFbank exactly:

  torchaudio.transforms.MelSpectrogram(
      sample_rate=24000, n_fft=1024, hop_length=256,
      n_mels=100, center=True, power=1,
  )
  clamp(min=1e-7).log()
  rms_norm(audio, target_rms=0.1)   # applied to waveform before mel

Also applied before feeding the model: multiply by feat_scale=0.1.
We ship the *unscaled* log-mel in the .f32 file so the client controls
the scale constant. Clients un-scale the generated output by the same
factor before vocos.

Outputs:

  <profile_dir>/prompt_mel.f32           — raw Float32 little-endian,
                                           shape (num_frames, 100),
                                           row-major (time-major)
  <profile_dir>/prompt_mel_meta.json     — {num_frames, n_mels, sample_rate,
                                            hop_length, feat_scale, sha256}

Usage:

  cd backend && source .venv/bin/activate
  python scripts/compute_prompt_mel.py \\
      data/profiles/0e97772fe314/sample.mp3

  # Or by profile id (resolves to the profile's source.wav | sample.*):
  python scripts/compute_prompt_mel.py --profile 0e97772fe314

Re-run whenever the reference audio changes. Deterministic — same input
produces the same bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROFILES_DIR = BACKEND_DIR / "data" / "profiles"

# ZipVoice's VocosFbank constants (from zipvoice/utils/feature.py).
SAMPLING_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 100
TARGET_RMS = 0.1
FEAT_SCALE = 0.1


def load_prompt_wav(path: Path) -> torch.Tensor:
    """Load + resample to 24 kHz mono.

    Matches ZipVoice's utils/infer.py load_prompt_wav(). torchaudio
    handles wav, mp3, flac, ogg out of the box via its ffmpeg backend.
    """
    wav, sr = torchaudio.load(str(path))
    if sr != SAMPLING_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLING_RATE)(wav)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # downmix to mono
    return wav  # (1, T)


def rms_norm(wav: torch.Tensor, target_rms: float = TARGET_RMS) -> torch.Tensor:
    """Normalize to target RMS if it's quieter than target. Matches
    ZipVoice's utils/infer.py rms_norm()."""
    rms = torch.sqrt(torch.mean(torch.square(wav)))
    if rms < target_rms:
        wav = wav * target_rms / rms
    return wav


def extract_mel(wav: torch.Tensor) -> torch.Tensor:
    """Apply VocosFbank exactly as ZipVoice does: power=1 mel, clamp+log."""
    fbank = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        center=True,
        power=1,
    )
    mel = fbank(wav)  # (1, n_mels, T)
    log_mel = mel.clamp(min=1e-7).log()
    return log_mel


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def resolve_input(args) -> Path:
    if args.profile:
        pdir = PROFILES_DIR / args.profile
        if not pdir.exists():
            print(f"No such profile: {pdir}", file=sys.stderr)
            sys.exit(2)
        # Try source.* first (cloned voices), then sample.* (uploaded).
        for name in ("source.wav", "source.mp3", "sample.mp3", "sample.wav"):
            candidate = pdir / name
            if candidate.exists():
                return candidate
        print(f"No audio file found in {pdir}", file=sys.stderr)
        sys.exit(2)
    if not args.audio:
        print("Pass --audio PATH or --profile ID", file=sys.stderr)
        sys.exit(2)
    return Path(args.audio)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="Path to the reference audio file")
    ap.add_argument("--audio", dest="audio_flag", help="(alternative) path to audio")
    ap.add_argument(
        "--profile",
        help="Voice profile id; resolves to the profile's reference audio",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write prompt_mel.f32 + meta here (default: next to input)",
    )
    args = ap.parse_args()

    # Support both positional and --audio for convenience.
    if args.audio_flag and not args.audio:
        args.audio = args.audio_flag

    input_path = resolve_input(args)
    print(f"Input: {input_path}")

    # --- Pipeline ---
    wav = load_prompt_wav(input_path)
    duration_s = wav.shape[-1] / SAMPLING_RATE
    print(f"Loaded {duration_s:.2f}s @ {SAMPLING_RATE} Hz mono")

    wav = rms_norm(wav)
    rms_after = float(torch.sqrt(torch.mean(torch.square(wav))))
    print(f"RMS after normalization: {rms_after:.4f}")

    log_mel = extract_mel(wav)  # (1, 100, T)
    log_mel_squeezed = log_mel.squeeze(0)  # (100, T)
    num_frames = log_mel_squeezed.shape[-1]

    # Transpose to (T, 100) so the browser can fill Float32Array
    # time-major without extra work.
    time_major = log_mel_squeezed.transpose(0, 1).contiguous().numpy().astype(np.float32)
    assert time_major.shape == (num_frames, N_MELS)

    # --- Write outputs ---
    out_dir = args.out_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    mel_path = out_dir / "prompt_mel.f32"
    meta_path = out_dir / "prompt_mel_meta.json"

    raw = time_major.tobytes()
    mel_path.write_bytes(raw)

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
        "sha256": sha256_bytes(raw),
        "source_audio": str(input_path.name),
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print()
    print(f"Wrote {mel_path} ({len(raw) / 1024:.1f} KB)")
    print(f"Wrote {meta_path}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
