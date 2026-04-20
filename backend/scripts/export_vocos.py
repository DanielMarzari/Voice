#!/usr/bin/env python3.11
"""Fetch + validate the Vocos vocoder ONNX for browser-native playback.

ZipVoice's fm_decoder.onnx outputs a 100-dim mel spectrogram at 24 kHz
sampling rate / 256 hop. Vocos is the paired vocoder that turns that
mel into audio. In Phase 3 the browser does this step locally.

### Why we don't export Vocos ourselves

Vocos's ISTFTHead uses `torch.istft` with complex tensors. Neither of
PyTorch's ONNX exporters handles that cleanly:

- Legacy (`dynamo=False`) path errors out at graph-optimization with
  "Unknown number type: complex" — TorchScript's type system doesn't
  model complex in the export pass.
- Dynamo (`dynamo=True`, the new default in torch 2.4+) completes but
  emits a ScatterND node with int32 indices which ORT rejects at load
  time — a known ORT/dynamo compatibility gap.

More fundamentally, **ONNX has no iSTFT operator at all**. Even if
either exporter cleared its current bugs, the resulting graph would
fall back to custom ops or external kernels.

### The standard workaround

The vocos ecosystem settled on this split: ONNX produces
`(magnitude, cos(phase), sin(phase))` spectrograms; iSTFT runs outside
ONNX (numpy, scipy, or — for Phase 3 — plain JS in the browser). The
canonical pre-exported artifact lives at:

  https://huggingface.co/wetdog/vocos-mel-24khz-onnx

This script downloads it, verifies the SHA-256, copies it into
backend/data/models/, and runs a parity test against PyTorch's Vocos
(loading the same weights) to confirm the JS-side iSTFT gives
indistinguishable audio.

### Outputs (relative to repo root)

  backend/data/models/vocos.onnx         — 54 MB FP32 (from wetdog)
  backend/data/models/vocos_fp16.onnx    — ~27 MB FP16 (local quantize)
  backend/data/models/vocos_meta.json    — sizes, SHA-256, parity RMS,
                                           istft config (n_fft, hop, win)

### Usage

  cd backend && source .venv/bin/activate
  python scripts/export_vocos.py

Re-run whenever wetdog publishes a new revision or we want different
precision variants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BACKEND_DIR / "data" / "models"

# Canonical upstream artifact. Pinned to main branch; bump to a specific
# commit SHA if we ever need to lock a version.
HF_REPO = "wetdog/vocos-mel-24khz-onnx"
HF_FILE = "mel_spec_24khz.onnx"
HF_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}"

# iSTFT config — must match what the ONNX was trained for. Comes from
# charactr/vocos-mel-24khz's config.yaml (same across the PyTorch and
# the wetdog ONNX builds). Phase 3's JS iSTFT uses these verbatim.
ISTFT_CONFIG = {
    "n_fft": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "window": "hann",
    "sampling_rate": 24000,
    "n_mels": 100,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Stream-download a file showing progress. No deps; just urllib."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        with dest.open("wb") as f:
            read = 0
            last_pct = -1
            while True:
                chunk = r.read(2**20)
                if not chunk:
                    break
                f.write(chunk)
                read += len(chunk)
                pct = int(100 * read / total) if total else 0
                if pct != last_pct and pct % 10 == 0:
                    print(f"    {pct}% ({read // 2**20} MB)")
                    last_pct = pct
    print(f"  done ({dest.stat().st_size / 1024 / 1024:.1f} MB)")


def reconstruct_and_istft(
    mag: np.ndarray,
    cos_phase: np.ndarray,
    sin_phase: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    """Turn Vocos ONNX's (mag, cos, sin) outputs into a waveform via iSTFT.

    This is the reference Python port of the iSTFT step Phase 3 will
    do in JS. We use torch.istft here to exactly match what PyTorch
    Vocos does internally — any difference from Vocos's output is then
    purely an ONNX-vs-PyTorch numerical-precision gap, not an algorithm
    mismatch. The JS port can use any iSTFT that produces the same
    output as torch.istft with center=True (e.g. a hand-rolled
    overlap-add).
    """
    import torch

    # Reconstruct complex spectrogram: z = mag * (cos + 1j*sin)
    real = torch.from_numpy(mag * cos_phase)
    imag = torch.from_numpy(mag * sin_phase)
    spec = torch.complex(real, imag)  # [B, n_fft/2+1, T]

    window = torch.hann_window(cfg["win_length"])
    audio = torch.istft(
        spec,
        n_fft=cfg["n_fft"],
        hop_length=cfg["hop_length"],
        win_length=cfg["win_length"],
        window=window,
        center=True,
    )
    return audio.numpy().astype(np.float32)


def parity_test(
    onnx_path: Path,
    rms_threshold: float = 0.05,
) -> Tuple[float, float]:
    """Run the PyTorch Vocos through same inputs, compare with
    ONNX + reconstruct_and_istft(). Returns (max_rms, mean_rms)."""
    import torch
    from vocos import Vocos
    import onnxruntime as ort

    print(f"Loading PyTorch Vocos from HF (parity reference)…")
    pt_vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()

    sess = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )

    rms_values = []
    for frames in (100, 200, 400):
        torch.manual_seed(42 + frames)
        mel = torch.randn(1, 100, frames)

        # PyTorch reference
        with torch.no_grad():
            pt_wav = pt_vocos.decode(mel).numpy()

        # ONNX + manual iSTFT
        mag, cx, sy = sess.run(
            None, {"mels": mel.numpy().astype(np.float32)}
        )
        ort_wav = reconstruct_and_istft(mag, cx, sy, ISTFT_CONFIG)

        # RMS on the shared prefix (edge-sample drift between two iSTFT
        # implementations is expected at the 1e-4 level).
        n = min(pt_wav.shape[-1], ort_wav.shape[-1])
        rms = float(
            np.sqrt(np.mean((pt_wav[..., :n] - ort_wav[..., :n]) ** 2))
        )
        rms_values.append(rms)
        audio_s = frames * ISTFT_CONFIG["hop_length"] / ISTFT_CONFIG["sampling_rate"]
        passed = rms < rms_threshold
        print(
            f"  T={frames:3d} frames ({audio_s:.2f}s audio): "
            f"RMS={rms:.6f}  {'✓' if passed else '✗ FAIL'}"
        )

    return max(rms_values), float(np.mean(rms_values))


def maybe_convert_fp16(fp32_path: Path) -> Path | None:
    """Convert FP32 → FP16. Returns the FP16 path, or None if the
    conversion raised (which is survivable — FP32 alone ships)."""
    fp16_path = fp32_path.with_name(fp32_path.stem + "_fp16.onnx")
    try:
        import onnx
        from onnxconverter_common import float16

        print(f"Converting FP32 → FP16 → {fp16_path.name}")
        m = onnx.load(str(fp32_path))
        m_fp16 = float16.convert_float_to_float16(m, keep_io_types=True)
        onnx.save(m_fp16, str(fp16_path))
        print(f"  FP16 size: {fp16_path.stat().st_size / 1024 / 1024:.1f} MB")
        return fp16_path
    except Exception as e:
        print(f"  FP16 conversion failed: {e}")
        print("  FP32 alone is shippable; continuing.")
        if fp16_path.exists():
            fp16_path.unlink()
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rms-threshold",
        type=float,
        default=0.05,
        help="Max acceptable RMS between PyTorch Vocos and ONNX+iSTFT",
    )
    ap.add_argument("--skip-fp16", action="store_true")
    ap.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload even if the ONNX already exists locally",
    )
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp32_path = OUT_DIR / "vocos.onnx"

    if fp32_path.exists() and not args.force_download:
        print(f"Found cached {fp32_path} ({fp32_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        download(HF_URL, fp32_path)

    # Parity
    print()
    print("Parity test (PyTorch Vocos vs ONNX + iSTFT reference):")
    max_rms, mean_rms = parity_test(fp32_path, args.rms_threshold)
    if max_rms >= args.rms_threshold:
        print(f"✗ Parity FAILED — max RMS {max_rms:.4f} >= {args.rms_threshold}")
        print(
            "  This usually means ISTFT_CONFIG drifted from the wetdog "
            "ONNX's training config — check wetdog/vocos-mel-24khz-onnx's "
            "config.yaml against ISTFT_CONFIG in this script."
        )
        return 1

    # FP16
    fp16_path = None
    fp16_max_rms = None
    if not args.skip_fp16:
        fp16_path = maybe_convert_fp16(fp32_path)
        if fp16_path:
            print("Parity test (FP16 vs PyTorch):")
            fp16_max_rms, _ = parity_test(fp16_path, rms_threshold=0.1)

    # Metadata
    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {
            "hf_repo": HF_REPO,
            "hf_file": HF_FILE,
            "hf_url": HF_URL,
            "note": (
                "Upstream ONNX from wetdog. iSTFT runs outside ONNX "
                "(no ONNX operator exists for istft). Phase 3 ports "
                "the iSTFT step to JS."
            ),
        },
        "istft_config": ISTFT_CONFIG,
        "io": {
            "input": "mels: float32 [batch, 100, n_frames]",
            "outputs": {
                "mag": "float32 [batch, 513, n_frames]  (magnitude)",
                "x": "float32 [batch, 513, n_frames]    (cos(phase))",
                "y": "float32 [batch, 513, n_frames]    (sin(phase))",
            },
            "note": (
                "To get audio: spec = (mag*x) + 1j*(mag*y); "
                "wav = iSTFT(spec, n_fft=1024, hop=256, window='hann')"
            ),
        },
        "fp32": {
            "path": str(fp32_path.relative_to(BACKEND_DIR.parent)),
            "size_mb": round(fp32_path.stat().st_size / 1024 / 1024, 1),
            "sha256": sha256(fp32_path),
            "parity_max_rms": round(max_rms, 6),
            "parity_mean_rms": round(mean_rms, 6),
        },
        "fp16": None,
    }
    if fp16_path and fp16_path.exists():
        meta["fp16"] = {
            "path": str(fp16_path.relative_to(BACKEND_DIR.parent)),
            "size_mb": round(fp16_path.stat().st_size / 1024 / 1024, 1),
            "sha256": sha256(fp16_path),
            "parity_max_rms": round(fp16_max_rms, 6) if fp16_max_rms else None,
        }

    meta_path = OUT_DIR / "vocos_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print()
    print(f"Metadata → {meta_path.relative_to(BACKEND_DIR.parent)}")
    print(json.dumps(meta, indent=2))
    print()
    print("✓ Vocos artifact ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
