#!/usr/bin/env python3.11
"""Compute ZipVoice prompt_mel for a voice's reference audio (CLI wrapper).

Thin wrapper around backend/mel_features.py. Useful for one-off
re-computation of a voice's prompt_mel outside the /api/clone flow —
e.g. testing, recomputing after a profile was manually staged, or
batch-reprocessing a directory of voices.

The /api/clone endpoint calls mel_features.compute_prompt_mel()
directly; this script is for the human-driven case.

Usage:

  cd backend && source .venv/bin/activate

  # By profile ID (resolves to the profile's reference audio):
  python scripts/compute_prompt_mel.py --profile 0e97772fe314

  # By explicit audio path (output lands next to the audio):
  python scripts/compute_prompt_mel.py data/profiles/0e97772fe314/sample.mp3

Outputs, in either case:
  prompt_mel.f32           — raw Float32 little-endian (num_frames, 100)
  prompt_mel_meta.json     — sizes, SHA-256, sample_rate, feat_scale, …

Re-run when the reference audio changes. Deterministic output bytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from mel_features import compute_prompt_mel  # noqa: E402

PROFILES_DIR = BACKEND_DIR / "data" / "profiles"


def resolve_input(args) -> Path:
    if args.profile:
        pdir = PROFILES_DIR / args.profile
        if not pdir.exists():
            print(f"No such profile: {pdir}", file=sys.stderr)
            sys.exit(2)
        for name in ("source.wav", "source.mp3", "sample.mp3", "sample.wav"):
            candidate = pdir / name
            if candidate.exists():
                return candidate
        print(f"No audio file found in {pdir}", file=sys.stderr)
        sys.exit(2)
    if not args.audio:
        print("Pass audio PATH or --profile ID", file=sys.stderr)
        sys.exit(2)
    return Path(args.audio)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", help="Path to the reference audio file")
    ap.add_argument("--profile", help="Voice profile id (resolves to the profile's reference audio)")
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Where to write prompt_mel.{f32,meta.json} (default: next to input)",
    )
    args = ap.parse_args()

    input_path = resolve_input(args)
    print(f"Input: {input_path}")

    result = compute_prompt_mel(input_path, out_dir=args.out_dir)
    print(f"Wrote {result.f32_path} ({result.byte_size / 1024:.1f} KB)")
    print(f"Wrote {result.meta_path}")
    print(json.dumps(result.meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
