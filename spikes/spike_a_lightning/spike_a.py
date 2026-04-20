#!/usr/bin/env python3.12
"""Spike A — ZipVoice-Distill ONNX validation on Lightning Studio (CPU).

Goal: confirm that k2-fsa's pre-exported ZipVoice-Distill ONNX files load
in ONNX Runtime CPU and run forward passes without errors. This answers
the Phase 0 go/no-go for Spike A on what's effectively the "happy path"
identified by Spike E — we don't need to export anything ourselves because
k2-fsa already did it and published to HuggingFace.

HuggingFace source: k2-fsa/ZipVoice, folder `zipvoice_distill/`
- fm_decoder.onnx         (~477 MB FP32) — flow-matching decoder
- fm_decoder_int8.onnx    (~124 MB INT8) — quantized decoder
- text_encoder.onnx       (~17 MB FP32)
- text_encoder_int8.onnx  (~5.5 MB INT8)
- tokens.txt, model.json — tokenizer + config

Target runtime: Ubuntu 24 / Python 3.12 / torch 2.4 CPU / onnxruntime 1.19.
Expected wall-clock: ~3-5 min (mostly HF download, CPU forward pass is seconds).

Output: /teamspace/studios/this_studio/Voice/spikes/results/parity_report_lightning.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download


REPO_ID = "k2-fsa/ZipVoice"
SUBFOLDER = "zipvoice_distill"

FILES = {
    "fm_decoder_fp32": "fm_decoder.onnx",
    "fm_decoder_int8": "fm_decoder_int8.onnx",
    "text_encoder_fp32": "text_encoder.onnx",
    "text_encoder_int8": "text_encoder_int8.onnx",
}

CONFIG_FILES = ["model.json", "tokens.txt", "zipvoice_base.json"]

STUDIO_ROOT = Path("/teamspace/studios/this_studio")
ART_DIR = STUDIO_ROOT / "spike-artifacts" / "zipvoice_distill"
VOICE_REPO = STUDIO_ROOT / "Voice"
REPORT_PATH = VOICE_REPO / "spikes" / "results" / "parity_report_lightning.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_artifacts() -> dict[str, Path]:
    """Pull all 4 ONNX files + config from HF into ART_DIR."""
    ART_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = {}
    for key, fname in FILES.items():
        t0 = time.time()
        local = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{SUBFOLDER}/{fname}",
            local_dir=str(ART_DIR.parent),  # HF mirrors the repo subfolder structure
        )
        dt = time.time() - t0
        path = Path(local)
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"[hf] {fname}: {size_mb:.1f} MB in {dt:.1f}s")
        downloaded[key] = path

    for cfg in CONFIG_FILES:
        try:
            local = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"{SUBFOLDER}/{cfg}",
                local_dir=str(ART_DIR.parent),
            )
            print(f"[hf] {cfg}: ok")
        except Exception as e:
            print(f"[hf] {cfg}: skipped ({e})")

    return downloaded


def describe_session(sess: ort.InferenceSession) -> dict:
    """Return a small serializable description of an ORT session's I/O."""
    def io(info):
        return {"name": info.name, "type": str(info.type), "shape": list(info.shape)}
    return {
        "inputs": [io(i) for i in sess.get_inputs()],
        "outputs": [io(o) for o in sess.get_outputs()],
        "providers": sess.get_providers(),
    }


def make_dummy_input(name: str, shape_template, dtype_str):
    """Build a dummy ndarray matching an ORT input's declared type + shape.

    Handles:
    - scalar inputs (shape=[] → 0-d numpy array with a sensible default value)
    - ZipVoice's first-letter dim convention: N=batch, T=sequence length
    - symbolic axes we haven't seen before → small test values
    - Named inputs we recognize (t, guidance_scale, speed, etc.) get
      semantically-correct dummy values, not just zeros.
    """
    # Map dim → concrete
    concrete = []
    for dim in shape_template:
        if isinstance(dim, int) and dim > 0:
            concrete.append(dim)
        elif isinstance(dim, str):
            d = dim.lower()
            # ZipVoice convention: single capital letters
            if dim == "N":
                concrete.append(1)  # batch=1
            elif dim == "T":
                concrete.append(50)  # sequence length
            elif "batch" in d:
                concrete.append(1)
            elif "seq" in d or "len" in d or "time" in d:
                concrete.append(50)
            elif "dim" in d or "channel" in d:
                concrete.append(100)
            else:
                concrete.append(10)
        else:
            concrete.append(1)

    # Pick a dtype
    is_int64 = "int64" in dtype_str
    is_int32 = "int32" in dtype_str
    is_fp16 = "float16" in dtype_str or "fp16" in dtype_str

    if is_int64:
        dtype = np.int64
    elif is_int32:
        dtype = np.int32
    elif is_fp16:
        dtype = np.float16
    else:
        dtype = np.float32

    # Named inputs we recognize get meaningful values instead of zeros/noise.
    # Flow-matching timestep: mid-trajectory
    if name == "t":
        return np.asarray(0.5, dtype=dtype), concrete
    # CFG guidance scale: ZipVoice uses 1.0 (no guidance) by default
    if name == "guidance_scale":
        return np.asarray(1.0, dtype=dtype), concrete
    # Speed: 1.0 = normal
    if name == "speed":
        return np.asarray(1.0, dtype=dtype), concrete
    # Prompt features length: number of prompt-mel frames
    if name == "prompt_features_len":
        return np.asarray(30, dtype=dtype), concrete
    # Tokens: small non-zero ints so we exercise embedding lookups
    if "tokens" in name.lower():
        return np.random.randint(1, 100, size=concrete or [1], dtype=dtype), concrete

    # Scalar (empty shape) → 0-d array with the right dtype
    if len(concrete) == 0:
        return np.asarray(0.0, dtype=dtype), concrete

    # Tensor: random for floats, zeros for ints
    if dtype in (np.int32, np.int64):
        return np.zeros(concrete, dtype=dtype), concrete
    return np.random.randn(*concrete).astype(dtype), concrete


def try_forward(sess: ort.InferenceSession, name: str) -> dict:
    """Attempt a forward pass with dummy inputs. Report timing + shape."""
    feeds = {}
    feed_shapes = {}
    try:
        for i in sess.get_inputs():
            arr, shape = make_dummy_input(i.name, i.shape, str(i.type))
            feeds[i.name] = arr
            feed_shapes[i.name] = shape
    except Exception as e:
        return {
            "status": "error",
            "error": f"input build failed: {type(e).__name__}: {e}",
        }

    try:
        t0 = time.time()
        outputs = sess.run(None, feeds)
        dt = time.time() - t0
        out_shapes = {o.name: list(arr.shape) for o, arr in zip(sess.get_outputs(), outputs)}
        return {
            "status": "ok",
            "wall_clock_s": dt,
            "input_shapes": feed_shapes,
            "output_shapes": out_shapes,
            "output_dtypes": {o.name: str(arr.dtype) for o, arr in zip(sess.get_outputs(), outputs)},
        }
    except Exception as e:
        return {
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "attempted_input_shapes": feed_shapes,
        }


def main():
    import datetime
    import platform

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"onnxruntime: {ort.__version__}")
    print(f"Providers available: {ort.get_available_providers()}")
    print()

    print("=" * 70)
    print("1. DOWNLOAD")
    print("=" * 70)
    t_dl0 = time.time()
    paths = download_artifacts()
    print(f"Total download time: {time.time() - t_dl0:.1f}s")
    print()

    print("=" * 70)
    print("2. LOAD + FORWARD PASS (CPU)")
    print("=" * 70)

    per_model = {}
    for key, path in paths.items():
        print(f"\n--- {key} ({path.name}) ---")
        size_mb = path.stat().st_size / 1024 / 1024
        print(f"Size: {size_mb:.1f} MB")

        model_record = {
            "file": path.name,
            "size_mb": round(size_mb, 1),
            "sha256_prefix": sha256(path)[:16],
        }
        try:
            t0 = time.time()
            sess = ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )
            load_s = time.time() - t0
            print(f"Load: {load_s:.2f}s")
            model_record["load_s"] = round(load_s, 2)

            desc = describe_session(sess)
            print(f"Inputs: {[(i['name'], i['shape'], i['type']) for i in desc['inputs']]}")
            print(f"Outputs: {[(o['name'], o['shape']) for o in desc['outputs']]}")
            model_record["session"] = desc

            fwd = try_forward(sess, key)
            model_record["forward"] = fwd
            print(f"Forward pass: {fwd['status']}", end="")
            if fwd["status"] == "ok":
                print(f" ({fwd['wall_clock_s']*1000:.0f} ms)")
            else:
                print(f" — {fwd['error']}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            model_record["error"] = f"{type(e).__name__}: {e}"

        per_model[key] = model_record

    print()
    print("=" * 70)
    print("3. REPORT")
    print("=" * 70)

    # Summaries that go into REPORT.md
    all_loaded = all("session" in m for m in per_model.values())
    all_forward_ok = all(m.get("forward", {}).get("status") == "ok" for m in per_model.values())
    total_fp32_mb = sum(
        m["size_mb"] for k, m in per_model.items() if "fp32" in k
    )
    total_int8_mb = sum(
        m["size_mb"] for k, m in per_model.items() if "int8" in k
    )

    verdict = "GREEN" if (all_loaded and all_forward_ok) else (
        "YELLOW" if all_loaded else "RED"
    )

    report = {
        "spike": "A",
        "variant": "lightning-cpu-prebuilt-onnx",
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
        },
        "source": {
            "hf_repo": REPO_ID,
            "hf_folder": SUBFOLDER,
            "note": "Using k2-fsa's pre-exported ONNX — no custom export needed.",
        },
        "models": per_model,
        "summary": {
            "all_models_load": all_loaded,
            "all_forward_pass": all_forward_ok,
            "fp32_total_mb": round(total_fp32_mb, 1),
            "int8_total_mb": round(total_int8_mb, 1),
            "verdict": verdict,
        },
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nVerdict: {verdict}")
    print(f"FP32 browser download: {total_fp32_mb:.0f} MB")
    print(f"INT8 browser download: {total_int8_mb:.0f} MB")
    print(f"Report: {REPORT_PATH}")

    return 0 if verdict == "GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
