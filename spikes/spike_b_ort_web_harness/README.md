# Spike B — ZipVoice-Distill in Chrome via ORT-Web

Tiny static page that loads k2-fsa's ZipVoice-Distill ONNX (text_encoder
+ fm_decoder) via `onnxruntime-web`, runs a 4-NFE flow sampling loop
with dummy tokens, and times load + per-step inference on either the
WebGPU or WASM execution provider.

**Spike A confirmed the ONNX loads and runs on CPU.** This spike's job
is to prove the same ONNX runs in a desktop browser — the core
architectural feasibility question for the whole browser-native pivot.

## What's in the directory

```
spike_b_ort_web_harness/
├── index.html          # minimal UI: precision + EP + seq + steps + buttons
├── harness.js          # ES module: loads ORT-Web from CDN, runs end-to-end
├── README.md           # this file
└── models/             # NOT in git (.onnx files are >600 MB; gitignored)
    ├── fm_decoder.onnx           (455 MB FP32)
    ├── fm_decoder_int8.onnx      (119 MB INT8)
    ├── text_encoder.onnx         (17 MB FP32)
    ├── text_encoder_int8.onnx    (5 MB INT8)
    ├── tokens.txt                (tokenizer vocabulary)
    └── model.json                (model config)
```

## One-time setup — pull the ONNX from HuggingFace

```bash
cd /Users/dmarzari/Server-Sites/Voice/spikes/spike_b_ort_web_harness
mkdir -p models && cd models
for f in fm_decoder.onnx fm_decoder_int8.onnx text_encoder.onnx text_encoder_int8.onnx tokens.txt model.json zipvoice_base.json; do
  curl -sSL -o "$f" "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice_distill/$f"
done
```

## How to run

```bash
cd /Users/dmarzari/Server-Sites/Voice/spikes/spike_b_ort_web_harness
python3 -m http.server 8088
# Then open http://localhost:8088 in Chrome desktop.
```

Click **"1. Load both ONNX sessions"** then **"2. Run end-to-end"**.
The log pane shows per-step timings; the table at the bottom summarizes.

## Controls

- **Precision**: `FP32` (472 MB total) vs `INT8` (124 MB total).
  INT8 is what we'd ship.
- **EP** (execution provider): `WebGPU` (fast path, currently bug-blocked
  on ZipVoice's Zipformer — see REPORT.md) or `WASM (fallback)` (slower
  but works end-to-end).
- **NFE steps**: 4 matches ZipVoice-Distill's distillation target.
  Bump to 8–32 to simulate the undistilled teacher for comparison.
- **Seq length (frames)**: number of mel frames to generate. 50 ≈ 0.5s
  of audio at 24 kHz / hop 256.

## Known issues

**WebGPU** currently fails on both FP32 and INT8 at the first
Zipformer conv module's output projection MatMul. The error is a
known-class ORT-Web kernel bug, not a problem with our ONNX (Spike A's
CPU run passes the same model cleanly). Expect a fix in ORT-Web 1.20+
or a workaround during Phase 3.

**WASM without COOP/COEP** runs single-threaded only. Python's
`http.server` doesn't set those headers, so these numbers are
pessimistic. Reader's Next.js config in Phase 3 will enable
multi-threaded WASM (`SharedArrayBuffer` requires COOP/COEP headers);
expect 3–4× speedup over what this harness measures.

## Pass criteria

- ✅ ONNX sessions load via ORT-Web (both WebGPU + WASM)
- ✅ At least one EP completes end-to-end forward pass without errors
- ✅ End-to-end wall-clock under 60 s per sentence

All three pass on WASM. WebGPU fails the "runs without errors" bar due
to the upstream kernel bug. See `spikes/results/spike_b_browser_timings.json`
for full numbers and `spikes/REPORT.md` for the commentary.

## What this doesn't validate

- **Audio quality** — we run with dummy random tokens and a zero
  speech_condition. The output mel is meaningless sound. Quality
  comparison is Spike D's / Phase 1's job.
- **Real tokenizer** — we use random int64 indices in [1, 99]. Real
  ZipVoice tokenization (phoneme + prompt concat) will be
  implemented in JS in Phase 3 (`src/lib/tts/tokenizer.ts`).
- **Vocoder / audio synthesis** — ZipVoice's fm_decoder outputs a mel
  spectrogram. Turning it into playable audio is either bundled into
  their decoder or needs a separate small vocos ONNX, TBD in Phase 3.

## Updating ORT-Web

`harness.js` imports ORT-Web 1.19.2 from jsdelivr. To try a newer
version, edit the first `import` line:

```js
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.webgpu.min.mjs";
```

If a newer version fixes the WebGPU MatMul bug, rerun and update the
REPORT.md matrix.
