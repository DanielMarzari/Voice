# Spike B — ORT-Web + WebGPU harness

Tiny static page that loads an ONNX model via `onnxruntime-web`, runs it
on the WebGPU execution provider, and times end-to-end synthesis of one
sentence.

## Prerequisites

From Spike A you should have:
- `f5_teacher_fp16.onnx` OR `zipvoice_distill.onnx` (one is enough)
- `vocos.onnx` (from Spike C — if not done yet, Spike B tests the
  generator in isolation and outputs a mel matrix instead of audio)
- `tokenizer.json` for whichever model you chose

Place them next to `index.html`:

```
spike_b_ort_web_harness/
├── index.html
├── harness.js
├── f5_teacher_fp16.onnx     (or zipvoice_distill.onnx)
├── vocos.onnx               (optional for this spike)
└── tokenizer.json
```

## How to run

```bash
cd spikes/spike_b_ort_web_harness
python3 -m http.server 8088
```

Open `http://localhost:8088` in Chrome 125+ or Chrome Canary with WebGPU
enabled (it's on by default on recent desktop Chrome).

Click **Run inference**. Watch the log for:
- Model load time (download + WASM/WebGPU session init)
- Token encode time
- Inference time (32 sampling steps for F5, 4 for ZipVoice)
- Vocos decode time (if vocos.onnx is present)
- Total wall clock

## Why static files + python http.server?

We want to benchmark **the pipeline**, not Next.js / webpack / bundler
overhead. If WebGPU feasibility fails here, it'll fail in Reader too — no
amount of framework work fixes it. This harness is the ground truth.

## Pass criteria (per the plan)

- Model loads without errors (no unsupported ops, no OOM)
- WebGPU EP initializes successfully (fallback to WASM is acceptable for
  the "it works" question but must be noted separately)
- One sentence completes in < 60 s total

## What to record in REPORT.md

See the Spike B section of `spikes/REPORT.md`. Key numbers:
- model load time
- single-sentence wall clock
- console warnings/errors
- WASM fallback time for comparison
