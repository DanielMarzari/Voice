# Phase 0 Feasibility Spike Report

**Status:** 🟡 in progress — fill in as spikes complete.
**Decision target:** go/no-go for Phase 1 (training infrastructure).

## Summary (fill in last)

| Spike | Status | Headline number | Go/No-go |
|-------|--------|-----------------|----------|
| A — ZipVoice-Distill ONNX on CPU | ✅ done | 124 MB INT8 / 472 MB FP32 total; 1.7× real-time on CPU | **GREEN** |
| B — ORT-Web load + run | ⬜ not started | — | — |
| C — Vocos → ONNX | ⏸ deferred | ZipVoice embeds its own vocoder; Vocos may not be needed | — |
| D — Fine-tune timing on T4 | ⬜ not started | — | — |
| E — Distillation starter survey | ✅ done | ZipVoice-Distill obsoletes custom distillation | **GREEN** |

**Overall recommendation (preliminary, after Spikes A + E):** 🟢 **Strong
proceed to Phase 1.** The architectural risks Phase 0 was designed to
surface have mostly resolved in the happy direction: k2-fsa has shipped
what we were going to build, the ONNX loads cleanly on CPU, download
budgets are better than planned (124 MB INT8 / 472 MB FP32), and Phase 2
distillation scope collapses to zero. Remaining risk is Spike B
(WebGPU in-browser) and Spike D (per-voice fine-tune timing on T4).

---

## Spike A — ZipVoice-Distill ONNX validation (CPU)

**Goal (revised after Spike E):** validate that k2-fsa's pre-exported
ZipVoice-Distill ONNX files load and run in ONNX Runtime CPU without
errors. Original goal was custom F5-TTS export; we pivoted when we
discovered k2-fsa has already published fully-exported ONNX to
HuggingFace (`k2-fsa/ZipVoice`, subfolder `zipvoice_distill/`).

**Environment:** Lightning AI Studio (sole-sapphire-nqrv), CPU mode, no
credits consumed. Ubuntu 24.04, Python 3.12.3, torch 2.4 CPU,
onnxruntime 1.19.2.
**Artifact:** `spikes/results/parity_report_lightning.json`
**Code:** `spikes/spike_a_lightning/spike_a.py`

### Results

All 4 ONNX files downloaded from HF, loaded in ORT CPU, forward-passed
successfully with sensible dummy inputs.

| Model | Size (MB) | Load (s) | Forward (ms) | Status |
|-------|-----------|----------|--------------|--------|
| `fm_decoder.onnx` (FP32) | 455.4 | 1.57 | 94 | ✅ |
| `fm_decoder_int8.onnx` | 118.9 | 1.17 | 67 | ✅ |
| `text_encoder.onnx` (FP32) | 16.8 | 0.20 | 20 | ✅ |
| `text_encoder_int8.onnx` | 5.3 | 0.21 | 20 | ✅ |

- Browser download budgets:
  - **FP32 bundle: 472.2 MB** (fm_decoder + text_encoder)
  - **INT8 bundle: 124.2 MB** ← **-81% vs F5's 670 MB plan target**
- CPU inference (one sentence ≈ text_encoder + 4 × fm_decoder at 50-frame batch):
  - FP32: 395 ms for ~0.5 s audio → **1.3× real-time**
  - INT8: 289 ms for ~0.5 s audio → **1.7× real-time**
- Input signatures captured in `parity_report_lightning.json` — match
  expectations for flow-matching TTS with CFG (`guidance_scale` input)
  and zero-shot voice cloning (`speech_condition` mel prompt input).

### Notes

- **Download was 0.2 s the second run** — HF files were already in
  `~/.cache/huggingface/` from the first (failed) run. Lightning's
  persistent 100 GB disk means Spike B can reuse them without
  re-downloading.
- k2-fsa split the model into two ONNX files (text_encoder + fm_decoder)
  instead of one. This matches how ORT-Web recommends staging large
  models — load the smaller text_encoder first, defer the bigger
  fm_decoder until first synth. Good for UX.
- ONNX uses opset that ORT 1.19 handles fine. No custom ops warning,
  no unsupported-op errors.
- INT8 is CPU-quantized (QLinear ops). On WebGPU the INT8 benefit is
  less pronounced because WebGPU doesn't do INT8 as efficiently as
  modern CPU AVX — we'll likely ship FP16 for web. Spike B will benchmark.

### Go/No-go

**✅ GREEN — PASS.** Proceed to Spike B using the exact same artifacts
from `/teamspace/studios/this_studio/spike-artifacts/zipvoice_distill/`
(already on the Lightning Studio's persistent disk).

**Phase 2 distillation scope collapses to near-zero** because k2-fsa
already did the distillation for us. Phase 2 is now "optionally
fine-tune ZipVoice-Distill per voice" rather than "research-grade
consistency/rectified-flow distillation of F5." This is the biggest
scope reduction the spikes have produced.

---

## Spike B — ORT-Web + WebGPU feasibility

**Goal:** load Spike A's ONNX in Chrome via `onnxruntime-web` WebGPU EP, run
1 sentence of inference end-to-end (F5 + Vocos from Spike C) in < 60 s.

**Environment:** _(browser + version, GPU, OS)_

### Results
- Model load time: __ s
- First inference wall-clock: __ s
- Sentence length tested: __ tokens
- Console errors / warnings: _(paste or "none")_
- WASM fallback also tested? ⬜ yes / ⬜ no — time: __ s
- Peak GPU memory observed (DevTools → Performance → Memory): __ MB

### Notes
_(shader compile issues, missing ops, anything weird)_

### Go/No-go

---

## Spike C — Vocos → ONNX export

**Goal:** same process as Spike A, for `charactr/vocos-mel-24khz` vocoder.

**Environment:**
**Commit pointer:**

### Results
- Export succeeded: ⬜
- ONNX size: __ MB (target < 100 MB)
- PyTorch vs ORT CPU RMS error: __
- Artifact SHA-256:

### Go/No-go

---

## Spike D — F5 fine-tune timing on T4

**Goal:** fine-tune F5 teacher on ~10 min of reference audio on a free-tier
GPU. Measure wall-clock to reasonable convergence (MCD stabilizing).

**Environment:** _(Colab free / Pro, Kaggle free, GPU model)_

### Results
- Reference audio: __ minutes / __ segments / __ MB
- Checkpoint interval: every __ steps
- Wall-clock to convergence criterion: __ hours
- Number of session restarts required: __
- Final loss: __
- Fits single free-tier session (9h Kaggle / 12h Colab)? ⬜ yes / ⬜ no

### Notes
_(OOM? Any hyperparameter tweaks needed beyond defaults?)_

### Go/No-go
_(PASS if < 12 h on T4; PARTIAL if only fits on Colab Pro / A100)_

---

## Spike E — Distillation reconnaissance

**Goal:** is there a maintained OSS repo we can adapt for flow-matching
distillation of F5?

### Candidate repos surveyed

| Repo | Stars | Last commit | Fit for F5? | Notes |
|------|-------|-------------|-------------|-------|
| _(fill in)_ | | | | |

### Key papers reviewed

- _(title — link — 1-line takeaway)_

### Recommendation
_(adopt X / fork Y and modify / give up and do 8-step student)_

### Go/No-go

---

## Cross-spike findings

_(Anything that cuts across — e.g. "all three ONNX spikes hit the same rotary
emb issue, solution is opset 18")_

## Decision

**Proceed to Phase 1?** _(yes / no / yes with scope change)_

_(signed + date)_
