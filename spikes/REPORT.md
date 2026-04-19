# Phase 0 Feasibility Spike Report

**Status:** 🟡 in progress — fill in as spikes complete.
**Decision target:** go/no-go for Phase 1 (training infrastructure).

## Summary (fill in last)

| Spike | Status | Headline number | Go/No-go |
|-------|--------|-----------------|----------|
| A — F5 → ONNX | ⬜ not started | — | — |
| B — ORT-Web load + run | ⬜ not started | — | — |
| C — Vocos → ONNX | ⬜ not started | — | — |
| D — Fine-tune timing on T4 | ⬜ not started | — | — |
| E — Distillation starter survey | ⬜ not started | — | — |

**Overall recommendation:** _(to be filled in after all 5 spikes complete)_

---

## Spike A — F5-TTS → ONNX export

**Goal:** export `SWivid/F5-TTS` (335M params) to a single ONNX file, verify
PyTorch and ORT CPU outputs match on 3 sample prompts within RMS error < 0.05.

**Environment:** _(Colab / Kaggle / Modal, GPU type)_
**Commit pointer:** _(branch + SHA of the notebook run)_

### Results
- Export succeeded: ⬜ yes / ⬜ no
- ONNX file size: __ MB (target: < 1.5 GB for FP16)
- PyTorch vs ORT CPU RMS error (3 prompts avg): __
- Ops that required custom handling: _(rotary emb, scaled_dot_product, etc.)_
- Artifact location + SHA-256: _(path + hash)_

### Notes
_(any surprises, work-arounds, follow-up)_

### Go/No-go
_(PASS / FAIL / PIVOT — e.g. "PIVOT to Kokoro-82M if F5 DiT rotary won't export")_

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
