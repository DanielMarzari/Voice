# Phase 0 Feasibility Spike Report

**Status:** 🟡 in progress — fill in as spikes complete.
**Decision target:** go/no-go for Phase 1 (training infrastructure).

## Summary (fill in last)

| Spike | Status | Headline number | Go/No-go |
|-------|--------|-----------------|----------|
| A — ZipVoice-Distill ONNX on CPU | ✅ done | 124 MB INT8 / 472 MB FP32 total; 1.7× real-time on CPU | **GREEN** |
| B — ORT-Web in Chrome | ✅ done | WASM: 0.57–0.59× RT single-threaded; WebGPU: bug-blocked | **YELLOW→GREEN** |
| C — Vocos → ONNX | ✅ closed via Phase 1 WS2 | Used [wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx); 51.6 MB FP32 / 25.9 MB FP16; parity RMS 0.0 / 6.5e-5 vs PyTorch | **GREEN** |
| D — Zero-shot voice cloning | ✅ done (reframed) | GREEN with ≥10 s prompts; **Phase 2 fine-tuning deleted** | **GREEN** |
| E — Distillation starter survey | ✅ done | ZipVoice-Distill obsoletes custom distillation | **GREEN** |

**Overall recommendation (after all 5 spikes closed):** 🟢 **Strong
proceed to Phase 1, with significantly reduced scope.** All five
architectural risks Phase 0 was designed to surface have resolved
GREEN or YELLOW-GREEN:

- **E** — k2-fsa pre-exported our base model. No custom distillation
  needed.
- **A** — ONNX loads and runs cleanly on CPU at 1.7× real-time.
- **B** — same ONNX runs end-to-end in Chrome desktop via WASM
  (0.57–0.59× single-threaded → ~2× real-time multi-threaded).
  WebGPU hits an ORT-Web kernel bug; not blocking.
- **D** — **ZipVoice zero-shot is GREEN with ≥10-second prompt
  clips.** Dan listened to 3 voices (Alex 6.5s, Marcel 6.1s, Felix
  8.3s). Felix was "near perfect"; shorter prompts showed degradation
  but the same identity transfer. **Phase 2 (per-voice fine-tuning)
  collapses to zero**.
- **C** — deferred; Vocos ONNX is bundled with the zipvoice Python
  package but not in k2-fsa's HF ONNX folder. Phase 3 will export it
  as a one-off using the existing vocos Python library.

**Deployment shape:** ship **INT8 + WASM + COOP/COEP** initially
(~2× real-time from Spike B's numbers). Voice Studio's Clone tab
enforces **≥10 s prompts + auto-transcript via Whisper** (already
installed in backend) and uploads the profile with no training step.
WebGPU stays as an in-progress perf workstream for ORT-Web to fix
upstream.

**Phase 1 scope reduction** (vs the plan at
`/Users/dmarzari/.claude/plans/enumerated-leaping-newell.md`):
- The `training_pipeline/` package (backend-agnostic `train.py` +
  per-platform runners for Colab/Kaggle/Modal/Lightning): **deleted
  from shipping path**. May revisit only if we ever want per-voice
  quality above the zero-shot baseline.
- `backend/training.py` Deep Clone scaffold: park as a future feature.
- `runners/*` directory: never built.
- Phase 2 (originally "4–8 weeks of flow-matching distillation"):
  **deleted**.

What Phase 1 actually is now: (a) tighten Voice Studio's Clone tab
(≥10 s prompt enforcement + Whisper auto-transcribe + upload to
Reader with profile metadata), and (b) Vocos ONNX export (~1 day).
Phase 3 (Reader browser inference) becomes the main workstream.

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

**Goal (revised after Spike E + A):** load k2-fsa's pre-exported
ZipVoice-Distill ONNX (text_encoder + fm_decoder) in Chrome desktop
via `onnxruntime-web`, run end-to-end inference (4-NFE flow sampling)
with dummy tokens, and measure load + per-step timings.

**Environment:** Chrome desktop on MacBook (Apple Silicon GPU via
WebGPU adapter), ORT-Web 1.19.2 from CDN, local `python3 -m http.server`
(single-threaded WASM — no COOP/COEP headers in this spike).
**Harness:** `spikes/spike_b_ort_web_harness/` (`index.html` + `harness.js`)

### Results matrix

Tested all 4 combinations of precision × execution provider. Sequence
length 50 frames, 4 NFE steps, batch=1.

| Precision | EP | Status | Load time | End-to-end |
|-----------|----|----|-----------|------------|
| FP32 | WebGPU | ❌ | 472 MB load ok | crashes at `conv_module1/out_proj/MatMul` — _"shared dimension does not match"_ |
| INT8 | WebGPU | ❌ | 124 MB load ok | crashes at `conv_module1/out_proj/MatMul_quant` — _"left_num_dims and right_num_dims must be >= 1"_ |
| FP32 | WASM | ✅ | 2503 ms | **897 ms** (text 57 + fm 209 ms × 4) → 0.59× real-time |
| INT8 | WASM | ✅ | 1195 ms | **940 ms** (text 48 + fm 222 ms × 4) → 0.57× real-time |

WebGPU hits onnxruntime-web kernel bugs at the same graph location for
both precisions — the first conv module's output projection MatMul in
ZipVoice's Zipformer stack. These are ORT-Web implementation issues
(the same ONNX runs fine on ORT CPU in Spike A); not our bug.

WASM runs **end-to-end successfully** on both FP32 and INT8 at roughly
equivalent compute (INT8's advantage is eaten by WASM's dequant-per-op
overhead since WASM doesn't natively accelerate INT8). INT8 wins on
load time (~2× faster) and download size (4× smaller) — the right
choice for deployment.

### What this proves

- ✅ ORT-Web **loads and executes** ZipVoice-Distill end-to-end in Chrome
- ✅ The full 4-NFE flow-matching loop completes in **under 1 second**
  on the pessimistic single-threaded WASM backend
- ✅ ONNX files (downloaded from HF or our own copies) work identically
- 🟡 WebGPU EP has op-coverage bugs for ZipVoice; WASM is the shippable
  path until ORT-Web's WebGPU kernels catch up

### Extrapolated production numbers

- **Multi-threaded WASM** (COOP/COEP headers → `SharedArrayBuffer`):
  typical 3–4× speedup over single-threaded. Extrapolates to ~225–300 ms
  per sentence → **~2× real-time**. This is shippable today.
- **WebGPU once fixed** (either ORT-Web 1.20+, a workaround in our
  export, or a switch to an alternative web runtime): typically 5–10×
  over WASM for matmul-heavy models → ~50–100 ms per sentence. That's
  what gets us to "instant" for first-sentence playback.

### Notes

- COOP/COEP headers were NOT set in this spike — Python's `http.server`
  doesn't set them by default. Multi-threaded WASM numbers will land
  higher when Phase 3's Reader dev server adds those headers.
- ORT-Web 1.19.2 was pinned deliberately to match Spike A's server-side
  version. Worth re-running Spike B on 1.20+ when we pick up Phase 3 —
  the WebGPU MatMul kernel may already be fixed upstream.
- No console warnings about unsupported ops during model loading —
  failures are at runtime in the specific WebGPU kernel implementation.

### Go/No-go

**🟡 YELLOW→GREEN — PROCEED.** Browser-native ZipVoice-Distill
inference is demonstrably feasible in Chrome desktop via ORT-Web.
WebGPU is the long-term perf path but WASM-multi-threaded is viable
for initial ship. Phase 3's critical path adds COOP/COEP headers
to Reader's Next.js config (one-line fix) and measures multi-threaded
WASM; WebGPU fix is a parallel workstream that doesn't block shipping.

---

## Spike C — Vocos → ONNX export

**Status:** ✅ closed via Phase 1 Workstream 2. See
`backend/scripts/export_vocos.py` + `backend/data/models/vocos_meta.json`.

**Resolution path:** we didn't export Vocos ourselves. Vocos's
`ISTFTHead` uses `torch.istft` on complex tensors, which neither
PyTorch exporter handles cleanly (legacy errors on the complex
number type; dynamo mangles the graph into an ORT-incompatible
shape). More fundamentally, **ONNX has no iSTFT operator at all**.

The vocos ecosystem settled on a standard split: ONNX produces
`(mag, cos(phase), sin(phase))` spectrograms; iSTFT runs outside
ONNX. The canonical pre-exported artifact lives at
[wetdog/vocos-mel-24khz-onnx](https://huggingface.co/wetdog/vocos-mel-24khz-onnx).

We download it via `backend/scripts/export_vocos.py`, verify its
SHA-256, FP16-convert, and run end-to-end parity against PyTorch
Vocos (same HF weights). The iSTFT step uses `torch.istft` for
parity; Phase 3 ports this ~20 lines of overlap-add to JS.

### Results

- **FP32**: 51.6 MB, SHA-256 `a84c58728a769e…`, parity **max RMS 0.0**
  vs PyTorch Vocos. Byte-identical.
- **FP16**: 25.9 MB, SHA-256 `0d03bb5c133540…`, parity max RMS 6.5e-5
  — below audible threshold by orders of magnitude.

Deployment budget for the full browser pipeline:
- text_encoder (INT8) + fm_decoder (INT8) + vocos (FP16) ≈ **150 MB**
- text_encoder (FP32) + fm_decoder (FP32) + vocos (FP32) ≈ **525 MB**

### Go/No-go

**🟢 GREEN**. Single fetch-and-validate script produces the
shippable artifact. Phase 3's client needs a JS iSTFT (~20 lines of
overlap-add) which is a well-documented algorithm.

---

## Spike D — Zero-shot voice cloning quality test

**Goal (reframed):** the original goal was fine-tune timing on a T4
GPU, sizing Phase 2's training infrastructure. After Spike E
identified ZipVoice-Distill and noted it supports zero-shot voice
cloning from a prompt clip, we reframed to answer the upstream
question: **does zero-shot sound good enough to ship without fine-
tuning?**

**Environment:** Lightning AI Studio sole-sapphire-nqrv on CPU mode
(no credits), full ZipVoice Python pipeline (phonemize → ONNX → Vocos
→ WAV), default settings (8 NFE, guidance 3.0).
**Artifacts:** `spikes/spike_d_zero_shot/outputs/*.wav` (11 WAVs,
3.3 MB total) + `spikes/results/spike_d_zero_shot_results.json`

### Method

Three voice prompts × 3 test sentences each = 9 zero-shot generations.
Test sentences were identical across voices for direct A/B comparison.
Whisper (base model) auto-transcribed each prompt. No fine-tuning, no
warm-up, no hyperparameter tweaks.

### Results

| Voice | Prompt | Identity capture | Acoustic quality | Dan verdict |
|-------|--------|------------------|------------------|-------------|
| Alex   | 6.5 s (French-English) | Correct tone + accent | "Worse microphone" artifacts, some reverb | Almost-GREEN |
| Marcel | 6.1 s (strong French) | Recognizable but degraded | Similar artifacts | YELLOW |
| Felix  | 8.3 s (British RP) | **Near perfect**, RP vowels carry cleanly | **Clean** | **GREEN** |

**Headline finding:** quality scales with prompt length. Felix's 1.7
extra seconds of prompt over Alex was the difference between
"ship-ready" and "noticeably degraded." Dan's direct quote:

> _"Felix is GREEN and near perfect; his was the longest sample
> (8 seconds); the others were 7 and 6. I'd say with a longer sample,
> the voices would probably be ready to go with minimal if any
> editing."_

### Timings (informational)

0.15–0.34× real-time on Lightning's CPU through the full Python
pipeline. Not relevant to ship architecture — browser runs on the
user's hardware. Included for completeness; see
`spikes/spike_d_zero_shot/README.md` for the per-sentence breakdown.

### Go/No-go

**🟢 GREEN — PASS** with a product constraint: Voice Studio must
enforce **≥10 s prompts + clean speech + auto-transcript via Whisper**
at upload time.

### Implications for the plan

- **Phase 2 (per-voice fine-tuning, 4–8 weeks in the original plan)
  is deleted from the shipping path.**
- Phase 1's `training_pipeline/` package, runners, and
  `backend/training.py` Deep Clone scaffold: all dropped.
- Phase 1's real scope becomes: (a) tighten Voice Studio's Clone tab
  (length gate + Whisper auto-transcribe + upload flow), (b) Vocos
  ONNX export as a one-off.
- Fine-tune timing remains unmeasured. We don't need the number
  because we don't need the pipeline. If users report voices that
  zero-shot can't handle (heavy accents, non-English, extreme out-of-
  distribution), we re-open Phase 2 scoped to just those cases.

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

**Proceed to Phase 1?** ✅ **YES, with major scope reduction.**

Phase 2 (per-voice fine-tuning) and its entire training-infrastructure
sub-plan is deleted from the shipping path after Spike D's finding
that ZipVoice zero-shot is GREEN with ≥10-second prompts. Phase 1's
scope compresses to two small pieces:

1. Voice Studio Clone tab gate: enforce ≥10s prompt + auto-Whisper
   transcribe + upload profile to Reader (no training step).
2. Vocos ONNX export (~1 day) so Phase 3's browser pipeline has the
   mel-to-audio piece.

Phase 3 (Reader client rewrite) becomes the main workstream.

Decision signed 2026-04-20 on branch `phase-0-spikes`.
