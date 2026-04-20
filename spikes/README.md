# Phase 0 — Feasibility Spikes

Five independent experiments that must pass before we commit to the browser-native
F5-TTS pipeline. Each spike is go/no-go; if any fails its exit criteria we
re-plan before starting Phase 1.

See `/Users/dmarzari/.claude/plans/enumerated-leaping-newell.md` for the full
architectural plan this branch is validating.

## Why spikes first

The plan assumes four things that could each independently blow up:

1. F5-TTS exports cleanly to ONNX.
2. ONNX Runtime Web can load + run that model in Chrome with WebGPU.
3. Vocos (the mel → waveform vocoder) exports + runs the same way.
4. F5 can be fine-tuned on a T4 / free-tier GPU in under ~12 h.
5. There's a usable distillation starting point for flow-matching models.

Building Phase 1's training infrastructure *before* validating these is how
6-month timelines happen. Each spike is < 1 day of work; together they buy
us a grounded go/no-go before we touch the Voice Studio UI.

## The spikes

| # | File | What it proves | Fail condition |
|---|------|---------------|----------------|
| A | [`spike_a_onnx_export.ipynb`](./spike_a_onnx_export.ipynb) | F5-TTS → single ONNX file; PyTorch vs ORT CPU outputs match within RMS < 0.05 | DiT rotary / attention ops don't export; outputs diverge; file too large to load |
| B | [`spike_b_ort_web_harness/`](./spike_b_ort_web_harness/) | Load Spike A's ONNX in Chrome via `onnxruntime-web` WebGPU EP; time one sentence | Model fails to load; WebGPU shader compile errors; OOM; > 60 s / sentence |
| C | [`spike_c_vocos_onnx.ipynb`](./spike_c_vocos_onnx.ipynb) | `charactr/vocos-mel-24khz` exports + runs in ORT-Web, same parity bar | Same criteria as A + B |
| D | [`spike_d_finetune_timing.ipynb`](./spike_d_finetune_timing.ipynb) | Full fine-tune of F5 (or ZipVoice) on ~10 min reference audio completes on a T4 in under 12 h | > 12 h on T4 (means free tier can't hold a run) |
| E | [`spike_e_distillation_research.md`](./spike_e_distillation_research.md) | Literature + OSS survey: is there a viable flow-matching distillation repo we can adapt? | No viable starter — Phase 2 timeline doubles |

## How to run each spike

Order doesn't strictly matter, but **E first** is cheapest and most
informative — it's just reading. A → B → C → D is the natural order for the
hands-on work because B depends on A's artifact and C is a sanity check B needs.

### Spike E (literature, 2–3 h)
Read `spike_e_distillation_research.md`. Append findings to the **Conclusions**
section. Update REPORT.md Spike E row.

### Spike A (ONNX export, Colab / Kaggle T4, ~2 h)
```
1. Upload spike_a_onnx_export.ipynb to Colab (or Kaggle Notebooks).
2. Runtime → Change runtime type → GPU (T4 is fine).
3. Run All.
4. Download f5_teacher.onnx (~1.3 GB) + parity_report.json.
5. Commit the parity report (NOT the onnx — too big for git) under spikes/results/.
```

### Spike B (ORT-Web harness, local, ~1 h)
Requires Spike A's `f5_teacher.onnx`.
```
cd spikes/spike_b_ort_web_harness
# Place f5_teacher.onnx + vocos.onnx + tokenizer.json next to index.html
python3 -m http.server 8088
# Open http://localhost:8088 in Chrome Canary or Chrome 125+.
# Click "Run inference". Note load time + inference time in the log.
```

### Spike C (Vocos ONNX, Colab / Kaggle T4, ~30 min)
```
1. Upload spike_c_vocos_onnx.ipynb to Colab.
2. Run All.
3. Download vocos.onnx (~60 MB) + parity_report_vocos.json.
```

### Spike D (fine-tune timing, Colab / Kaggle T4, 6–12 h overnight)
**This is the longest spike.** Don't start it during the day if you need the
laptop responsive for anything else.
```
1. Upload spike_d_finetune_timing.ipynb to Colab Pro (A100 speeds this up
   a lot; T4 is the actual test).
2. Drop a 10-minute clean-speech reference + transcript into
   /content/reference.wav + reference.txt (notebook will prompt).
3. Run All overnight. Notebook saves a checkpoint every 500 steps so you can
   resume after Kaggle's 9h cutoff.
4. Report wall-clock-to-convergence in REPORT.md.
```

## Exit criteria for the phase

Fill out `REPORT.md` with numbers for every spike. A one-page go/no-go for
each. If 3+ are green we proceed to Phase 1. If 2+ are red we stop and
re-plan — likely pivoting to a smaller base model (Kokoro-82M was the
flagged alternative in the plan's risk register).

## Results storage

Large binary artifacts (.onnx files, audio samples) do **not** go in git.
- Small parity reports, timing logs, and JSON metrics → `spikes/results/`
- Large binaries → stash at `~/voice-spike-artifacts/` locally or a
  scratch Drive folder. REPORT.md should point at where they are and their
  SHA-256s so someone else could reproduce.
