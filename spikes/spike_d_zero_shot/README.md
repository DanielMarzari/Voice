# Spike D — Zero-shot voice cloning quality test

**Verdict: 🟢 GREEN — with ≥10 s prompt clips.** Phase 2 (per-voice
fine-tuning) collapses to zero for the shipping path.

## What we tested and why

Spike D's original goal was to measure **fine-tune timing on a T4 GPU**,
sizing Phase 2's training infrastructure. After Spike E identified
ZipVoice-Distill as the base model and noted it supports zero-shot
voice cloning from a prompt clip, we reframed the spike to answer the
upstream question first:

> **Does ZipVoice zero-shot sound good enough to ship without ever
> fine-tuning?**

If yes, Phase 2 and all its per-voice training gymnastics (cloud GPU
backend selection, multi-session Kaggle checkpointing, 25k-step
distillation) can be deleted from the plan entirely.

## Method

- Environment: Lightning AI Studio (sole-sapphire-nqrv) on CPU mode
  (no credit burn). Full ZipVoice Python pipeline: piper_phonemize →
  ONNX text_encoder → ONNX fm_decoder (8 NFE default, guidance=3.0) →
  Vocos vocoder → WAV save. Default settings, no tuning.
- Three voice prompts of different durations:
  - **Alex** — 6.5 s — French-accented English demo voice
  - **Marcel** — 6.1 s — stronger French accent ("charming native")
  - **Felix** — 8.3 s — British RP ("warm contemporary")
- Auto-transcribed each with Whisper base (CPU) for the prompt-text
  input.
- Generated 3 test sentences per voice (same 3 sentences across all
  voices for direct A/B comparison):
  1. "The morning light filtered through the kitchen blinds..."
  2. "Have you already finished the report you were working on last night?"
  3. "Although the conference had run long, nobody wanted to skip the
     final panel..."

## Results

Subjective listening verdict from Dan (the user whose Phase 1+ depends
on the decision):

| Voice | Prompt length | Identity capture | Acoustic quality | Verdict |
|-------|---------------|------------------|------------------|---------|
| Alex   | 6.5 s | Correct tone + accent | "Worse microphone", reverb, flow issues | Almost-GREEN |
| Marcel | 6.1 s | Recognizable but degraded | Similar artifacts | YELLOW |
| Felix  | 8.3 s | **Near perfect**, RP accent carries cleanly | **Clean** | **GREEN** |

**Clear correlation: longer prompt = higher fidelity.** The 1.7-second
difference between Felix and Alex is the difference between
"ship-ready" and "noticeably degraded."

Dan's verdict on the trajectory:

> _"Felix is GREEN and near perfect; his was the longest sample
> (8 seconds); the others were 7 and 6. I'd say with a longer sample,
> the voices would probably be ready to go with minimal if any editing."_

## Timings (informational, not critical for shipping)

Wall-clock on Lightning's CPU instance, full Python pipeline:

| Sentence | Audio | Gen time | RT factor |
|----------|-------|----------|-----------|
| Alex #1 (med)     | 5.4 s | 21 s | 0.26× |
| Alex #2 (short)   | 3.7 s | 15 s | 0.25× |
| Alex #3 (long)    | 7.9 s | 23 s | 0.34× |
| Felix #1   | 5.6 s | 25 s | 0.22× |
| Felix #2   | 3.4 s | 22 s | 0.15× |
| Felix #3   | 7.9 s | 32 s | 0.25× |
| Marcel #1  | 5.3 s | 23 s | 0.23× |
| Marcel #2  | 3.7 s | 20 s | 0.18× |
| Marcel #3  | 7.8 s | 28 s | 0.28× |

~**0.23× RT average** on Lightning CPU through the full pipeline
(phonemize + ONNX + vocos). This does NOT bound browser playback speed
— the browser runs inference on the user's GPU/CPU, not Lightning.
Spike B's WASM-single-threaded run was 0.57× RT for a subset; WebGPU
or multi-threaded WASM will be much faster.

## Implications for Phase 1 + the overall plan

**Phase 2 (per-voice fine-tuning, 4–8 weeks of the original plan) is
deleted from the shipping path.** Product UX becomes:

- User uploads a **10–15 second voice prompt clip** (enforce minimum
  in the Voice Studio Clone tab)
- Optionally auto-generate the prompt transcript with Whisper (already
  in Voice Studio's backend for the Deep Clone tab)
- Zero-shot inference happens client-side in the browser from then on;
  no training, no cloud GPU, no credit burn

**Phase 1 scope shrinks dramatically:**

- Original plan's `training_pipeline/` package with backend-agnostic
  train.py + per-platform runners (Colab/Kaggle/Modal/Lightning): **not
  needed for MVP.**
- `backend/training.py` Deep Clone scaffold: can be removed/parked as
  a future feature for "quality enhancement" above 15s prompts.
- Focus shifts to: (a) tightening the Clone tab to require ≥10s +
  Whisper auto-transcript, (b) the Reader client (Phase 3) browser
  inference path.

**Fine-tune timing on T4 remains unmeasured** — intentionally. We
don't need the number because we don't need the pipeline. If we ever
decide we want per-voice quality above zero-shot (e.g. for
professionally-produced audiobook voices), we'll do the GPU timing
test then.

## Risk notes

- **Voices that need shorter prompts** (<5 s) are NOT supported by
  this path. The Voice Studio Clone tab must enforce the minimum and
  explain why.
- **Voices the model is strongly out-of-distribution for** (e.g. heavy
  regional accents outside its training data, non-English) may need
  fine-tuning even with 15 s prompts. This is Phase 1 user feedback
  material, not an architectural blocker.
- **Reference audio quality matters** — background noise, reverb, or
  music in the prompt bleeds into the zero-shot output. The Clone tab
  should warn about clean-speech requirements.

## Artifacts in this directory

- `outputs/` — 11 WAVs: 3 prompt originals + 9 zero-shot generations.
  Checked in (small files, ~1.6 MB total); useful for future A/B
  comparisons when ZipVoice updates or we evaluate alternatives.
- `spike_d_results.json` — structured results + verdict.
- `README.md` — this file.
