# Spike E — Distillation Reconnaissance

**Question:** Is there a maintained OSS flow-matching distillation repo we
can adapt for F5-TTS, or are we writing one from scratch?

**Surprise finding:** The answer is probably neither — **ZipVoice-Distill**
is a 123M-param pre-distilled flow-matching TTS (Apache 2.0, 4 NFE, ONNX
export built in, zero-shot cloning supported) that likely obsoletes most of
Phase 2 entirely. Details below.

---

## Landscape (as of April 2026)

The flow-matching / rectified-flow distillation space matured a lot through
2025. Four relevant categories:

### Category 1 — Already-distilled TTS models we could adopt wholesale

These are the game-changers. If one works, Phase 2 becomes "pick one" instead
of "do research."

| Model | Params | Steps | License | ONNX | Repo |
|-------|--------|-------|---------|------|------|
| **ZipVoice-Distill** | 123M | 4–8 | Apache-2.0 | ✅ built-in (`infer_zipvoice_onnx`) | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) |
| RapFlow-TTS | ~?M | 2–10 | Apache-2.0-ish (NAVER) | ❌ (PyTorch only) | [naver-ai/RapFlow-TTS](https://github.com/naver-ai/RapFlow-TTS) |
| VoiceFlow-TTS | — | — | MIT | ❌ | [X-LANCE/VoiceFlow-TTS](https://github.com/X-LANCE/VoiceFlow-TTS) (ICASSP 2024, older) |
| StableTTS | — | — | MIT | ❌ | [KdaiP/StableTTS](https://github.com/KdaiP/StableTTS) (DiT-based, smaller scale) |

### Category 2 — Methods that reduce F5 steps WITHOUT retraining

**EPSS (Empirically Pruned Step Sampling)** — Interspeech 2025, Zheng et al.
- **What it does:** Non-uniform time-step sampling that prunes redundant
  steps. 4× speedup, 7 NFE, no retraining required.
- **Why this matters:** If 7 NFE × F5's teacher pass is fast enough in the
  browser, Phase 2 (distillation) can be skipped ENTIRELY. We ship the
  stock F5 teacher weights with an EPSS sampler on the client side.
- **Status:** Paper is out; we'd have to port the sampling logic ourselves
  (likely < 200 LOC given it's a scheduler change).
- **Key link:** [Interspeech 2025 PDF](https://www.isca-archive.org/interspeech_2025/zheng25d_interspeech.pdf)

**Sway Sampling** — built into F5-TTS itself since v1 (March 2025). Free
speed-up already in the stock repo.

### Category 3 — Generic flow-matching distillation libraries

If we need to distill ourselves (because EPSS isn't fast enough and ZipVoice
doesn't work for us), these are the starting codebases:

| Repo | Method | Notes |
|------|--------|-------|
| [YangLing0818/consistency_flow_matching](https://github.com/YangLing0818/consistency_flow_matching) | Consistency-FM (velocity consistency loss) | Paper claims 4.4× faster training than consistency models; generic, not TTS-specific |
| [lucidrains/rectified-flow-pytorch](https://github.com/lucidrains/rectified-flow-pytorch) | Rectified Flow + followups | Reference impl, high quality, image-domain |
| [gnobitab/RectifiedFlow](https://github.com/gnobitab/RectifiedFlow) | 2-Rectification (InstaFlow family) | Original RF repo, image-domain |

All three are image-model codebases. Adapting to flow-matching TTS means:
- Swapping the data loader (mel spectrogram pairs + text condition)
- Swapping the generator (F5 DiT instead of a UNet)
- Keeping the distillation loss + training loop structure

Estimated effort: 2–4 weeks of real research engineering on top of the
"backend-agnostic `train.py`" we were going to write anyway.

### Category 4 — Academic papers with no released code

- **TraFlow** (arXiv 2502.16972, Feb 2025) — trajectory distillation on pre-trained
  rectified flow. No code at last check.
- **SCoT** (Straight Consistent Trajectory, 2025) — bridges consistency +
  rectified flow. Only paper.
- **Score Distillation of Flow Matching Models** (arXiv 2509.25127) — uses
  score distillation instead of trajectory matching. Interesting but no code.

These inform our understanding but aren't shippable on the phase 0 timeline.

---

## Recommendation

**Reshape Phase 2 around ZipVoice-Distill, not from-scratch distillation.**

The plan's Phase 2 originally budgeted 4–8 weeks for "implement Rectified
Flow 2-Rectification for F5-TTS." That risk item is now:

- **Primary path (low risk):** Skip F5 entirely. Use **ZipVoice-Distill**
  as the base model. It's smaller (123M → ~250 MB FP16 download, not 670 MB),
  already has 4-step inference, has first-class ONNX export, and is Apache-2.0.
  Per-voice customization becomes fine-tuning a pre-distilled model rather
  than distilling it ourselves. This is conceptually simpler and orders of
  magnitude less risky.
- **Secondary path (medium risk):** Stick with F5 as planned but skip
  distillation. Port EPSS sampling to the client; ship teacher weights with
  7-NFE inference. Phase 2 becomes a 1-week JS port instead of 4–8 weeks of
  research.
- **Tertiary path (original plan, high risk):** Distill F5 ourselves using
  Consistency-FM or rectified-flow-pytorch as the starting codebase. Only
  if both above fail a quality bar.

### Impact on other spikes

- **Spike A** (F5 → ONNX): Run as planned for F5, but **add a parallel cell
  that exports ZipVoice-Distill to ONNX using its built-in tooling.** If
  ZipVoice exports cleanly and F5 doesn't, pivot on the spot.
- **Spike B** (ORT-Web harness): Test both models if both Spike A paths
  succeed. Compare inference time (expected: ZipVoice 3–5× faster due to
  smaller model × fewer steps).
- **Spike D** (fine-tune timing): Fine-tune ZipVoice-Distill instead of F5
  (or both). ZipVoice's smaller size should cut fine-tune time substantially.

### Impact on Phase 1 training infrastructure

Mostly unchanged. The `training_pipeline/` package and backend-agnostic
`train.py` are still correct, just pointed at ZipVoice's training script
instead of F5's. Per-platform runners don't care which base model is
underneath.

### Impact on Phase 2 distillation

**Radically simplified** if ZipVoice path works. Phase 2 collapses into:
- Fine-tune ZipVoice-Distill on per-voice data (this is closer to Phase 1
  scope)
- Export to ONNX
- Validate

No flow-matching research, no consistency-loss implementation, no multi-stage
training regime. The 4–8 week "highest-risk phase" becomes a 1–2 week fine-tune.

### Impact on Phase 3 client

No effective change — still ORT-Web + WebGPU + flow sampling loop. The
sampler count drops from 32 (F5 teacher) to 4–8 (ZipVoice-Distill), making
the client-side loop simpler.

---

## Go/no-go criteria for Spike E

- ✅ At least one viable starting codebase exists (**yes** — ZipVoice, plus
  3–4 reasonable alternatives)
- ✅ At least one already-distilled flow-matching TTS with reference-voice
  cloning + Apache/MIT license exists (**yes** — ZipVoice-Distill)
- ✅ If we must distill ourselves, a usable reference implementation exists
  (**yes** — Consistency-FM, rectified-flow-pytorch)

**Verdict: GREEN.** Proceed to Spikes A–D. Expect ZipVoice path to dominate.

---

## Follow-ups to track once hands-on spikes start

1. **Verify ZipVoice ONNX export actually works.** Their README claims it;
   Spike A should confirm on a fresh Colab.
2. **Measure ZipVoice quality on Dan's reference voices.** Lab-benchmark
   numbers don't always transfer to a specific speaker's voice. Spike D
   will produce subjective samples.
3. **Check ZipVoice model size post-FP16.** Claimed 123M params × 2 bytes =
   ~250 MB. With vocoder + tokenizer overhead, likely 300–350 MB first-use
   download. That's a much easier UX sell than 670 MB.
4. **Understand ZipVoice's voice-cloning API.** Zero-shot from a prompt
   clip is claimed — does that obviate per-voice fine-tuning entirely? If
   so, Phase 2 collapses to **zero**. Test with a short (10 s) reference
   clip in Spike D.

## Key sources

- [F5-TTS paper (arXiv 2410.06885)](https://arxiv.org/abs/2410.06885)
- [F5-TTS repo](https://github.com/SWivid/F5-TTS)
- [ZipVoice paper (arXiv 2506.13053)](https://arxiv.org/abs/2506.13053)
- [ZipVoice repo (k2-fsa)](https://github.com/k2-fsa/ZipVoice)
- [RapFlow-TTS paper (arXiv 2506.16741)](https://arxiv.org/abs/2506.16741)
- [RapFlow-TTS repo (naver-ai)](https://github.com/naver-ai/RapFlow-TTS)
- [Accelerating Flow-Matching TTS (EPSS), Interspeech 2025](https://www.isca-archive.org/interspeech_2025/zheng25d_interspeech.pdf)
- [Consistency Flow Matching](https://github.com/YangLing0818/consistency_flow_matching)
- [rectified-flow-pytorch (lucidrains)](https://github.com/lucidrains/rectified-flow-pytorch)
- [TraFlow (arXiv 2502.16972)](https://arxiv.org/abs/2502.16972)
