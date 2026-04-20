# Spike A — Lightning Studio variant

Runs on a Lightning AI Studio in CPU mode (no credit burn). Replaces the
Colab-flavored `spike_a_onnx_export.ipynb` for this project since we
discovered k2-fsa already pre-exported ZipVoice-Distill to ONNX and
published it on HuggingFace — so Spike A collapses from "export + parity"
to "download + validate forward pass."

## Why Lightning instead of Colab

- **CPU mode is free on Lightning Studios** — no credits consumed by this
  spike at all. We save the 15 monthly credits for Spike D's fine-tune.
- **100 GB persistent disk** at `/teamspace/studios/this_studio/` means
  the downloaded ONNX files stay around for Spike B (browser harness)
  and can be reused without re-downloading.
- **Full SSH access** — the spike runs end-to-end via
  `ssh sole-sapphire-nqrv 'python spike_a.py'` from the Mac, with
  deterministic Linux + Python 3.12 + torch 2.4 CPU + onnxruntime 1.19.
- **No upload/download dance** — the Mac rsyncs the script up and pulls
  the JSON report back.

## What Spike E changed

Original Spike A plan: export F5-TTS (335M params, DiT + rotary) to ONNX
ourselves and verify PyTorch-vs-ORT parity. High risk — ONNX export of
DiT-with-rotary is historically flaky.

After Spike E's research, we pivoted the primary path to ZipVoice-Distill
(123M params, Zipformer). Then while planning the Lightning run, we found
**k2-fsa has already published fully-exported ONNX** in their HF repo
(`k2-fsa/ZipVoice`, subfolder `zipvoice_distill/`):

- `fm_decoder.onnx` — 477 MB FP32 (the flow-matching decoder, 4 NFE distilled)
- `fm_decoder_int8.onnx` — 124 MB INT8 quantized
- `text_encoder.onnx` — 17 MB FP32
- `text_encoder_int8.onnx` — 5.5 MB INT8
- `tokens.txt`, `model.json` — tokenizer + config

So the "can F5 export cleanly?" risk the spike was designed to probe has
effectively been answered by k2-fsa's release engineering. Our remaining
Phase 0 job is just to confirm their artifacts load + run via
`onnxruntime` CPU. That's this script.

## Files

- `spike_a.py` — downloads, loads, runs a dummy forward pass, writes
  `spikes/results/parity_report_lightning.json`
- `README.md` — this file

## Running it

### On your Mac (remote execution)

```bash
# One-time setup (covered in our main conversation): install the
# lightning CLI, run `lightning configure ssh`, confirm SSH works.
ssh sole-sapphire-nqrv 'echo ok'

# Push this folder to the Studio and run:
rsync -az /Users/dmarzari/Server-Sites/Voice/spikes/spike_a_lightning/ \
  sole-sapphire-nqrv:/teamspace/studios/this_studio/Voice/spikes/spike_a_lightning/
ssh sole-sapphire-nqrv \
  'cd /teamspace/studios/this_studio && source spike-venv/bin/activate && \
   cd Voice && python spikes/spike_a_lightning/spike_a.py'

# Pull the report back
rsync -az sole-sapphire-nqrv:/teamspace/studios/this_studio/Voice/spikes/results/ \
  /Users/dmarzari/Server-Sites/Voice/spikes/results/
```

### Inside the Studio (direct)

```bash
cd /teamspace/studios/this_studio
source spike-venv/bin/activate
cd Voice
python spikes/spike_a_lightning/spike_a.py
```

## Expected wall-clock

- HF download: 2-3 min (620 MB total across 4 files on Lightning's network)
- ORT session loads (×4): <2 s each on CPU
- Forward passes (×4): <5 s each with dummy inputs at sensible shapes
- **Total: ~3-5 min**

## Artifacts produced

- `/teamspace/studios/this_studio/spike-artifacts/zipvoice_distill/*.onnx`
  — the downloaded ONNX files (live on the Studio's persistent disk, used
  again by Spike B)
- `spikes/results/parity_report_lightning.json` — the JSON report (gets
  committed)

## Pass criteria

- ✅ GREEN: all 4 ONNX files load and forward-pass succeeds on CPU
- 🟡 YELLOW: all 4 load but some forward pass fails (e.g. missing or
  misshapen inputs) — we still have a viable path, just need to match
  the expected input signature
- 🔴 RED: any file fails to load (unsupported ops, corrupt file)

## What this does NOT validate

- Quality parity vs the PyTorch reference. That requires running the
  actual ZipVoice pipeline end-to-end, which needs the k2-fsa codebase
  installed. Deferred to Spike D / Phase 1 — by then we'll have a proper
  fine-tune workflow and can compare samples.
- WebGPU inference speed. That's Spike B's job.
- Browser download UX of 495 MB / 130 MB blobs. Also Spike B.
- Voice-cloning fidelity on Dan's reference voices. Phase 1.

## Venv notes

The Studio's `/teamspace/studios/this_studio/spike-venv/` has:
- `torch==2.4.0+cpu` (CPU-only wheel, ~200 MB install)
- `onnxruntime==1.19.2`
- `onnx==1.16.2`
- `huggingface_hub==0.25.2`
- `numpy`

It's intentionally separate from the Voice Studio backend's venv — nothing
the spike installs is compatible with our `backend/requirements.txt`
(different torch versions, transformers pin, etc.). If you want to tear
it down: `rm -rf /teamspace/studios/this_studio/spike-venv`.
