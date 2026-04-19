"""
Deep Clone — Stage 1 (dataset prep) + Stage 2 (train-script rendering).

Zero-shot cloning (the Clone tab) takes a 10-20 s reference clip and is
great for ballpark mimicry, but wobbles over long reads. "Deep Clone" is
the ElevenLabs-equivalent: fine-tune a base model on ~30 min of clean
speech so the resulting voice is rock-steady for anything you throw at it.

Actually fine-tuning F5-TTS or XTTS on this user's Mac (M2, 16 GB, no
NVIDIA GPU) takes hours and is finicky, so we split it in two:

  Stage 1 — PREPARE (this module, synchronous, ~30 s–few min):
    * Concatenate uploaded audio into a single mono 24 kHz waveform
    * Split into 3–10 s segments via energy-based VAD (webrtcvad-less —
      pure numpy so we don't pick up yet another dep)
    * Transcribe each segment with faster-whisper
    * Write to `data/training/<id>/` as:
        - segment_0001.wav, segment_0002.wav, ...
        - metadata.csv            ("<wav_path>|<transcript>|<normalized>")
        - manifest.json           (name, created_at, segments, duration, ...)
        - train.sh                (the actual finetune command)
        - checkpoints/            (user's finetune writes here)

  Stage 2 — TRAIN (out-of-process, hours):
    * The user runs `bash train.sh` in their terminal. The script is
      written for either F5-TTS's finetune_cli.py or Coqui's XTTS
      recipe — comments explain runtime, where the checkpoint lands,
      and how to stop/resume.
    * The UI polls training_status() every 30 s; once a checkpoint
      file exists at the expected path, status flips to "ready" and
      the voice becomes selectable as `deep:<slug>` in the Design tab.

The tradeoffs here are deliberate:
  - We DON'T try to shell out `python -m f5_tts.train.finetune_cli` from
    the API process. Uvicorn would either block for hours or spawn a
    child that outlives the reload — both paint the user into corners.
    A script the user runs themselves is boring and correct.
  - We DO include the checkpoint detection poll so the UI can promote
    the voice to "ready" without requiring the user to remember to click
    anything post-training.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import shutil
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import numpy as np

log = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).parent / "data" / "training"
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

Engine = Literal["f5", "xtts"]

# Segment window. 3 s is the lower bound below which whisper stops giving
# useful transcripts; 10 s is where F5-TTS's training loss starts to spike
# because attention gets noisy. Sweet spot is 6-8 s.
MIN_SEGMENT_S = 3.0
MAX_SEGMENT_S = 10.0
# Target sample rate for all engines (F5 and XTTS both expect 24 kHz mono).
TARGET_SR = 24000


@dataclass
class TrainingManifest:
    id: str
    name: str
    description: str
    engine: Engine
    created_at: str
    segment_count: int
    total_duration_s: float
    # Relative-to-repo paths we expose to the frontend — display only.
    train_script: str
    checkpoint_glob: str
    status: str = "prepared"  # "prepared" | "training" | "ready" | "failed"
    error: Optional[str] = None
    # Filled in once the user runs train.sh and we detect a checkpoint.
    checkpoint_path: Optional[str] = None
    # Extra metadata surfaced to the frontend list view.
    notes: list[str] = field(default_factory=list)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.lower().strip()).strip("-")
    return slug or "voice"


def _training_dir(training_id: str) -> Path:
    return TRAINING_DIR / training_id


def _manifest_path(training_id: str) -> Path:
    return _training_dir(training_id) / "manifest.json"


# ---------------- Audio loading + segmentation ----------------


def _load_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode any common format to mono float32 PCM + sample rate.
    Uses pydub/ffmpeg (already a dep)."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
    seg = seg.set_channels(1).set_frame_rate(TARGET_SR).set_sample_width(2)
    samples = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, TARGET_SR


def _concat_audio(audio_blobs: list[bytes]) -> tuple[np.ndarray, int]:
    """Decode and concatenate multiple input files into one stream at
    TARGET_SR. Small silences between tracks are fine — the segmenter
    will cut them out anyway."""
    arrays: list[np.ndarray] = []
    for raw in audio_blobs:
        if not raw:
            continue
        pcm, _sr = _load_audio(raw)
        arrays.append(pcm)
        # 0.3 s silence between files to prevent word-smashing across boundaries.
        arrays.append(np.zeros(int(0.3 * TARGET_SR), dtype=np.float32))
    if not arrays:
        raise ValueError("no usable audio")
    return np.concatenate(arrays), TARGET_SR


def _segment_by_silence(
    audio: np.ndarray,
    sr: int,
    *,
    min_s: float = MIN_SEGMENT_S,
    max_s: float = MAX_SEGMENT_S,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 300,
) -> list[tuple[int, int]]:
    """Return (start_sample, end_sample) ranges.

    Strategy: compute RMS energy over 20 ms frames; anything under
    `silence_thresh_db` for at least `min_silence_ms` is a split point.
    Then greedily coalesce short non-silence chunks up to `max_s`. This
    is the same approach pydub's `split_on_silence` uses but inlined so
    we can tweak it for training-data needs (no throwing away content
    between segments — we want contiguous coverage, just with clean cuts).
    """
    frame_len = int(sr * 0.02)  # 20 ms
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return [(0, len(audio))]

    # RMS per frame → dBFS
    trimmed = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt((trimmed ** 2).mean(axis=1) + 1e-12)
    db = 20.0 * np.log10(rms / 1.0 + 1e-12)

    silent = db < silence_thresh_db
    # How many consecutive silent frames make a "gap"?
    min_silence_frames = max(1, int(min_silence_ms / 20))

    # Find runs of silent frames that are long enough; each run is a cut point.
    cuts: list[int] = [0]
    run = 0
    for i, s in enumerate(silent):
        if s:
            run += 1
        else:
            if run >= min_silence_frames:
                # Cut in the middle of the silent run — preserves a bit of
                # natural breathing room on both sides.
                cut_frame = i - run // 2
                cuts.append(cut_frame * frame_len)
            run = 0
    cuts.append(len(audio))

    # Coalesce short chunks up to max_s.
    min_len = int(min_s * sr)
    max_len = int(max_s * sr)
    segments: list[tuple[int, int]] = []
    start = cuts[0]
    for end in cuts[1:]:
        if end - start >= min_len and end - start <= max_len:
            segments.append((start, end))
            start = end
        elif end - start < min_len:
            # Too short on its own; extend into next cut.
            continue
        else:
            # Too long — subdivide at fixed max_len boundaries.
            cur = start
            while end - cur > max_len:
                segments.append((cur, cur + max_len))
                cur += max_len
            if end - cur >= min_len:
                segments.append((cur, end))
            start = end
    # Any trailing audio that never hit a silence — tack it on if long enough.
    if segments and segments[-1][1] < len(audio):
        tail = len(audio) - segments[-1][1]
        if tail >= min_len:
            segments.append((segments[-1][1], len(audio)))
    return segments


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Write a mono float32 array to a 16-bit WAV."""
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(audio, -1.0, 1.0), sr, subtype="PCM_16")


# ---------------- Transcription ----------------


_WHISPER_MODEL = None  # lazy — only loaded on first prepare call
_WHISPER_MODEL_SIZE = "base.en"  # ~150 MB, fast on M2


def _transcribe_segments(segments: list[tuple[Path, np.ndarray, int]]) -> list[str]:
    """Transcribe each segment with faster-whisper.

    Kept as a single pass so the model only loads once. Returns a list
    of transcripts in segment order. Falls back to empty string per
    segment if transcription fails — the user can hand-edit metadata.csv.
    """
    global _WHISPER_MODEL
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        log.warning("faster-whisper not installed (%s); writing blank transcripts.", e)
        return ["" for _ in segments]

    if _WHISPER_MODEL is None:
        log.info("Loading faster-whisper (%s) for deep-clone transcription…",
                 _WHISPER_MODEL_SIZE)
        # compute_type="int8" is the sweet spot on CPU — ~2x faster than
        # float16 with no measurable quality loss on speech ≤30 s.
        _WHISPER_MODEL = WhisperModel(
            _WHISPER_MODEL_SIZE, device="cpu", compute_type="int8"
        )

    transcripts: list[str] = []
    for idx, (wav_path, _audio, _sr) in enumerate(segments, start=1):
        try:
            segs, _info = _WHISPER_MODEL.transcribe(  # type: ignore[union-attr]
                str(wav_path), language="en", vad_filter=False
            )
            text = " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            log.warning("Whisper failed on segment %d: %s", idx, e)
            text = ""
        transcripts.append(text)
    return transcripts


# ---------------- Train script rendering ----------------


def _render_f5_script(training_dir: Path, slug: str) -> str:
    """Render the F5-TTS finetune shell script.

    F5-TTS exposes two ways to fine-tune:
      (a) `accelerate launch src/f5_tts/train/finetune_cli.py …` — the
          preferred CLI, takes a dataset dir + learning_rate args
      (b) The Gradio app's Train tab — UI-driven, same underlying config

    We codegen (a) because the user shouldn't need another web app
    running for this. Defaults are tuned for "cosplay a voice on a Mac
    from ~30 min of audio" rather than "train from scratch".
    """
    return f"""#!/usr/bin/env bash
# Deep Clone training — F5-TTS fine-tune
# Voice: {slug}
# Dataset dir: {training_dir}
#
# Expected runtime on Apple Silicon (M2, MPS, 16 GB):
#   ~6–10 hours for 30 min of training audio, 1000 steps, batch 2
# Expected runtime on CUDA (RTX 3090+):
#   ~30–60 min for the same dataset
#
# Stop: Ctrl+C once in terminal — checkpoints auto-save every 200 steps,
#       so you can re-run this script to resume from the latest.
# Resume: rerun this script verbatim. finetune_cli.py auto-picks up the
#       most recent checkpoint in {training_dir}/checkpoints/.
#
# When training is done, Voice Studio's UI will auto-detect the final
# checkpoint and offer to register the voice as `deep:{slug}` in the
# Design tab.

set -euo pipefail

cd "{training_dir}"

# Install F5-TTS's training extras if missing.
python -c "import f5_tts.train.finetune_cli" 2>/dev/null || \\
    pip install "f5-tts[eval]>=0.3" accelerate

export PYTHONPATH="${{PYTHONPATH:-}}"
export TOKENIZERS_PARALLELISM=false

# MPS on Apple Silicon is faster than CPU for this model, but PyTorch's
# MPS backend still OOMs on float32 attention — force float16 via
# PYTORCH_ENABLE_MPS_FALLBACK=1 to let the few missing ops fall back
# to CPU without crashing.
export PYTORCH_ENABLE_MPS_FALLBACK=1

accelerate launch --num_processes=1 \\
    -m f5_tts.train.finetune_cli \\
    --exp_name F5TTS_Base \\
    --learning_rate 1e-5 \\
    --batch_size_per_gpu 2 \\
    --batch_size_type frame \\
    --max_samples 64 \\
    --grad_accumulation_steps 1 \\
    --max_grad_norm 1.0 \\
    --epochs 10 \\
    --num_warmup_updates 200 \\
    --save_per_updates 200 \\
    --last_per_steps 200 \\
    --dataset_name "{slug}" \\
    --tokenizer char \\
    --finetune True \\
    --log_samples False \\
    --logger none

echo ""
echo "Training finished. Final checkpoint should be at:"
echo "  {training_dir}/checkpoints/model_last.pt"
echo ""
echo "Head back to Voice Studio's Deep Clone tab — the UI should now show"
echo "'Ready' and offer a 'Use in Design tab' button."
"""


def _render_xtts_script(training_dir: Path, slug: str) -> str:
    """Render the XTTS-v2 fine-tune shell script.

    Coqui's official XTTS fine-tune path is via their own training
    recipes, not `tts --continue_path`. The recipe needs a `metadata.csv`
    in LJSpeech format (which is exactly what we write) plus a dataset
    YAML. This script codegens the YAML and invokes the trainer.
    """
    return f"""#!/usr/bin/env bash
# Deep Clone training — XTTS-v2 fine-tune
# Voice: {slug}
# Dataset dir: {training_dir}
#
# Expected runtime on Apple Silicon (M2, MPS, 16 GB):
#   ~12–20 hours for 30 min of training audio, 20 epochs, batch 1
#   XTTS is heavier than F5 — if you can use CUDA, do.
# Expected runtime on CUDA (RTX 3090+):
#   ~1.5–3 hours for the same dataset
#
# Stop: Ctrl+C — checkpoints auto-save each epoch. Re-run this script to resume
#       (the trainer auto-picks up the latest checkpoint in {training_dir}/run/).
#
# When training is done, Voice Studio's UI will auto-detect the final
# checkpoint and offer to register the voice as `deep:{slug}` in the
# Design tab.

set -euo pipefail

cd "{training_dir}"

# Coqui's training extras come with the base `TTS` package but need a
# specific recipe file. We generate it inline below.
python - <<'PY'
import os
from pathlib import Path
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.xtts import Xtts

DATA_DIR = Path("{training_dir}")
RUN_DIR = DATA_DIR / "run"
RUN_DIR.mkdir(exist_ok=True)

config_dataset = BaseDatasetConfig(
    formatter="ljspeech",
    dataset_name="{slug}",
    path=str(DATA_DIR),
    meta_file_train="metadata.csv",
    language="en",
)

audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
config = XttsConfig(
    output_path=str(RUN_DIR),
    model_args={{"model_dir": "./", "temperature": 0.75, "length_penalty": 1.0, "repetition_penalty": 5.0, "top_k": 50, "top_p": 0.85}},
    run_name="xtts_ft_{slug}",
    project_name="voice-studio-deep-clone",
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=1,
    batch_group_size=48,
    eval_batch_size=1,
    num_loader_workers=4,
    eval_split_max_size=16,
    print_step=50,
    plot_step=100,
    log_model_step=1000,
    save_step=1000,
    save_n_checkpoints=3,
    save_checkpoints=True,
    print_eval=False,
    optimizer="AdamW",
    optimizer_wd_only_on_weights=True,
    optimizer_params={{"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2}},
    lr=5e-6,
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={{"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1}},
    test_sentences=[],
    epochs=20,
    datasets=[config_dataset],
)

train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=0.01,
)
model = Xtts.init_from_config(config)
trainer = Trainer(
    TrainerArgs(restore_path=None, skip_train_epoch=False),
    config,
    output_path=str(RUN_DIR),
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()
PY

echo ""
echo "Training finished. Final checkpoint should be at:"
echo "  {training_dir}/run/xtts_ft_{slug}/best_model.pth"
echo ""
echo "Head back to Voice Studio's Deep Clone tab — the UI should now show"
echo "'Ready' and offer a 'Use in Design tab' button."
"""


def _checkpoint_glob(training_dir: Path, engine: Engine) -> list[Path]:
    """Expected checkpoint locations for each engine, highest-precedence
    first. training_status() treats the first existing file as "ready"."""
    if engine == "f5":
        return [
            training_dir / "checkpoints" / "model_last.pt",
            training_dir / "checkpoints" / "model_best.pt",
        ]
    # xtts
    return sorted(training_dir.glob("run/**/best_model.pth")) + \
           sorted(training_dir.glob("run/**/checkpoint_*.pth"))


def write_train_script(training_dir: Path, engine: Engine, slug: str) -> Path:
    """Render + persist train.sh. Returns the path."""
    if engine == "f5":
        body = _render_f5_script(training_dir, slug)
    elif engine == "xtts":
        body = _render_xtts_script(training_dir, slug)
    else:
        raise ValueError(f"unknown engine '{engine}' (want 'f5' or 'xtts')")
    path = training_dir / "train.sh"
    path.write_text(body)
    path.chmod(0o755)
    return path


# ---------------- Public API ----------------


def prepare_dataset(
    audio_blobs: list[bytes],
    name: str,
    description: str = "",
    engine: Engine = "f5",
) -> TrainingManifest:
    """Synchronously prepare a training dataset.

    1. Concatenate + resample the input audio.
    2. Split into 3–10 s segments.
    3. Transcribe each with faster-whisper.
    4. Write segment_*.wav + metadata.csv + train.sh + manifest.json.

    Returns the manifest. Raises on empty audio or segmentation failure.
    """
    if not audio_blobs:
        raise ValueError("no audio files provided")
    if engine not in ("f5", "xtts"):
        raise ValueError(f"engine must be 'f5' or 'xtts', got {engine!r}")

    training_id = uuid.uuid4().hex[:12]
    slug = _slugify(name)
    training_dir = _training_dir(training_id)
    training_dir.mkdir(parents=True, exist_ok=True)
    wavs_dir = training_dir / "wavs"
    wavs_dir.mkdir(exist_ok=True)

    log.info("deep-clone prep[%s]: decoding + concatenating %d input(s)",
             training_id, len(audio_blobs))
    audio, sr = _concat_audio(audio_blobs)
    total_s = len(audio) / sr
    log.info("deep-clone prep[%s]: %.1f s total audio", training_id, total_s)

    log.info("deep-clone prep[%s]: segmenting…", training_id)
    ranges = _segment_by_silence(audio, sr)
    if not ranges:
        # Fallback: hard-split the whole thing at max_s intervals.
        step = int(MAX_SEGMENT_S * sr)
        ranges = [(i, min(i + step, len(audio))) for i in range(0, len(audio), step)]
    log.info("deep-clone prep[%s]: %d segments (target 3-10 s each)",
             training_id, len(ranges))

    # Write segment WAVs.
    segment_paths: list[tuple[Path, np.ndarray, int]] = []
    for i, (a, b) in enumerate(ranges, start=1):
        seg_audio = audio[a:b]
        wav_path = wavs_dir / f"segment_{i:04d}.wav"
        _write_wav(wav_path, seg_audio, sr)
        segment_paths.append((wav_path, seg_audio, sr))

    # Transcribe.
    log.info("deep-clone prep[%s]: transcribing %d segments with whisper…",
             training_id, len(segment_paths))
    transcripts = _transcribe_segments(segment_paths)

    # Write LJSpeech-style metadata.csv:
    #   <basename>|<raw_transcript>|<normalized_transcript>
    # Both F5-TTS and Coqui's LJSpeech formatter accept this shape. We
    # use only the basename (no extension) for the wav column because
    # LJSpeech formatter resolves paths relative to `wavs/`.
    csv_path = training_dir / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for (wav_path, _a, _sr), text in zip(segment_paths, transcripts):
            basename = wav_path.stem  # "segment_0001"
            writer.writerow([basename, text, text])

    # Render train.sh.
    train_script = write_train_script(training_dir, engine, slug)

    checkpoint_glob = (
        f"{training_dir}/checkpoints/model_*.pt"
        if engine == "f5"
        else f"{training_dir}/run/**/best_model.pth"
    )

    manifest = TrainingManifest(
        id=training_id,
        name=name.strip(),
        description=description.strip(),
        engine=engine,
        created_at=_now(),
        segment_count=len(segment_paths),
        total_duration_s=round(total_s, 2),
        train_script=str(train_script),
        checkpoint_glob=checkpoint_glob,
        status="prepared",
        notes=[
            f"{len(segment_paths)} segments averaging "
            f"{total_s / max(1, len(segment_paths)):.1f} s each.",
            "Run train.sh to kick off fine-tuning. Takes hours on CPU/MPS.",
        ],
    )
    _manifest_path(training_id).write_text(json.dumps(asdict(manifest), indent=2))
    log.info("deep-clone prep[%s]: ready at %s", training_id, training_dir)
    return manifest


def _read_manifest(training_id: str) -> Optional[TrainingManifest]:
    mp = _manifest_path(training_id)
    if not mp.exists():
        return None
    try:
        data = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return None
    return TrainingManifest(**data)


def _save_manifest(m: TrainingManifest) -> None:
    _manifest_path(m.id).write_text(json.dumps(asdict(m), indent=2))


def get_training(training_id: str) -> Optional[TrainingManifest]:
    return _read_manifest(training_id)


def list_trainings() -> list[TrainingManifest]:
    out: list[TrainingManifest] = []
    if not TRAINING_DIR.exists():
        return out
    for p in sorted(TRAINING_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        m = _read_manifest(p.name)
        if m is not None:
            # Refresh live status each listing so a checkpoint that just
            # appeared on disk shows as "ready" even without polling.
            refreshed = _refresh_status(m)
            if refreshed.status != m.status or refreshed.checkpoint_path != m.checkpoint_path:
                _save_manifest(refreshed)
            out.append(refreshed)
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


def delete_training(training_id: str) -> bool:
    d = _training_dir(training_id)
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True


def _refresh_status(m: TrainingManifest) -> TrainingManifest:
    """Check the filesystem for a checkpoint and flip status if found.

    We never downgrade from "ready" back to "training" — once a checkpoint
    exists, even if the user later deletes it, we keep the manifest
    pointing at the last known location (they can always delete the job).
    """
    if m.status == "ready" and m.checkpoint_path and Path(m.checkpoint_path).exists():
        return m
    td = _training_dir(m.id)
    if not td.exists():
        m.status = "failed"
        m.error = "training dir vanished"
        return m
    candidates = _checkpoint_glob(td, m.engine)  # type: ignore[arg-type]
    for c in candidates:
        if c.exists():
            m.status = "ready"
            m.checkpoint_path = str(c)
            m.error = None
            return m
    # If train.sh process wrote anything under run/ or checkpoints/, treat
    # that as "training in progress".
    run_dir = td / ("checkpoints" if m.engine == "f5" else "run")
    if run_dir.exists() and any(run_dir.iterdir()):
        m.status = "training"
    else:
        # Status stays at "prepared" until user runs train.sh.
        if m.status not in ("failed", "training", "ready"):
            m.status = "prepared"
    return m


def training_status(training_id: str) -> Optional[dict]:
    m = _read_manifest(training_id)
    if m is None:
        return None
    refreshed = _refresh_status(m)
    if refreshed.status != m.status or refreshed.checkpoint_path != m.checkpoint_path:
        _save_manifest(refreshed)
    return {
        "id": refreshed.id,
        "status": refreshed.status,
        "segments": refreshed.segment_count,
        "duration_s": refreshed.total_duration_s,
        "checkpoint_path": refreshed.checkpoint_path,
        "error": refreshed.error,
    }


def read_train_script(training_id: str) -> Optional[str]:
    path = _training_dir(training_id) / "train.sh"
    if not path.exists():
        return None
    return path.read_text()


# ---------------- Deep-clone registration ----------------
#
# Once a checkpoint exists, the user can "register" the voice so it shows
# up as a selectable base voice in the Design tab's dropdown. We stash a
# lightweight record in `data/training/_registry.json` — `tts.resolve_preset`
# reads this to convert `deep:<slug>` IDs back to (checkpoint_path, ref_text).

REGISTRY_PATH = TRAINING_DIR / "_registry.json"


def _read_registry() -> dict[str, dict]:
    if not REGISTRY_PATH.exists():
        return {}
    try:
        return json.loads(REGISTRY_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def _write_registry(reg: dict[str, dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2))


def register_training(training_id: str) -> dict:
    """Promote a 'ready' training job to a selectable deep-clone voice.

    Writes to the registry so the Design tab's base-voice dropdown picks
    it up on next render. The ID is `deep:<slug>-<training_id[:6]>` so
    two trainings with the same display name don't collide.
    """
    m = _read_manifest(training_id)
    if m is None:
        raise ValueError(f"training '{training_id}' not found")
    refreshed = _refresh_status(m)
    if refreshed.status != "ready" or not refreshed.checkpoint_path:
        raise ValueError(
            f"training '{training_id}' is '{refreshed.status}', not 'ready'. "
            f"Wait for train.sh to produce a checkpoint before registering."
        )
    _save_manifest(refreshed)

    slug = _slugify(refreshed.name)
    deep_id = f"deep:{slug}-{training_id[:6]}"
    # Reference clip: first segment WAV is a good zero-shot fallback for
    # engines that can't load our fine-tuned checkpoint. F5's engine API
    # in this repo hasn't been wired to load external .pt checkpoints yet;
    # this keeps the surface coherent until that lands.
    ref_candidates = sorted((_training_dir(training_id) / "wavs").glob("segment_*.wav"))
    ref_path = str(ref_candidates[0]) if ref_candidates else ""

    # Pull the transcript for the reference clip out of metadata.csv so
    # F5's zero-shot path can use it verbatim.
    ref_text = ""
    meta = _training_dir(training_id) / "metadata.csv"
    if meta.exists() and ref_candidates:
        first_stem = ref_candidates[0].stem
        for row in meta.read_text().splitlines():
            parts = row.split("|")
            if parts and parts[0] == first_stem and len(parts) > 1:
                ref_text = parts[1].strip()
                break

    reg = _read_registry()
    reg[deep_id] = {
        "id": deep_id,
        "training_id": training_id,
        "name": refreshed.name,
        "description": refreshed.description,
        "engine": refreshed.engine,
        "checkpoint_path": refreshed.checkpoint_path,
        "ref_audio_path": ref_path,
        "ref_text": ref_text,
        "registered_at": _now(),
    }
    _write_registry(reg)
    return reg[deep_id]


def list_deep_clones() -> list[dict]:
    """List all registered deep-clone voices. Used by tts.resolve_preset
    and exposed to the Design tab as selectable base voices."""
    reg = _read_registry()
    out = list(reg.values())
    out.sort(key=lambda r: r.get("registered_at", ""), reverse=True)
    return out


def resolve_deep_clone(deep_id: str) -> tuple[str, str]:
    """Return (ref_audio_path, ref_text) for a `deep:*` ID.

    Currently this hands back the first training segment as the reference
    clip — equivalent to zero-shot cloning of a very clean sample of the
    target voice. Once engines.py grows a `load_checkpoint` hook, this is
    where we'll route to the fine-tuned weights instead.
    """
    reg = _read_registry()
    entry = reg.get(deep_id)
    if not entry:
        raise ValueError(f"unknown deep-clone id: {deep_id}")
    ref = entry.get("ref_audio_path") or ""
    if not ref or not Path(ref).exists():
        raise FileNotFoundError(
            f"reference clip for {deep_id} not found at {ref}. "
            f"Delete and re-prepare the training job."
        )
    return ref, entry.get("ref_text", "")
