"""
Voice Studio — FastAPI entrypoint.

Routes:
  GET  /api/health              — liveness + server-config status
  GET  /api/engines             — list TTS engines with availability
  GET  /api/presets             — list the design preset speakers
  POST /api/clone               — clone a voice from an uploaded audio clip
                                  (save=true → JSON profile; save=false → MP3 bytes)
  POST /api/design              — persist a designed voice → JSON profile
  POST /api/design/preview      — synth a designed voice without saving → MP3 bytes
  POST /api/import              — store user-provided audio as a voice (no synth)
  POST /api/deep-clone/prepare  — Stage 1 of deep cloning: segment + transcribe
  GET  /api/deep-clone          — list prepared training jobs
  GET  /api/deep-clone/{id}     — single training manifest + live status
  GET  /api/deep-clone/{id}/script — stream the rendered train.sh as text
  DELETE /api/deep-clone/{id}   — remove a training job on disk
  POST /api/deep-clone/{id}/register — promote a ready checkpoint to a deep: voice
  GET  /api/deep-clones         — list registered deep-clone voices
  GET  /api/profiles            — list local profiles
  GET  /api/profiles/{id}       — fetch one profile
  GET  /api/profiles/{id}/sample — stream the preview MP3
  DELETE /api/profiles/{id}     — delete local + remote
  POST /api/profiles/{id}/sync  — re-push to Reader if missing there
  POST /api/synthesize          — generate audio for arbitrary text with
                                  a saved voice (standalone synth helper)
  GET  /api/synthesize/voices   — voices Reader can use for synthesis
  GET  /api/render-queue        — render worker snapshot (UI polling)
  POST /api/render-queue/pause
  POST /api/render-queue/resume
  POST /api/render-queue/settings — overnight_only / render_worker_enabled
  POST /api/render-queue/test   — confirm Reader bearer token works

All routes bind to 127.0.0.1 only via start.sh. CORS open to localhost
for the Next.js frontend.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

import mel_features
import profiles
import reader_client
import render_worker
import training
import transcribe
import tts

# Zero-shot voice cloning (ZipVoice-Distill) needs a reference-clip
# transcript + sufficient prompt duration to carry voice identity.
# Spike D (see spikes/spike_d_zero_shot/) established that prompts < ~8s
# degrade noticeably. We enforce 10s as a safe minimum.
MIN_PROMPT_DURATION_S = 10.0

# Load .env.local from the repo root (one level up from backend/).
ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("voice-studio")

app = FastAPI(title="Voice Studio", version="0.1.0")


@app.on_event("startup")
def _start_render_worker() -> None:
    # Apply persisted preferences before the worker starts polling.
    s = reader_client.get_settings()
    render_worker.worker.set_overnight_only(bool(s.get("overnight_only", False)))
    if bool(s.get("render_worker_enabled", True)):
        render_worker.worker.start()
    else:
        render_worker.worker.pause()


@app.on_event("shutdown")
def _stop_render_worker() -> None:
    render_worker.worker.stop()


# CORS is deliberately wide-open for local/known origins:
#   - localhost:3007 → the Voice Studio frontend itself
#   - localhost:3006 → Reader running in dev next to us
#   - reader.danmarzari.com → Reader's deployed UI. When the user is on
#     their Mac reading from reader.danmarzari.com and Voice Studio is up,
#     Reader's browser fetches this backend directly for TTS. Browsers
#     allow HTTPS → http://localhost cross-origin as long as CORS matches.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3007", "http://127.0.0.1:3007",
        "http://localhost:3006", "http://127.0.0.1:3006",
        "https://reader.danmarzari.com",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Models --------

class HealthResponse(BaseModel):
    ok: bool
    reader_configured: bool
    reader_base_url: Optional[str]
    reader_token_set: bool
    device: str
    default_engine: str
    engines: list[dict]


class ConfigUpdate(BaseModel):
    reader_base_url: Optional[str] = None
    reader_auth_token: Optional[str] = None


# Default preview text — one prompt to rule them all. Chosen because it:
#   - exercises a broad range of vowels and consonants,
#   - pairs question intonation with declarative reply (prosody!),
#   - reads in ~10 s at normal speaking rate (long enough to evaluate,
#     short enough that nobody's waiting around),
#   - sounds conversational without being weirdly self-aware.
DEFAULT_PREVIEW_TEXT = (
    "Hi there! I'm your new voice. Do I sound natural, clear, and "
    "expressive? There's plenty more where this came from."
)


class DesignRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    base_voice: str
    engine: str = Field("xtts")  # "f5" or "xtts"
    language: str = Field("en", min_length=2, max_length=5)
    pitch: float = Field(0.0, ge=-6.0, le=6.0)      # semitones
    speed: float = Field(1.0, ge=0.5, le=2.0)
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    # XTTS only: built-in speaker name. When set, overrides reference-clip
    # cloning — gives us explicit access to female / male / accented voices.
    speaker_name: Optional[str] = None
    # Optional 4-color palette computed by the frontend from the sliders.
    # Shipped to Reader with the profile so the sphere there matches the
    # one the user saw here when saving.
    colors: Optional[list[str]] = None
    preview_text: str = Field(DEFAULT_PREVIEW_TEXT, max_length=400)


class ProfileResponse(BaseModel):
    id: str
    name: str
    kind: str
    engine: str
    created_at: str
    design: dict
    synced: bool
    # Optional because profiles created before Phase 1's Clone gate
    # (or "designed" voices which don't have a reference clip at all)
    # won't have these populated.
    prompt_text: Optional[str] = None
    duration_s: Optional[float] = None
    prompt_mel_frames: Optional[int] = None


def _to_response(p: profiles.VoiceProfile) -> ProfileResponse:
    return ProfileResponse(
        id=p.id, name=p.name, kind=p.kind, engine=p.engine,
        created_at=p.created_at, design=p.design, synced=p.synced,
        prompt_text=p.prompt_text, duration_s=p.duration_s,
        prompt_mel_frames=p.prompt_mel_frames,
    )


# -------- Routes --------

@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        ok=True,
        reader_configured=reader_client.is_configured(),
        reader_base_url=reader_client.current_base_url(),
        reader_token_set=reader_client.token_is_set(),
        device=os.environ.get("TTS_DEVICE", "mps"),
        default_engine=tts.DEFAULT_ENGINE,
        engines=tts.list_engines(),
    )


@app.post("/api/config")
def update_config(cfg: ConfigUpdate):
    """Update the in-memory Reader connection. Lives only in the running
    backend process — restart reverts to .env.local values. This keeps
    secrets out of any persisted file."""
    reader_client.set_config(cfg.reader_base_url, cfg.reader_auth_token)
    return {
        "reader_configured": reader_client.is_configured(),
        "reader_base_url": reader_client.current_base_url(),
        "reader_token_set": reader_client.token_is_set(),
    }


@app.get("/api/engines")
def list_engines():
    """Returns the catalog of available engines with load state.
    Frontend uses this for the engine picker."""
    return {"default": tts.DEFAULT_ENGINE, "engines": tts.list_engines()}


@app.get("/api/xtts/speakers")
def list_xtts_speakers():
    """List XTTS's built-in speakers (gendered) so the Design tab can
    offer a 'pick a base voice' dropdown with actual female/male options,
    not just one coral-ish cloned reference clip."""
    try:
        xtts = tts.get_engine("xtts")
    except ValueError:
        raise HTTPException(404, "XTTS engine not registered")
    if not xtts.is_available():
        raise HTTPException(501, "XTTS not installed (pip install TTS>=0.22)")
    try:
        speakers = xtts.list_speakers()  # type: ignore[attr-defined]
    except Exception as e:
        log.exception("xtts speakers")
        raise HTTPException(500, f"Couldn't load XTTS speakers: {e}")
    return {"speakers": speakers}


@app.get("/api/presets")
def list_presets():
    return {"presets": tts.PRESETS}


@app.get("/api/references")
def list_references():
    """User-uploaded reference clips (from the Design tab). Separate from
    the built-in mood presets — these are distinct reference voices the
    user wants to synthesize in."""
    return {"references": tts.list_user_references()}


@app.post("/api/references")
async def upload_reference(
    name: str = Form(..., min_length=1, max_length=60),
    audio_file: UploadFile = File(...),
    description: str = Form(""),
    transcript: str = Form(""),
):
    """Save a user-uploaded reference clip. Any common audio format works
    — it gets transcoded to 24 kHz mono WAV via ffmpeg. Description
    (e.g. 'Friendly, inviting and balanced') shows next to the name in
    the Base voice dropdown."""
    raw = await audio_file.read()
    if not raw:
        raise HTTPException(400, "audio_file is empty")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(413, "audio too large (max 10 MB)")
    try:
        info = tts.save_user_reference(
            name=name,
            audio_bytes=raw,
            description=description,
            transcript=transcript,
        )
    except Exception as e:
        log.exception("reference save failed")
        raise HTTPException(500, f"Save failed: {e}")
    return {"reference": info}


@app.delete("/api/references/{slug}")
def remove_reference(slug: str):
    if tts.delete_user_reference(slug):
        return {"deleted": slug}
    raise HTTPException(404, f"Reference '{slug}' not found")


# ---------------------------------------------------------------------
# Import — user brings their own finished audio clip, we persist it as a
# Reader voice profile verbatim. No synthesis involved. Optional cover
# image or mood palette for the sphere.
# ---------------------------------------------------------------------


@app.post("/api/import", response_model=ProfileResponse)
async def import_voice(
    name: str = Form(..., min_length=1, max_length=60),
    description: str = Form(""),
    audio_file: UploadFile = File(...),
    cover_image: Optional[UploadFile] = File(None),
    colors: str = Form(""),  # JSON-encoded 4-string array, optional
    upload: bool = Form(True),
):
    """Import a user-provided voice clip directly. Transcodes the input
    audio to MP3 via pydub/ffmpeg so Reader (and the browser) can play
    it, then persists it as a local profile and (if connected) ships it
    to Reader with the optional cover image."""
    raw_audio = await audio_file.read()
    if not raw_audio:
        raise HTTPException(400, "audio_file is empty")
    if len(raw_audio) > 20 * 1024 * 1024:
        raise HTTPException(413, "audio too large (max 20 MB)")

    # Parse colors if provided (empty string → None).
    color_list: Optional[list[str]] = None
    if colors.strip():
        try:
            parsed = json.loads(colors)
            if isinstance(parsed, list) and len(parsed) == 4:
                color_list = [str(c) for c in parsed]
        except Exception:
            log.warning("import: ignoring malformed colors JSON: %r", colors[:120])

    profile = profiles.VoiceProfile(
        id=profiles.new_id(),
        name=name.strip(),
        kind="uploaded",
        engine="user",
        created_at=profiles._now(),
        design={
            "description": description.strip() or None,
            "colors": color_list,
        },
    )
    profile.dir().mkdir(parents=True, exist_ok=True)

    # Transcode to MP3 (128kbps) and save as the sample.
    try:
        from pydub import AudioSegment
        import io as _io
        seg = AudioSegment.from_file(_io.BytesIO(raw_audio))
        buf = _io.BytesIO()
        seg.export(buf, format="mp3", bitrate="128k")
        profile.sample_path().write_bytes(buf.getvalue())
    except Exception as e:
        log.exception("import transcode failed")
        shutil.rmtree(profile.dir(), ignore_errors=True)
        raise HTTPException(500, f"Couldn't transcode audio: {e}")

    # Save cover image (original extension) if provided.
    cover_path: Optional[Path] = None
    if cover_image is not None and cover_image.filename:
        img_bytes = await cover_image.read()
        if len(img_bytes) > 0:
            if len(img_bytes) > 5 * 1024 * 1024:
                raise HTTPException(413, "cover image too large (max 5 MB)")
            ext = os.path.splitext(cover_image.filename)[1].lower() or ".png"
            if ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                raise HTTPException(400, f"unsupported cover format: {ext}")
            cover_path = profile.dir() / f"cover{ext}"
            cover_path.write_bytes(img_bytes)

    profiles.save(profile)

    if upload and reader_client.is_configured():
        try:
            reader_client.upload_profile(profile, cover_path=cover_path)
            profile.synced = True
            profiles.save(profile)
        except Exception as e:
            log.warning("Upload to Reader failed: %s", e)

    return _to_response(profile)


@app.post("/api/clone")
async def clone_voice(
    name: str = Form("Preview", min_length=1, max_length=60),
    audio_file: UploadFile = File(...),
    engine_id: str = Form("xtts"),
    language: str = Form("en"),
    preview_text: str = Form(DEFAULT_PREVIEW_TEXT),
    ref_text: Optional[str] = Form(None),
    upload: bool = Form(True),
    save: bool = Form(True),
):
    """Clone a voice from an uploaded audio clip.

    If save=True (default), creates a local profile and (if upload=True)
    pushes metadata + sample to Reader. Returns the ProfileResponse JSON.

    If save=False, synthesizes but does NOT persist anything — returns
    the raw audio/mpeg bytes. Used by the "Preview" button on the frontend
    so the user can listen before committing.
    """
    try:
        engine = tts.get_engine(engine_id)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # Read the uploaded clip once into memory. Both the save and preview
    # paths need it — the save path persists it as source.wav, the preview
    # path writes it to a tempfile for the synthesis engine.
    raw = await audio_file.read()
    if not raw:
        raise HTTPException(400, "audio_file is empty")
    src_ext = os.path.splitext(audio_file.filename or "")[1].lower() or ".wav"

    if not save:
        # Preview-only path. Write to a tempfile, synthesize, stream bytes back.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=src_ext, delete=False) as tf:
            tf.write(raw)
            tmp_path = tf.name
        try:
            result = engine.synthesize(
                text=preview_text,
                ref_audio_path=tmp_path,
                ref_text=ref_text,
                speed=1.0,
                language=language,
            )
            mp3 = tts.encode_mp3(result)
        except Exception as e:
            log.exception("Preview clone synthesis failed")
            raise HTTPException(500, f"Synthesis failed: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return Response(content=mp3, media_type="audio/mpeg")

    # Save path (default).
    profile = profiles.VoiceProfile(
        id=profiles.new_id(),
        name=name.strip(),
        kind="cloned",
        engine=engine.id,
        created_at=profiles._now(),
        design={"language": language},
    )
    profile.dir().mkdir(parents=True, exist_ok=True)
    src_path = profile.dir() / f"source{src_ext}"
    src_path.write_bytes(raw)

    # Gate 1: duration. Per Spike D, zero-shot voice identity transfer
    # needs >= 10 s of clean speech. We check after writing to disk so
    # ffprobe sees the real file (not the in-memory bytes).
    try:
        duration_s = transcribe.probe_duration(src_path)
    except Exception as e:
        shutil.rmtree(profile.dir(), ignore_errors=True)
        raise HTTPException(400, f"Could not read audio file: {e}")
    if duration_s < MIN_PROMPT_DURATION_S:
        shutil.rmtree(profile.dir(), ignore_errors=True)
        raise HTTPException(
            400,
            f"Reference clip is {duration_s:.1f}s — need at least "
            f"{MIN_PROMPT_DURATION_S:.0f}s for good zero-shot quality. "
            f"Record a longer clip (10–20 s of clean speech works best).",
        )
    profile.duration_s = round(duration_s, 2)

    # Gate 2: transcript. The user can supply one via ref_text; otherwise
    # we auto-transcribe with Whisper. Either way, the final transcript
    # lives on the profile for Reader to hand to ZipVoice at inference time.
    if ref_text and ref_text.strip():
        profile.prompt_text = ref_text.strip()
    else:
        log.info("Clone %s: auto-transcribing %.1fs reference…", profile.id, duration_s)
        try:
            auto = transcribe.transcribe_file(src_path, language=language)
        except transcribe.TranscribeError as e:
            log.warning("Auto-transcribe failed: %s", e)
            auto = None
        if auto is None:
            shutil.rmtree(profile.dir(), ignore_errors=True)
            raise HTTPException(
                400,
                "Auto-transcribe unavailable. Either install faster-whisper "
                "(`pip install -r backend/requirements.txt` in the backend venv) "
                "or paste what's being said in the Advanced → transcript box and retry.",
            )
        profile.prompt_text = auto
        log.info("Clone %s: transcript: %s", profile.id, auto[:100])

    try:
        result = engine.synthesize(
            text=preview_text,
            ref_audio_path=str(src_path),
            ref_text=profile.prompt_text,
            speed=1.0,
            language=language,
        )
        mp3 = tts.encode_mp3(result)
        profile.sample_path().write_bytes(mp3)
    except Exception as e:
        log.exception("Clone synthesis failed")
        shutil.rmtree(profile.dir(), ignore_errors=True)
        raise HTTPException(500, f"Synthesis failed: {e}")

    # Gate 3: prompt_mel. Reader's browser inference needs the reference
    # clip's log-mel spectrogram as `speech_condition` — without it the
    # flow-matching model has no voice identity to clone from and emits
    # noise. Compute matches ZipVoice's VocosFbank exactly via
    # mel_features.compute_prompt_mel (same helper the one-off CLI uses).
    try:
        mel_result = mel_features.compute_prompt_mel(
            src_path, out_dir=profile.dir()
        )
        profile.prompt_mel_frames = mel_result.num_frames
        log.info(
            "Clone %s: prompt_mel %d frames (%.1f KB)",
            profile.id, mel_result.num_frames, mel_result.byte_size / 1024,
        )
    except Exception as e:
        log.warning("prompt_mel computation failed: %s", e)
        # Don't abort the whole clone — the voice still works for
        # Voice Studio's own preview synthesis (which uses F5/XTTS, not
        # ZipVoice). Reader's browser inference won't be able to use
        # this voice until prompt_mel is recomputed, but that's
        # recoverable via scripts/compute_prompt_mel.py.

    profiles.save(profile)

    if upload and reader_client.is_configured():
        try:
            reader_client.upload_profile(profile)
            profile.synced = True
            profiles.save(profile)
        except Exception as e:
            log.warning("Upload to Reader failed: %s", e)
            # Local profile still saved — user can retry via /sync.

    return _to_response(profile)


def _synthesize_design(req: "DesignRequest") -> bytes:
    """Core synthesis used by both save and preview paths. Raises HTTPException
    on failure so the route handlers can pass-through the error.

    We log each phase separately so when a request dies we know which step
    actually blew up (ref resolution / engine load / synthesis / pitch / mp3).
    """
    log.info(
        "design: engine=%s lang=%s base=%s pitch=%.2f speed=%.2f temp=%.2f",
        req.engine, req.language, req.base_voice, req.pitch, req.speed, req.temperature,
    )

    try:
        ref_path, ref_text = tts.resolve_preset(req.base_voice)
        log.info("design: resolved preset → %s", ref_path)
    except (ValueError, FileNotFoundError) as e:
        log.warning("design: preset resolution failed: %s", e)
        raise HTTPException(400, str(e))

    try:
        engine = tts.get_engine(req.engine)
    except ValueError as e:
        raise HTTPException(400, str(e))

    try:
        if req.speaker_name:
            log.info(
                "design: synthesizing with %s on %s, speaker=%s…",
                engine.id, engine.device, req.speaker_name,
            )
        else:
            log.info(
                "design: synthesizing with %s on %s (from ref clip)…",
                engine.id, engine.device,
            )
        result = engine.synthesize(
            text=req.preview_text,
            ref_audio_path=ref_path,
            ref_text=ref_text,
            speed=req.speed,
            language=req.language,
            speaker_name=req.speaker_name,
        )
        log.info(
            "design: synth ok — %d samples @ %d Hz",
            len(result.audio), result.sample_rate,
        )
        if abs(req.pitch) > 0.01:
            result = tts.pitch_shift(result, req.pitch)
            log.info("design: pitch shifted by %.2f semitones", req.pitch)
        mp3 = tts.encode_mp3(result)
        log.info("design: encoded %d bytes of mp3", len(mp3))
        return mp3
    except HTTPException:
        raise
    except Exception as e:
        log.exception("design: synthesis failed")
        raise HTTPException(500, f"Synthesis failed: {e}")


@app.post("/api/design/preview")
def preview_design(req: DesignRequest):
    """Synthesize without persisting. Returns audio/mpeg bytes for in-browser
    playback. The user can tweak sliders and call this repeatedly before
    committing via POST /api/design."""
    mp3 = _synthesize_design(req)
    return Response(content=mp3, media_type="audio/mpeg")


@app.post("/api/design", response_model=ProfileResponse)
def design_voice(req: DesignRequest, upload: bool = True):
    """Persist a 'designed' voice: synthesize, save locally, optionally
    push to Reader."""
    try:
        engine = tts.get_engine(req.engine)
    except ValueError as e:
        raise HTTPException(400, str(e))

    mp3 = _synthesize_design(req)

    profile = profiles.VoiceProfile(
        id=profiles.new_id(),
        name=req.name.strip(),
        kind="designed",
        engine=engine.id,
        created_at=profiles._now(),
        design={
            "base_voice": req.base_voice,
            "pitch": req.pitch,
            "speed": req.speed,
            "temperature": req.temperature,
            "language": req.language,
            "speaker_name": req.speaker_name,
            # Store the exact palette the user saw so Reader can reproduce it.
            "colors": req.colors,
        },
    )
    profile.dir().mkdir(parents=True, exist_ok=True)
    profile.sample_path().write_bytes(mp3)
    profiles.save(profile)

    if upload and reader_client.is_configured():
        try:
            reader_client.upload_profile(profile)
            profile.synced = True
            profiles.save(profile)
        except Exception as e:
            log.warning("Upload to Reader failed: %s", e)

    return _to_response(profile)


# ---------------------------------------------------------------------
# Deep Clone — fine-tune a model on ~30 min of clean speech for
# ElevenLabs-grade consistency. Two-stage pipeline: Prepare (synchronous
# segment + transcribe) then Train (out-of-process shell script the user
# runs themselves).
# ---------------------------------------------------------------------


@app.post("/api/deep-clone/prepare")
async def deep_clone_prepare(
    name: str = Form(..., min_length=1, max_length=60),
    description: str = Form(""),
    engine: str = Form("f5"),
    audio_file: list[UploadFile] = File(...),
):
    """Stage 1: accept one or more long audio files, concatenate, VAD-split
    into 3–10 s segments, transcribe each with whisper, and persist to
    backend/data/training/<id>/. Returns the manifest (includes the path
    to a rendered train.sh the user runs themselves)."""
    if engine not in ("f5", "xtts"):
        raise HTTPException(400, f"engine must be 'f5' or 'xtts', got {engine!r}")
    if not audio_file:
        raise HTTPException(400, "at least one audio_file is required")

    blobs: list[bytes] = []
    total_bytes = 0
    # 500 MB cap across all files — generous for ~30 min of high-bitrate
    # audio, but prevents a malformed 2 GB upload from running us out of
    # RAM (we do decode in-memory).
    MAX_TOTAL = 500 * 1024 * 1024
    for upload in audio_file:
        raw = await upload.read()
        if not raw:
            continue
        total_bytes += len(raw)
        if total_bytes > MAX_TOTAL:
            raise HTTPException(413, "total audio too large (max 500 MB combined)")
        blobs.append(raw)

    if not blobs:
        raise HTTPException(400, "all uploaded audio_file parts were empty")

    try:
        manifest = training.prepare_dataset(
            audio_blobs=blobs,
            name=name,
            description=description,
            engine=engine,  # type: ignore[arg-type]
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("deep-clone prep failed")
        raise HTTPException(500, f"Dataset prep failed: {e}")

    return _manifest_dict(manifest)


@app.get("/api/deep-clone")
def list_deep_clone_jobs():
    return {"trainings": [_manifest_dict(m) for m in training.list_trainings()]}


@app.get("/api/deep-clone/{training_id}")
def get_deep_clone_job(training_id: str):
    m = training.get_training(training_id)
    if m is None:
        raise HTTPException(404, "Training not found")
    status = training.training_status(training_id) or {}
    body = _manifest_dict(m)
    body["status"] = status.get("status", body.get("status"))
    body["checkpoint_path"] = status.get("checkpoint_path")
    return body


@app.get("/api/deep-clone/{training_id}/script")
def get_deep_clone_script(training_id: str):
    body = training.read_train_script(training_id)
    if body is None:
        raise HTTPException(404, "train.sh not found")
    return Response(content=body, media_type="text/plain")


@app.delete("/api/deep-clone/{training_id}")
def delete_deep_clone_job(training_id: str):
    if training.delete_training(training_id):
        return {"deleted": training_id}
    raise HTTPException(404, "Training not found")


@app.post("/api/deep-clone/{training_id}/register")
def register_deep_clone(training_id: str):
    """Once train.sh has produced a checkpoint, call this to make the
    voice show up as `deep:<slug>` in the Design tab's base-voice dropdown."""
    try:
        entry = training.register_training(training_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return entry


@app.get("/api/deep-clones")
def list_registered_deep_clones():
    return {"deep_clones": training.list_deep_clones()}


def _manifest_dict(m: training.TrainingManifest) -> dict:
    from dataclasses import asdict
    return asdict(m)


@app.get("/api/profiles")
def list_profiles():
    return {"profiles": [_to_response(p).model_dump() for p in profiles.list_all()]}


@app.get("/api/profiles/{profile_id}", response_model=ProfileResponse)
def get_profile(profile_id: str):
    p = profiles.load(profile_id)
    if p is None:
        raise HTTPException(404, "Profile not found")
    return _to_response(p)


@app.get("/api/profiles/{profile_id}/sample")
def get_sample(profile_id: str):
    p = profiles.load(profile_id)
    if p is None or not p.sample_path().exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(p.sample_path(), media_type="audio/mpeg")


@app.delete("/api/profiles/{profile_id}")
def delete_profile(profile_id: str):
    p = profiles.load(profile_id)
    if p is None:
        raise HTTPException(404, "Profile not found")
    # Best-effort remote delete first; don't block local delete on failure.
    if reader_client.is_configured():
        try:
            reader_client.delete_profile(profile_id)
        except Exception as e:
            log.warning("Remote delete failed (will delete locally anyway): %s", e)
    profiles.delete(profile_id)
    return {"deleted": profile_id}


@app.post("/api/profiles/{profile_id}/sync", response_model=ProfileResponse)
def sync_profile(profile_id: str):
    p = profiles.load(profile_id)
    if p is None:
        raise HTTPException(404, "Profile not found")
    if not reader_client.is_configured():
        raise HTTPException(400, "Reader not configured (set READER_BASE_URL + READER_AUTH_TOKEN)")
    try:
        reader_client.upload_profile(p)
    except Exception as e:
        raise HTTPException(502, f"Upload failed: {e}")
    p.synced = True
    profiles.save(p)
    return _to_response(p)


# ---------------------------------------------------------------------
# Synthesize arbitrary text with a saved profile. This is the entry point
# Reader hits for read-aloud — it's self-hosted TTS using whichever
# engine the profile was saved with (F5 or XTTS), no external APIs.
# ---------------------------------------------------------------------


class SynthesisRequest(BaseModel):
    voice_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1, max_length=2000)
    # Speechify-style context hints. F5 and XTTS don't expose prev/next text
    # fields the way ElevenLabs does, so these are currently informational
    # (we log them and they'd be useful for a future prepend/strip trick to
    # smooth chunk boundaries). Accepting them keeps Reader's client code
    # engine-agnostic.
    previous_text: Optional[str] = None
    next_text: Optional[str] = None
    speed: float = Field(1.0, ge=0.5, le=2.5)
    language: str = Field("en", min_length=2, max_length=5)


def _resolve_voice_reference(profile: profiles.VoiceProfile) -> tuple[str, Optional[str], Optional[str]]:
    """Return (ref_audio_path, ref_text, speaker_name) for a saved profile.

    - Cloned voices: use the original source.<ext> clip from the profile dir.
    - Designed voices: re-resolve the base_voice preset (same clip used at
      design time; slider params re-applied below).
    - Uploaded voices: we don't synthesize new text for these; caller
      should handle the 400 before calling us.
    """
    if profile.kind == "cloned":
        # Clone stored source.wav/.mp3/.m4a in its dir. Find whichever is there.
        dir_ = profile.dir()
        sources = [p for p in dir_.iterdir() if p.stem == "source" and p.is_file()]
        if not sources:
            raise HTTPException(
                500,
                f"Cloned profile {profile.id} missing source audio — delete + re-clone.",
            )
        return str(sources[0]), None, None

    if profile.kind == "designed":
        design = profile.design or {}
        speaker_name = design.get("speaker_name")  # XTTS-only, optional
        base_voice = design.get("base_voice")
        if not base_voice:
            raise HTTPException(500, "Designed profile missing design.base_voice")
        ref_path, ref_text = tts.resolve_preset(base_voice)
        return ref_path, ref_text, speaker_name

    raise HTTPException(
        400,
        f"Voice kind '{profile.kind}' can't synthesize new text. "
        f"Imported voices only hold a static clip.",
    )


@app.post("/api/synthesize")
def synthesize(req: SynthesisRequest):
    """Generate audio for `text` using a saved voice profile. Returns
    audio/mpeg bytes. Called by Reader's chunked streaming player when
    the user picks the 'Voice Studio' engine."""
    profile = profiles.load(req.voice_id)
    if profile is None:
        raise HTTPException(404, f"No voice profile {req.voice_id}")

    try:
        ref_path, ref_text, speaker_name = _resolve_voice_reference(profile)
    except HTTPException:
        raise

    # Engine is whatever the profile was created with ("f5" or "xtts").
    # Designed profiles persist their engine; cloned ones do too. Fall
    # back to the default engine if somehow missing (shouldn't happen).
    engine_id = profile.engine or tts.DEFAULT_ENGINE
    # Normalize legacy engine strings like "f5-tts" → "f5".
    engine_id = {"f5-tts": "f5"}.get(engine_id, engine_id)

    try:
        engine = tts.get_engine(engine_id)
    except ValueError as e:
        raise HTTPException(500, f"Engine '{engine_id}' not registered: {e}")

    # Honor designed-voice slider params (speed/pitch). If the client sent
    # a speed override, the client wins; otherwise use the profile's saved
    # speed. For pitch we honor the profile value and apply post-synth.
    design = profile.design or {}
    effective_speed = req.speed if req.speed and req.speed != 1.0 else float(design.get("speed", 1.0))
    pitch_semitones = float(design.get("pitch", 0.0))

    log.info(
        "synth: voice=%s engine=%s kind=%s speed=%.2f pitch=%.2f text_len=%d%s",
        req.voice_id, engine_id, profile.kind, effective_speed, pitch_semitones,
        len(req.text),
        f" prev={len(req.previous_text)}" if req.previous_text else "",
    )

    try:
        result = engine.synthesize(
            text=req.text,
            ref_audio_path=ref_path,
            ref_text=ref_text,
            speed=effective_speed,
            language=req.language,
            speaker_name=speaker_name,
        )
        if abs(pitch_semitones) > 0.01:
            result = tts.pitch_shift(result, pitch_semitones)
        mp3 = tts.encode_mp3(result)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("synthesize failed")
        raise HTTPException(500, f"Synthesis failed: {e}")

    return Response(content=mp3, media_type="audio/mpeg")


@app.get("/api/synthesize/voices")
def list_synthesize_voices():
    """Voices that can synthesize new text (cloned + designed). Imports
    are excluded — they're static audio clips with no engine to re-invoke.
    Reader hits this to populate its voice picker when the 'Voice Studio'
    engine is selected."""
    out = []
    for p in profiles.list_all():
        if p.kind == "uploaded":
            continue
        out.append({
            "id": p.id,
            "name": p.name,
            "kind": p.kind,
            "engine": p.engine,
        })
    return {"voices": out}


# ---------------------------------------------------------------------
# Render queue — status + control for the background worker that drains
# Reader's render_jobs table.
# ---------------------------------------------------------------------


class RenderQueueSettings(BaseModel):
    render_worker_enabled: Optional[bool] = None
    overnight_only: Optional[bool] = None


@app.get("/api/render-queue")
def get_render_queue():
    """Snapshot of the worker: running/paused, current job + progress,
    completed/failed counters, whether Reader is reachable. The Queue
    tab polls this every few seconds to render its UI."""
    snapshot = render_worker.worker.snapshot()
    snapshot["settings"] = reader_client.get_settings()
    return snapshot


@app.post("/api/render-queue/pause")
def render_queue_pause():
    render_worker.worker.pause()
    reader_client.update_settings({"render_worker_enabled": False})
    return render_worker.worker.snapshot()


@app.post("/api/render-queue/resume")
def render_queue_resume():
    render_worker.worker.resume()
    reader_client.update_settings({"render_worker_enabled": True})
    return render_worker.worker.snapshot()


@app.post("/api/render-queue/settings")
def render_queue_settings(patch: RenderQueueSettings):
    """Update persisted worker knobs. Pass only the fields you want to
    change; others stay put."""
    update: dict = {}
    if patch.render_worker_enabled is not None:
        update["render_worker_enabled"] = bool(patch.render_worker_enabled)
    if patch.overnight_only is not None:
        update["overnight_only"] = bool(patch.overnight_only)
    reader_client.update_settings(update)
    # Push live into the worker so the change takes effect without restart.
    if "render_worker_enabled" in update:
        if update["render_worker_enabled"]:
            render_worker.worker.resume()
        else:
            render_worker.worker.pause()
    if "overnight_only" in update:
        render_worker.worker.set_overnight_only(update["overnight_only"])
    snapshot = render_worker.worker.snapshot()
    snapshot["settings"] = reader_client.get_settings()
    return snapshot


@app.post("/api/render-queue/test")
def render_queue_test():
    """Manual 'ping' — hits Reader's pending-jobs endpoint so the user
    can confirm the bearer token works. Returns {count} or an error."""
    try:
        jobs = render_client_list_pending()
    except Exception as e:
        raise HTTPException(502, f"Reader not reachable: {e}")
    return {"ok": True, "pending_count": len(jobs)}


def render_client_list_pending() -> list[dict]:
    """Indirection layer so /render-queue/test doesn't import
    render_client at module-top — keeps the import graph clean."""
    import render_client as rc
    return rc.list_pending_jobs(limit=5)
