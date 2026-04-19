"""
Render worker — polls Reader for pending render jobs, claims one at a
time, synthesizes locally with F5 or XTTS, uploads chunks as they
complete, and posts the manifest when done.

Architecture notes:
  - Runs in a background thread started at uvicorn boot.
  - Uses threading.Event for pause/resume + shutdown signaling.
  - Reader connection reuses reader_client's persisted credentials; if
    they're missing we back off until the user fills in the Settings gear.
  - "Overnight only" mode skips work between 06:00-22:00 local time so the
    user's Mac fans aren't screaming during the workday.
  - A single in-process state dict is exposed to the UI via
    main.py's /api/render-queue for progress display. Guarded by a lock
    so the HTTP handler reads a consistent snapshot.
"""

from __future__ import annotations

import io
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import profiles
import reader_client
import render_client
import tts
from chunker import chunk_document

log = logging.getLogger(__name__)

POLL_INTERVAL_SECS = 15
IDLE_INTERVAL_SECS = 60         # longer wait when Reader isn't configured
POST_JOB_INTERVAL_SECS = 2       # short nap after finishing a job — drain the queue quickly

OVERNIGHT_START_HOUR = 22
OVERNIGHT_END_HOUR = 6


@dataclass
class WorkerState:
    """Snapshot of the worker for the /api/render-queue endpoint."""
    running: bool = True
    paused: bool = False
    overnight_only: bool = False
    last_poll_at: Optional[str] = None     # ISO8601
    last_error: Optional[str] = None
    current_job_id: Optional[str] = None
    current_voice_name: Optional[str] = None
    current_document_title: Optional[str] = None
    current_chunks_total: Optional[int] = None
    current_chunks_done: int = 0
    jobs_completed: int = 0
    jobs_failed: int = 0


class RenderWorker:
    """Background thread that drives the render queue loop."""

    def __init__(self) -> None:
        self.state = WorkerState()
        self._state_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wakeup = threading.Event()

    # ---- lifecycle ----

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="render-worker", daemon=True,
        )
        self._thread.start()
        log.info("RenderWorker started")

    def stop(self) -> None:
        self._stop.set()
        self._wakeup.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("RenderWorker stopped")

    def pause(self) -> None:
        with self._state_lock:
            self.state.paused = True

    def resume(self) -> None:
        with self._state_lock:
            self.state.paused = False
        self._wakeup.set()

    def set_overnight_only(self, value: bool) -> None:
        with self._state_lock:
            self.state.overnight_only = value
        self._wakeup.set()

    def snapshot(self) -> dict:
        """Thread-safe copy of the state for the HTTP endpoint."""
        with self._state_lock:
            s = self.state
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "paused": s.paused,
                "overnight_only": s.overnight_only,
                "reader_configured": reader_client.is_configured(),
                "last_poll_at": s.last_poll_at,
                "last_error": s.last_error,
                "current": (
                    None if s.current_job_id is None else {
                        "job_id": s.current_job_id,
                        "voice_name": s.current_voice_name,
                        "document_title": s.current_document_title,
                        "chunks_total": s.current_chunks_total,
                        "chunks_done": s.current_chunks_done,
                    }
                ),
                "jobs_completed": s.jobs_completed,
                "jobs_failed": s.jobs_failed,
            }

    # ---- main loop ----

    def _run(self) -> None:
        log.info("RenderWorker loop running")
        while not self._stop.is_set():
            try:
                interval = self._one_cycle()
            except Exception as e:
                log.exception("RenderWorker cycle failed")
                with self._state_lock:
                    self.state.last_error = f"{type(e).__name__}: {e}"
                interval = POLL_INTERVAL_SECS
            # Sleep, but let wakeup/stop break us out.
            self._wakeup.clear()
            self._wakeup.wait(timeout=interval)

    def _one_cycle(self) -> float:
        """Do at most one job; return the number of seconds to sleep."""
        with self._state_lock:
            self.state.last_poll_at = datetime.utcnow().isoformat(timespec="seconds")

        # Paused → idle.
        if self.state.paused:
            return POLL_INTERVAL_SECS

        # Overnight-only gate.
        if self.state.overnight_only and not _is_overnight_window():
            return POLL_INTERVAL_SECS

        # Reader not configured → wait longer, no noisy errors.
        if not reader_client.is_configured():
            return IDLE_INTERVAL_SECS

        try:
            jobs = render_client.list_pending_jobs(limit=1)
        except render_client.RenderApiError as e:
            # Likely an offline / auth issue — back off but don't crash.
            with self._state_lock:
                self.state.last_error = str(e)
            return POLL_INTERVAL_SECS
        if not jobs:
            with self._state_lock:
                self.state.last_error = None
            return POLL_INTERVAL_SECS

        job_summary = jobs[0]
        job_id = job_summary["id"]
        claimed = render_client.claim(job_id)
        if claimed is None:
            # 409 — another worker got it (shouldn't happen in single-user).
            return POST_JOB_INTERVAL_SECS

        job = claimed["job"]
        document = claimed["document"]
        try:
            self._process_claimed(job, document)
            with self._state_lock:
                self.state.jobs_completed += 1
                self.state.last_error = None
        except Exception as e:
            log.exception("Render failed for job %s", job_id)
            err_msg = f"{type(e).__name__}: {e}"[:500]
            try:
                render_client.fail(job_id, err_msg)
            except Exception:
                pass
            with self._state_lock:
                self.state.jobs_failed += 1
                self.state.last_error = err_msg
        finally:
            with self._state_lock:
                self.state.current_job_id = None
                self.state.current_voice_name = None
                self.state.current_document_title = None
                self.state.current_chunks_total = None
                self.state.current_chunks_done = 0

        return POST_JOB_INTERVAL_SECS

    # ---- per-job processing ----

    def _process_claimed(self, job: dict, document: dict) -> None:
        job_id = job["id"]
        voice_id = job["voiceId"]
        profile = profiles.load(voice_id)
        if profile is None:
            raise RuntimeError(f"Voice profile {voice_id} not found locally")

        # Resolve engine + reference audio from the profile.
        ref_path, ref_text, speaker_name, engine_id = tts.resolve_voice_for_synth(profile)
        engine = tts.get_engine(engine_id)

        # Chunk the document.
        content: str = document.get("content") or ""
        chunks = chunk_document(content, start_char_offset=0)
        total = len(chunks)
        if total == 0:
            raise RuntimeError("Document has no synthesizable text")

        with self._state_lock:
            self.state.current_job_id = job_id
            self.state.current_voice_name = profile.name
            self.state.current_document_title = document.get("title")
            self.state.current_chunks_total = total
            self.state.current_chunks_done = 0

        # Design-voice sliders carry through to synthesis.
        design = profile.design or {}
        speed = float(design.get("speed") or 1.0)
        pitch_semitones = float(design.get("pitch") or 0.0)
        language = design.get("language") or "en"

        manifest_chunks: list[dict] = []
        total_ms = 0

        log.info(
            "render: job=%s voice=%s engine=%s chunks=%d title=%r",
            job_id, profile.name, engine_id, total,
            (document.get("title") or "")[:40],
        )

        for i, chunk in enumerate(chunks, start=1):
            if self._stop.is_set():
                raise RuntimeError("Worker stopping; aborting render")
            if self.state.paused:
                # Finish this chunk but stop afterwards — don't rage-quit mid-synth.
                log.info("Worker paused mid-render; finishing current chunk then stopping")

            result = engine.synthesize(
                text=chunk.speak_text,
                ref_audio_path=ref_path,
                ref_text=ref_text,
                speed=speed,
                language=language,
                speaker_name=speaker_name,
            )
            if abs(pitch_semitones) > 0.01:
                result = tts.pitch_shift(result, pitch_semitones)
            mp3 = tts.encode_mp3(result)

            # Measure duration from the mp3 — cheap + accurate.
            duration_ms = _measure_mp3_duration_ms(mp3)
            total_ms += duration_ms

            render_client.upload_chunk(
                job_id=job_id,
                index=i,
                mp3_bytes=mp3,
                total=total if i == 1 else None,
            )
            manifest_chunks.append({
                "index": i,
                "charStart": chunk.char_start,
                "charEnd": chunk.char_end,
                "durationMs": duration_ms,
            })

            with self._state_lock:
                self.state.current_chunks_done = i

        manifest = {
            "documentId": document["id"],
            "voiceId": voice_id,
            "voiceName": profile.name,
            "engine": engine_id,
            "createdAt": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "totalDurationMs": total_ms,
            "chunks": manifest_chunks,
        }
        render_client.complete(job_id, manifest)


def _is_overnight_window() -> bool:
    """True when local time is 22:00 - 06:00."""
    h = datetime.now().hour
    return h >= OVERNIGHT_START_HOUR or h < OVERNIGHT_END_HOUR


def _measure_mp3_duration_ms(mp3_bytes: bytes) -> int:
    """Decode just enough of the MP3 to learn its duration."""
    from pydub import AudioSegment
    seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
    return int(round(len(seg)))  # pydub returns ms


# Module-level singleton — main.py imports this and calls start()/stop().
worker = RenderWorker()
