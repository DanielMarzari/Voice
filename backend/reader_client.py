"""
Client for pushing voice profiles to the Reader server.

The Reader endpoint is `POST /api/voices` which accepts multipart:
  - field `metadata`: JSON blob with {id, name, kind, engine, createdAt, design}
  - field `sample`:   audio/mpeg file (the preview MP3)
  - field `cover`:    image/* file (optional profile picture)
  - header Authorization: Bearer <token>

Configuration sources, highest priority first:
  1. Runtime overrides set by set_config() — typed in the Settings gear
     in Voice Studio's header. Persisted to backend/data/config.json so
     the user only has to paste credentials once.
  2. Environment variables READER_BASE_URL / READER_AUTH_TOKEN (from
     .env.local loaded at startup).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx

from profiles import VoiceProfile

log = logging.getLogger(__name__)


class ReaderConfigError(RuntimeError):
    """Raised when READER_BASE_URL or READER_AUTH_TOKEN aren't configured."""


class ReaderUploadError(RuntimeError):
    """Raised on non-2xx response from the Reader server."""


# Persisted override store. Lives next to the rest of the user's local
# data (gitignored) so the token stays put across backend restarts. Mode
# 0600 ensures other users on the machine can't read it.
_CONFIG_PATH = Path(__file__).parent / "data" / "config.json"

# Runtime overrides — initialized from _CONFIG_PATH at import time.
_override_base: Optional[str] = None
_override_token: Optional[str] = None

# Extra settings stored alongside the reader config. Kept here (not in a
# separate file) so one config.json round-trips the full backend state.
_settings: dict = {
    "render_worker_enabled": True,
    "overnight_only": False,
}


def _load_persisted() -> None:
    global _override_base, _override_token
    try:
        if _CONFIG_PATH.exists():
            raw = json.loads(_CONFIG_PATH.read_text())
            b = raw.get("reader_base_url")
            t = raw.get("reader_auth_token")
            if b:
                _override_base = b.rstrip("/")
            if t:
                _override_token = t
            if isinstance(raw.get("settings"), dict):
                _settings.update(raw["settings"])
            log.info("Loaded persisted reader config from %s", _CONFIG_PATH)
    except Exception as e:
        log.warning("Couldn't read %s (%s) — falling back to env only", _CONFIG_PATH, e)


def _save_persisted() -> None:
    """Write current overrides to disk. 0600 perms so only the local user
    can read the token. Writes atomically via a temp file + replace."""
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reader_base_url": _override_base,
            "reader_auth_token": _override_token,
            "settings": _settings,
        }
        tmp = _CONFIG_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.chmod(tmp, 0o600)
        tmp.replace(_CONFIG_PATH)
    except Exception as e:
        log.warning("Couldn't persist reader config: %s", e)


def get_settings() -> dict:
    """Read-only snapshot of the persisted settings (worker toggles etc.)."""
    return dict(_settings)


def update_settings(patch: dict) -> dict:
    """Merge `patch` into persisted settings, write to disk, return the
    updated dict. Unknown keys are accepted but intended to be one of
    the known worker knobs."""
    _settings.update(patch)
    _save_persisted()
    return dict(_settings)


# Load persisted config at module import so later requests see it.
_load_persisted()


def set_config(base_url: Optional[str], auth_token: Optional[str]) -> None:
    """Update the override store and persist to disk. Passing None for
    either field clears that override (re-reads the env value).

    Note on security: the token is written to backend/data/config.json
    with mode 0600. That path is gitignored. If you want secrets to live
    only in env vars, clear the overrides and use .env.local."""
    global _override_base, _override_token
    _override_base = base_url.rstrip("/") if base_url else None
    _override_token = auth_token if auth_token else None
    _save_persisted()


def _config() -> tuple[str, str]:
    base = (_override_base
            or os.environ.get("READER_BASE_URL", "")).rstrip("/")
    token = _override_token or os.environ.get("READER_AUTH_TOKEN", "")
    if not base or not token:
        raise ReaderConfigError(
            "Reader not configured. Click the settings gear in the Voice "
            "Studio header and paste your Reader URL + token."
        )
    return base, token


def is_configured() -> bool:
    try:
        _config()
        return True
    except ReaderConfigError:
        return False


def current_base_url() -> Optional[str]:
    """For displaying in the Settings UI — may differ from env if user
    overrode it at runtime."""
    return _override_base or os.environ.get("READER_BASE_URL") or None


def token_is_set() -> bool:
    """Returns True if a token is configured, without exposing it."""
    return bool(_override_token or os.environ.get("READER_AUTH_TOKEN"))


def upload_profile(
    profile: VoiceProfile,
    cover_path: Optional[Path] = None,
) -> dict:
    """POST a profile + its sample.mp3 (+ optional cover image) to Reader.
    Returns the server's JSON response on success.

    `cover_path` is an absolute path to an image file (PNG/JPEG/WebP/GIF).
    When present, it's shipped as a third multipart field and Reader uses
    it as the profile picture instead of rendering the dynamic sphere.
    """
    base, token = _config()

    sample_path = profile.sample_path()
    if not sample_path.exists():
        raise FileNotFoundError(f"sample.mp3 missing for profile {profile.id}")

    # Reader stores `design` as a JSON blob (voice_profiles.meta_json),
    # so packing prompt_text + duration_s in there avoids a Reader-side
    # schema migration. Reader's ZipVoice inference path will read them
    # back out of design when synthesizing. Keep top-level design fields
    # untouched for back-compat with existing designed voices.
    design_payload = dict(profile.design)
    if profile.prompt_text is not None:
        design_payload["prompt_text"] = profile.prompt_text
    if profile.duration_s is not None:
        design_payload["prompt_duration_s"] = profile.duration_s
    if profile.prompt_mel_frames is not None:
        design_payload["prompt_mel_frames"] = profile.prompt_mel_frames

    payload = {
        "id": profile.id,
        "name": profile.name,
        "kind": profile.kind,
        "engine": profile.engine,
        "createdAt": profile.created_at,
        "design": design_payload,
    }

    headers = {"Authorization": f"Bearer {token}"}

    sample_f = sample_path.open("rb")
    cover_f = cover_path.open("rb") if cover_path and cover_path.exists() else None
    # Ship the prompt_mel Float32 blob + its metadata JSON as two
    # additional multipart fields. Reader saves them alongside sample.mp3
    # in the voice's storage dir and exposes them via
    # /api/voices/<id>/prompt-mel{,-meta}, which the browser client
    # fetches to feed fm_decoder's `speech_condition`.
    prompt_mel_path = profile.prompt_mel_path()
    prompt_mel_meta_path = profile.prompt_mel_meta_path()
    prompt_mel_f = (
        prompt_mel_path.open("rb") if prompt_mel_path.exists() else None
    )
    prompt_mel_meta_f = (
        prompt_mel_meta_path.open("rb") if prompt_mel_meta_path.exists() else None
    )
    try:
        files = [
            ("metadata", (None, json.dumps(payload), "application/json")),
            ("sample", ("sample.mp3", sample_f, "audio/mpeg")),
        ]
        if cover_f is not None:
            assert cover_path is not None
            ctype = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(cover_path.suffix.lower(), "application/octet-stream")
            files.append(("cover", (f"cover{cover_path.suffix.lower()}", cover_f, ctype)))
        if prompt_mel_f is not None:
            files.append(
                (
                    "prompt_mel",
                    ("prompt_mel.f32", prompt_mel_f, "application/octet-stream"),
                )
            )
        if prompt_mel_meta_f is not None:
            files.append(
                (
                    "prompt_mel_meta",
                    ("prompt_mel_meta.json", prompt_mel_meta_f, "application/json"),
                )
            )

        try:
            r = httpx.post(
                f"{base}/api/voices",
                files=files,
                headers=headers,
                timeout=60.0,
            )
        except httpx.HTTPError as e:
            raise ReaderUploadError(f"Network error to {base}: {e}") from e
    finally:
        sample_f.close()
        if cover_f is not None:
            cover_f.close()
        if prompt_mel_f is not None:
            prompt_mel_f.close()
        if prompt_mel_meta_f is not None:
            prompt_mel_meta_f.close()

    if r.status_code // 100 != 2:
        raise ReaderUploadError(
            f"Reader server returned {r.status_code}: {r.text[:300]}"
        )
    return r.json()


def delete_profile(profile_id: str) -> None:
    """DELETE /api/voices/<id> on the server. Swallows 404 (already gone)."""
    base, token = _config()
    try:
        r = httpx.delete(
            f"{base}/api/voices/{profile_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        raise ReaderUploadError(f"Network error to {base}: {e}") from e
    if r.status_code == 404:
        return
    if r.status_code // 100 != 2:
        raise ReaderUploadError(
            f"Reader server returned {r.status_code}: {r.text[:300]}"
        )
