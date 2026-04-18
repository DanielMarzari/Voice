"""
Client for pushing voice profiles to the Reader server.

The Reader endpoint is `POST /api/voices` which accepts multipart:
  - field `metadata`: JSON blob with {id, name, kind, engine, createdAt, design}
  - field `sample`:   audio/mpeg file (the preview MP3)
  - header Authorization: Bearer <token>

The token is generated from Reader → Voice Lab → "Generate Studio Token".
Stored hashed server-side, shown once at creation.

Configuration sources, highest priority first:
  1. Runtime overrides set by set_config() — typed in the Settings gear
     in Voice Studio's header.
  2. Environment variables READER_BASE_URL / READER_AUTH_TOKEN (from
     .env.local loaded at startup).

This means the user can paste credentials in the UI without editing
files or restarting the backend.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import httpx

from profiles import VoiceProfile

log = logging.getLogger(__name__)


class ReaderConfigError(RuntimeError):
    """Raised when READER_BASE_URL or READER_AUTH_TOKEN aren't configured."""


class ReaderUploadError(RuntimeError):
    """Raised on non-2xx response from the Reader server."""


# Runtime overrides set via POST /api/config. When None, we fall back to env.
_override_base: Optional[str] = None
_override_token: Optional[str] = None


def set_config(base_url: Optional[str], auth_token: Optional[str]) -> None:
    """Update the in-memory config. Pass None for either to clear its
    override (re-reads the env value for that field). Does NOT persist
    to disk — a backend restart reverts to env-only config, which is the
    desired behavior so secrets stay out of committed files."""
    global _override_base, _override_token
    _override_base = base_url.rstrip("/") if base_url else None
    _override_token = auth_token if auth_token else None


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


def upload_profile(profile: VoiceProfile) -> dict:
    """POST a profile + its sample.mp3 to the Reader server.
    Returns the server's JSON response on success."""
    base, token = _config()

    sample_path = profile.sample_path()
    if not sample_path.exists():
        raise FileNotFoundError(f"sample.mp3 missing for profile {profile.id}")

    payload = {
        "id": profile.id,
        "name": profile.name,
        "kind": profile.kind,
        "engine": profile.engine,
        "createdAt": profile.created_at,
        "design": profile.design,
    }

    with sample_path.open("rb") as f:
        files = {
            "metadata": (None, json.dumps(payload), "application/json"),
            "sample":   ("sample.mp3", f, "audio/mpeg"),
        }
        headers = {"Authorization": f"Bearer {token}"}
        try:
            r = httpx.post(
                f"{base}/api/voices",
                files=files,
                headers=headers,
                timeout=60.0,
            )
        except httpx.HTTPError as e:
            raise ReaderUploadError(f"Network error to {base}: {e}") from e

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
