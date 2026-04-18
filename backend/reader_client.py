"""
Client for pushing voice profiles to the Reader server.

The Reader endpoint is `POST /api/voices` which accepts multipart:
  - field `metadata`: JSON blob with {id, name, kind, engine, createdAt, design}
  - field `sample`:   audio/mpeg file (the preview MP3)
  - header Authorization: Bearer <token>

The token is generated from Reader → Voice Lab → "Generate Studio Token".
Stored hashed server-side, shown once at creation.
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


def _config() -> tuple[str, str]:
    base = os.environ.get("READER_BASE_URL", "").rstrip("/")
    token = os.environ.get("READER_AUTH_TOKEN", "")
    if not base or not token:
        raise ReaderConfigError(
            "READER_BASE_URL and READER_AUTH_TOKEN must be set in .env.local. "
            "Generate a token from the Reader → Voice Lab UI."
        )
    return base, token


def is_configured() -> bool:
    try:
        _config()
        return True
    except ReaderConfigError:
        return False


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
