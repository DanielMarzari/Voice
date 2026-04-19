"""
HTTP client for Reader's render-job API. Thin wrapper — the render_worker
loop calls these functions. All requests carry the bearer token that
reader_client stores in config.json.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

import reader_client

log = logging.getLogger(__name__)


class RenderApiError(RuntimeError):
    """Non-2xx response from a render-job endpoint."""
    def __init__(self, status: int, body: str):
        super().__init__(f"Reader render API {status}: {body[:300]}")
        self.status = status
        self.body = body


def _base_and_token() -> tuple[str, str]:
    # Reuse reader_client's _config — same credentials, same persisted store.
    # Raises ReaderConfigError when missing, which the worker catches to
    # back off until the user fills in the Settings gear.
    return reader_client._config()  # pyright: ignore[reportPrivateUsage]


def _auth_headers() -> dict[str, str]:
    _, token = _base_and_token()
    return {"Authorization": f"Bearer {token}"}


def list_pending_jobs(limit: int = 1, timeout: float = 10.0) -> list[dict]:
    """GET /api/render-jobs?status=pending&limit=N. Returns the raw jobs
    list (may be empty). Orders: priority desc, requested_at asc."""
    base, _ = _base_and_token()
    try:
        r = httpx.get(
            f"{base}/api/render-jobs",
            params={"status": "pending", "limit": limit},
            headers=_auth_headers(),
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise RenderApiError(0, f"network: {e}") from e
    if r.status_code // 100 != 2:
        raise RenderApiError(r.status_code, r.text)
    data = r.json()
    return data.get("jobs") or []


def claim(job_id: str, timeout: float = 10.0) -> Optional[dict]:
    """POST /api/render-jobs/:id/claim. On success returns
    {job, document:{id,title,content}}. Returns None on 409
    (someone else got there first)."""
    base, _ = _base_and_token()
    try:
        r = httpx.post(
            f"{base}/api/render-jobs/{job_id}/claim",
            headers=_auth_headers(),
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise RenderApiError(0, f"network: {e}") from e
    if r.status_code == 409:
        return None
    if r.status_code // 100 != 2:
        raise RenderApiError(r.status_code, r.text)
    return r.json()


def upload_chunk(
    job_id: str,
    index: int,
    mp3_bytes: bytes,
    *,
    total: Optional[int] = None,
    timeout: float = 60.0,
) -> dict:
    """POST /api/render-jobs/:id/chunks (multipart). Returns updated job row."""
    base, _ = _base_and_token()
    files = {
        "mp3": (f"chunk_{index:04d}.mp3", mp3_bytes, "audio/mpeg"),
    }
    data = {"index": str(index)}
    if total is not None:
        data["total"] = str(total)
    try:
        r = httpx.post(
            f"{base}/api/render-jobs/{job_id}/chunks",
            headers=_auth_headers(),
            data=data,
            files=files,
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise RenderApiError(0, f"network: {e}") from e
    if r.status_code // 100 != 2:
        raise RenderApiError(r.status_code, r.text)
    return r.json()


def complete(job_id: str, manifest: dict, timeout: float = 30.0) -> dict:
    """POST /api/render-jobs/:id/complete. Body = manifest JSON; Reader
    persists it to disk + flips the job status to 'ready'."""
    base, _ = _base_and_token()
    try:
        r = httpx.post(
            f"{base}/api/render-jobs/{job_id}/complete",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            content=json.dumps(manifest),
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise RenderApiError(0, f"network: {e}") from e
    if r.status_code // 100 != 2:
        raise RenderApiError(r.status_code, r.text)
    return r.json()


def fail(job_id: str, error: str, timeout: float = 10.0) -> dict:
    """POST /api/render-jobs/:id/fail. Called by the worker on exceptions."""
    base, _ = _base_and_token()
    try:
        r = httpx.post(
            f"{base}/api/render-jobs/{job_id}/fail",
            headers={**_auth_headers(), "Content-Type": "application/json"},
            content=json.dumps({"error": error[:500]}),
            timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise RenderApiError(0, f"network: {e}") from e
    if r.status_code // 100 != 2:
        raise RenderApiError(r.status_code, r.text)
    return r.json()
