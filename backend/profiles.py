"""
Local profile store.

Each voice profile is a directory under `backend/data/profiles/<id>/`:

  meta.json      — VoiceProfile as JSON
  source.wav     — original reference clip (cloned voices only)
  sample.mp3     — ~6s preview synthesized once at creation time

This store is the source of truth on the local machine. The server has
a copy of meta.json + sample.mp3 for listing/preview, but not source.wav
(we don't want the training audio leaving the user's machine).
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data" / "profiles"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VoiceKind = Literal["cloned", "designed"]


@dataclass
class VoiceProfile:
    id: str
    name: str
    kind: VoiceKind
    engine: str                       # "f5-tts" for Phase 1
    created_at: str                   # ISO8601 UTC
    # Design-only knobs (empty dict for cloned voices).
    design: dict = field(default_factory=dict)
    # "Did we upload this to the server yet?" — mirrors server state.
    synced: bool = False

    def dir(self) -> Path:
        return DATA_DIR / self.id

    def meta_path(self) -> Path:
        return self.dir() / "meta.json"

    def sample_path(self) -> Path:
        return self.dir() / "sample.mp3"

    def source_path(self) -> Path:
        return self.dir() / "source.wav"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def save(profile: VoiceProfile) -> None:
    profile.dir().mkdir(parents=True, exist_ok=True)
    profile.meta_path().write_text(json.dumps(asdict(profile), indent=2))


def load(profile_id: str) -> Optional[VoiceProfile]:
    meta = DATA_DIR / profile_id / "meta.json"
    if not meta.exists():
        return None
    try:
        raw = json.loads(meta.read_text())
    except json.JSONDecodeError:
        log.warning("Corrupt meta.json at %s", meta)
        return None
    return VoiceProfile(**raw)


def list_all() -> list[VoiceProfile]:
    out: list[VoiceProfile] = []
    if not DATA_DIR.exists():
        return out
    for p in sorted(DATA_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        v = load(p.name)
        if v is not None:
            out.append(v)
    out.sort(key=lambda v: v.created_at, reverse=True)
    return out


def delete(profile_id: str) -> bool:
    d = DATA_DIR / profile_id
    if not d.exists():
        return False
    shutil.rmtree(d)
    return True
