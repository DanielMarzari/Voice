#!/usr/bin/env python3.11
"""Publish TTS assets to the Reader production server (dan-server).

The Reader browser-inference path needs two classes of files on the
server:

  1. Shared base ONNX bundle — every voice shares the same
     text_encoder.onnx, fm_decoder.onnx, vocos_fp16.onnx + tokens.txt +
     model.json. About 500 MB total. Published once per upstream change
     (ZipVoice retrain, vocos upgrade). Lives under
     `/var/www/apps/reader/public/tts-assets/shared/` on dan-server
     and is fronted by Caddy's immutable-cache rule.

  2. Per-voice prompt_mel data — the Float32 log-mel of each voice's
     reference clip + its JSON sidecar. Normally published via the
     HTTP push flow (POST /api/voices) from the Voice Studio UI's
     "Sync to Reader" button — see `reader_client.upload_profile`. No
     rsync needed for that case; this script skips it.

This script is the rsync path for (1). It uses the existing
`dan-server` SSH host (see ~/.ssh/config + Documents/Server/dan-server.key).

Staged-upload pattern:

  Server target dir `shared.staging/` receives the rsync first. We
  then compute sha256sum on both ends and bail if any file mismatches.
  Only when every hash matches do we atomic-rename:
      shared/         → shared.old/        (one quick mv)
      shared.staging/ → shared/            (one quick mv)
      shared.old/     → <deleted>          (cleanup)
  This means a mid-rsync kill never leaves the Reader serving a
  partial bundle.

Usage:

  python backend/publish.py shared
  python backend/publish.py shared --dry-run
  python backend/publish.py shared --source /Users/.../Reader/public/tts-assets/shared

  # List what's currently deployed:
  python backend/publish.py status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------- Defaults ----------

SSH_HOST = "dan-server"
SERVER_SHARED_DIR = "/var/www/apps/reader/public/tts-assets/shared"
SERVER_STAGING_DIR = "/var/www/apps/reader/public/tts-assets/shared.staging"
SERVER_OLD_DIR = "/var/www/apps/reader/public/tts-assets/shared.old"

# Reader's local source-of-truth symlink directory. publish.py follows
# the symlinks with rsync's --copy-links, so moving or renaming any of
# the underlying ONNX files in Voice (e.g. after a retrain) just works
# as long as the symlink still resolves.
DEFAULT_SOURCE = Path.home() / "Server-Sites" / "Reader" / "public" / "tts-assets" / "shared"

# Files we expect to find in the shared dir. Any file in the source
# dir not in this set is a surprise — we'll warn and ship it anyway.
# Any file in this set missing from the source is an error.
REQUIRED_SHARED_FILES = {
    "fm_decoder.onnx",
    "text_encoder.onnx",
    "vocos_fp16.onnx",
    "tokens.txt",
    "model.json",
}


# ---------- Helpers ----------


@dataclass
class FileRecord:
    """One shared file: its local path, size, and sha256. Used to
    cross-check what we uploaded against what landed on the server."""

    name: str
    local_path: Path
    size: int
    sha256: str


def sha256_of(path: Path) -> str:
    """Compute sha256 in 4 MB chunks so a 455 MB fm_decoder doesn't blow
    up memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(4 * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def collect_local_shared(source: Path) -> list[FileRecord]:
    """Walk the source dir, resolve symlinks, and build a FileRecord list.
    Errors out if any REQUIRED_SHARED_FILES are missing."""
    if not source.exists():
        raise SystemExit(f"Source dir does not exist: {source}")
    if not source.is_dir():
        raise SystemExit(f"Source path is not a directory: {source}")

    records: list[FileRecord] = []
    seen: set[str] = set()
    for child in sorted(source.iterdir()):
        if child.is_dir():
            # No nested dirs expected; skip defensively.
            continue
        # Resolve symlinks so we hash + ship the actual file, not the
        # link. The Reader dev box keeps these as symlinks; on the
        # server rsync --copy-links turns them into real files.
        real = child.resolve(strict=True)
        size = real.stat().st_size
        records.append(
            FileRecord(
                name=child.name,
                local_path=real,
                size=size,
                sha256=sha256_of(real),
            )
        )
        seen.add(child.name)

    missing = REQUIRED_SHARED_FILES - seen
    if missing:
        raise SystemExit(
            f"Source {source} is missing required file(s): {sorted(missing)}. "
            f"Has Voice Studio's ONNX export pipeline been run?"
        )
    extra = seen - REQUIRED_SHARED_FILES
    if extra:
        print(
            f"[publish] Note: shipping unexpected files too: {sorted(extra)}",
            file=sys.stderr,
        )
    return records


def ssh(cmd: str, *, capture: bool = False) -> str:
    """Run a command on dan-server via SSH. If capture, return stdout;
    otherwise stream to the parent tty. Raises on non-zero exit."""
    full = ["ssh", SSH_HOST, cmd]
    if capture:
        r = subprocess.run(full, check=True, capture_output=True, text=True)
        return r.stdout
    subprocess.run(full, check=True)
    return ""


def remote_sha256sums(remote_dir: str, names: list[str]) -> dict[str, str]:
    """Ask the server for `sha256sum <dir>/<name>` for each file and
    parse into {name: hex}. Missing files show up as empty string."""
    quoted = " ".join(
        shlex.quote(f"{remote_dir}/{n}") for n in names
    )
    # `|| true` so one missing file doesn't tank the whole check — we'll
    # cross-check the parsed dict and flag mismatches ourselves.
    raw = ssh(f"sha256sum {quoted} 2>/dev/null || true", capture=True)
    result: dict[str, str] = {}
    for line in raw.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        hex_digest, filepath = parts
        basename = Path(filepath).name
        result[basename] = hex_digest
    return result


# ---------- Commands ----------


def cmd_shared(args: argparse.Namespace) -> int:
    source = Path(args.source).expanduser().resolve()
    print(f"[publish] Source: {source}")
    print(f"[publish] Target: {SSH_HOST}:{SERVER_SHARED_DIR}")

    records = collect_local_shared(source)
    total_bytes = sum(r.size for r in records)
    print(f"[publish] {len(records)} files, {total_bytes / 1e6:.1f} MB total")
    for r in records:
        print(f"  {r.name:<24} {r.size / 1e6:>7.1f} MB  sha256={r.sha256[:12]}…")

    if args.dry_run:
        print("[publish] --dry-run: skipping rsync + mv")
        return 0

    # 1. Ensure server parent dir exists + clean any leftover staging.
    print("[publish] Preparing server staging dir…")
    ssh(
        f"mkdir -p {shlex.quote(str(Path(SERVER_SHARED_DIR).parent))} "
        f"&& rm -rf {shlex.quote(SERVER_STAGING_DIR)}"
    )

    # 2. rsync local → server staging. --copy-links materializes
    #    Reader's symlinks into real files; -a preserves perms, -h
    #    human sizes, --progress shows per-file progress. (macOS ships
    #    rsync 2.6.9 which doesn't understand --info=progress2, so we
    #    stick to the 2001-era flag set.)
    print("[publish] rsync → staging…")
    rsync_cmd = [
        "rsync",
        "-ah",
        "--copy-links",
        "--progress",
        f"{source}/",
        f"{SSH_HOST}:{SERVER_STAGING_DIR}/",
    ]
    subprocess.run(rsync_cmd, check=True)

    # 3. Verify sha256 on both ends.
    print("[publish] Verifying sha256 on server…")
    names = [r.name for r in records]
    remote = remote_sha256sums(SERVER_STAGING_DIR, names)
    mismatches: list[str] = []
    for r in records:
        got = remote.get(r.name, "")
        if got != r.sha256:
            mismatches.append(
                f"  {r.name}: local={r.sha256[:12]}… server={got[:12] or '(missing)'}…"
            )
    if mismatches:
        print("[publish] SHA-256 MISMATCH — aborting, staging left in place:", file=sys.stderr)
        for m in mismatches:
            print(m, file=sys.stderr)
        return 2
    print("[publish] ✓ all sha256 match")

    # 4. Atomic swap: shared → shared.old (if exists), staging → shared.
    #    Two quick mvs — the window where neither exists is microseconds.
    #    Browser/Service Worker already has the old bytes cached anyway.
    print("[publish] Swapping staging → live…")
    ssh(
        f"if [ -d {shlex.quote(SERVER_SHARED_DIR)} ]; then "
        f"  rm -rf {shlex.quote(SERVER_OLD_DIR)}; "
        f"  mv {shlex.quote(SERVER_SHARED_DIR)} {shlex.quote(SERVER_OLD_DIR)}; "
        f"fi && "
        f"mv {shlex.quote(SERVER_STAGING_DIR)} {shlex.quote(SERVER_SHARED_DIR)} && "
        f"rm -rf {shlex.quote(SERVER_OLD_DIR)}"
    )
    print("[publish] ✓ live")

    # 5. Print final manifest (belt & suspenders — confirms the swap
    #    actually took effect).
    print("[publish] Final server manifest:")
    ssh(f"ls -lh {shlex.quote(SERVER_SHARED_DIR)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Read-only summary of what's currently deployed."""
    print(f"[publish] {SSH_HOST}:{SERVER_SHARED_DIR}")
    try:
        ssh(
            f"if [ -d {shlex.quote(SERVER_SHARED_DIR)} ]; then "
            f"  ls -lh {shlex.quote(SERVER_SHARED_DIR)}; "
            f"  echo '---'; "
            f"  cd {shlex.quote(SERVER_SHARED_DIR)} && sha256sum * | head -20; "
            f"else "
            f"  echo '(not deployed — run: publish.py shared)'; "
            f"fi"
        )
    except subprocess.CalledProcessError as e:
        return e.returncode
    return 0


# ---------- Entry point ----------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Publish TTS assets to the Reader production server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    p_shared = sub.add_parser(
        "shared",
        help="Rsync the shared ONNX bundle (~500 MB) to dan-server.",
    )
    p_shared.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help=f"Source dir (default: {DEFAULT_SOURCE})",
    )
    p_shared.add_argument(
        "--dry-run",
        action="store_true",
        help="List files + hashes but skip rsync + swap.",
    )
    p_shared.set_defaults(func=cmd_shared)

    p_status = sub.add_parser(
        "status",
        help="Show what's currently deployed to dan-server.",
    )
    p_status.set_defaults(func=cmd_status)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
