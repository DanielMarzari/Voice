#!/usr/bin/env bash
# Start Voice Studio: FastAPI backend on :8000, Next.js frontend on :3007.
# Ctrl-C cleanly kills both.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Load .env.local into the environment if present.
if [[ -f .env.local ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

BACKEND_VENV="$SCRIPT_DIR/backend/.venv"
if [[ ! -d "$BACKEND_VENV" ]]; then
  echo "▶ Missing backend/.venv — run: cd backend && python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

if [[ ! -d "$SCRIPT_DIR/frontend/node_modules" ]]; then
  echo "▶ Missing frontend/node_modules — run: cd frontend && npm install"
  exit 1
fi

cleanup() {
  echo ""
  echo "▶ Shutting down…"
  if [[ -n "${BACKEND_PID:-}" ]]; then kill "$BACKEND_PID" 2>/dev/null || true; fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then kill "$FRONTEND_PID" 2>/dev/null || true; fi
  wait 2>/dev/null || true
  exit 0
}
trap cleanup SIGINT SIGTERM

echo "▶ Starting FastAPI backend on :8000…"
(
  cd backend
  # shellcheck disable=SC1091
  source .venv/bin/activate
  exec uvicorn main:app --host 127.0.0.1 --port 8000 --reload
) &
BACKEND_PID=$!

echo "▶ Starting Next.js frontend on :3007…"
(
  cd frontend
  exec npm run dev -- --port 3007
) &
FRONTEND_PID=$!

echo ""
echo "▶ Voice Studio running."
echo "  Backend:  http://127.0.0.1:8000/docs"
echo "  Frontend: http://127.0.0.1:3007"
echo "  Ctrl-C to stop."
echo ""

wait
