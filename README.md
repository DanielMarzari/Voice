# Voice Studio

Local voice cloning / voice design GUI for the [Reader](https://github.com/DanielMarzari/Reader) app.

Runs on your Mac (tested on Apple Silicon M2). Does the heavy ML locally so the $0/month Oracle server stays light — only the finished voice profile + preview MP3 get uploaded.

## What it does

- **Clone a voice** from a 5-30 second audio sample using [F5-TTS](https://github.com/SWivid/F5-TTS) (Apache 2.0, zero-shot).
- **Design a voice** from presets + pitch/speed/temperature sliders.
- Saves profiles locally under `backend/data/profiles/<id>/`.
- Uploads profile metadata + a preview MP3 to your Reader server so it shows up in the Voice Lab tab.

## Architecture

```
localhost:3007  (Next.js UI)  ──fetch──▶  localhost:8000  (FastAPI + F5-TTS)
                                                │
                                                ├─── writes: backend/data/profiles/<id>/{source.wav, sample.mp3, meta.json}
                                                │
                                                └─── POST $READER_BASE_URL/api/voices   (uploads meta + sample.mp3)
```

The Python backend loads F5-TTS lazily on first synthesis so startup stays snappy. It then stays in memory between requests.

## First-time setup

Requirements: macOS with Apple Silicon, Python 3.11+ (via `brew install python@3.11`), Node 20+, and `ffmpeg`.

```bash
# System deps
brew install ffmpeg python@3.11

# Env
cp .env.example .env.local
# Then edit .env.local: set READER_BASE_URL and paste the token from
# Reader → Voice Lab → "Generate Studio Token"

# Backend
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

## Run

From repo root:

```bash
./start.sh
```

That spawns FastAPI on :8000 and Next.js on :3007, then traps SIGINT so Ctrl-C cleans up both.

Open <http://localhost:3007>.

## Voice cloning quality tips

- Sample length: **10–20 seconds** of clean speech. One continuous clip beats many short ones.
- No background music, no reverb, no overlapping voices.
- WAV or MP3 both fine; gets resampled to 24 kHz mono internally.
- The first clone after startup takes ~30s (model download + warm-up). Subsequent clones are ~5–10s on M2.

## Licensing notes

- **F5-TTS**: Apache 2.0.
- XTTS-v2 is intentionally NOT used — its Coqui Public Model License is non-commercial. If you want to experiment with it locally, `backend/tts.py` has a commented-out adapter.
