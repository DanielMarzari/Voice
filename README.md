# Voice Studio

Local voice cloning / voice design GUI for the [Reader](https://github.com/DanielMarzari/Reader) app.

Runs on your Mac (tested on Apple Silicon M2). Does the heavy ML locally so the $0/month Oracle server stays light — only the finished voice profile + preview MP3 get uploaded.

## What it does

- **Clone a voice** from a 5-30 second audio sample. Pick the engine per voice:
  - **XTTS-v2** (Coqui, CPML non-commercial, 17 languages) — default
  - **F5-TTS** (Apache 2.0, English + Chinese, 2024)
- **Design a voice** from presets + pitch/speed/temperature sliders, on either engine.
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

## Render queue (Reader audiobooks)

Once Voice Studio is connected to Reader (Settings gear → paste URL + token), a background worker polls Reader for pending render jobs. When you click **▶ Listen with…** on a document in Reader, it queues a render job; Voice Studio picks it up, synthesizes locally, and ships back MP3 chunks + word timings. Reader plays back pre-rendered audio from its own disk — no live synthesis, no localhost dance.

Control all of this from the **Queue** tab in Voice Studio:
- See what's currently rendering (doc title, voice, chunk progress).
- Pause / Resume the worker.
- Toggle **Overnight only** — worker stays idle 6 AM – 10 PM and burns through the queue while you sleep.
- **Test Reader connection** — one click to confirm the bearer token still works.

## Auto-start on login

Two options (pick one; they can coexist):

**LaunchAgent (headless, fire-and-forget):**
```bash
./scripts/install-autostart.sh
```
Creates `~/Library/LaunchAgents/com.danmarzari.voice-studio.plist`. Voice Studio starts on login, logs to `~/Library/Logs/voice-studio.{out,err}.log`, and auto-restarts if it crashes. Uninstall: `launchctl unload ~/Library/LaunchAgents/com.danmarzari.voice-studio.plist && rm "$_"`.

**Desktop .app (click to launch):**
```bash
./scripts/build-app.sh
```
Drops a `Voice Studio.app` on your Desktop. Double-clicking it opens a Terminal window with the backend running + Safari at `localhost:3007`. Drag to Applications and add to System Settings → Login Items for auto-start with a visible window.

## Voice cloning quality tips

- Sample length: **10–20 seconds** of clean speech. One continuous clip beats many short ones.
- No background music, no reverb, no overlapping voices.
- WAV or MP3 both fine; gets resampled to 24 kHz mono internally.
- The first clone after startup takes ~30s (model download + warm-up). Subsequent clones are ~5–10s on M2.

## Troubleshooting

**`Failed to proxy … socket hang up` / `ECONNRESET` in the dev-server log** — the frontend used to route requests through Next.js's rewrites, which times out at ~30s. The frontend now calls the backend directly on `127.0.0.1:8000` so this is gone. If you still hit it, make sure `src/lib/api.ts` has the direct `API_BASE` — see the comment at the top of that file.

**Backend crashes or gives a 500 partway through synthesis on Apple Silicon** — PyTorch's MPS backend has occasional bugs in STFT / istft paths (you'll see warnings like `An output with one or more elements was resized since it had shape []`). If you hit this, set `TTS_DEVICE=cpu` in `.env.local` and restart. Synthesis will be ~2× slower but 100% stable. XTTS falls back to CPU automatically if MPS `.to()` errors; F5-TTS does not, so the env var is the knob.

**`ImportError: cannot import name 'BeamSearchScorer' from 'transformers'`** — transformers 4.44+ removed the class that Coqui TTS imports. Pin the old version:
```bash
cd backend
source .venv/bin/activate
pip install "transformers==4.36.2" --force-reinstall
```
Then restart the backend. `requirements.txt` already pins this for fresh installs; `--force-reinstall` is the fix for venvs created before the pin landed.

**`Preset … is missing`** — shouldn't happen anymore; presets share a single reference clip that auto-downloads on first use. If the download can't reach GitHub, drop any ~5s clean-speech WAV at `backend/data/presets/base.wav` as a manual override.

## Licensing notes

- **F5-TTS**: Apache 2.0 — use freely.
- **XTTS-v2**: Coqui Public Model License — **non-commercial use only**. Fine for personal projects like this one. If you ever want to build a commercial product on top of Voice Studio, switch the default engine to F5 in `.env.local` (`TTS_ENGINE_DEFAULT=f5`) and don't expose XTTS in the picker.

The first call to XTTS downloads ~1.8 GB of weights. First call to F5 downloads ~1.3 GB. After that, each engine stays in RAM until the backend restarts — having both loaded uses ~3 GB.
