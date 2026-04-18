# Design presets

`backend/tts.py` references preset reference clips for the "Design a voice" flow:

- `amber.wav`
- `cobalt.wav`
- `rose.wav`
- `slate.wav`

Each should be ~5 seconds of clean, dry speech at 24 kHz mono.

Drop any four WAV files with those names here (or record your own). They are intentionally **not** committed to git because voice presets are a personal preference — keep the samples you like best on your machine.

If a preset file is missing, `/api/design` will return a 400 with a clear message pointing back here.
