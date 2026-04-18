# Design presets

The Design tab picks a preset name + slider defaults, then synthesizes against a **single shared reference clip**. You don't need to supply per-preset WAVs.

On first use, the reference clip is resolved in priority order:

1. **`base.wav`** in this directory, if you dropped one. Recommended: ~5s of clean neutral speech at 24 kHz mono. Optional companion `base.txt` with the exact transcript (helps the model match prosody).
2. **F5-TTS's bundled example** from `pip install f5-tts` — works automatically if F5 is installed.
3. **Auto-downloaded** from the [F5-TTS GitHub repo](https://github.com/SWivid/F5-TTS) (Apache 2.0), cached as `fallback_ref_en.wav`.

The four presets (`amber`, `cobalt`, `rose`, `slate`) differ only in the default positions of the pitch / speed / temperature sliders on the Design tab. They all use the same reference clip at synthesis time, because what matters for voice character in a zero-shot TTS is the reference audio + the sliders — not a separate clip per preset.

If you want genuinely different base voices, drop in your own `base.wav` whenever you like. Or clone a voice from the Clone tab and use that profile directly (future: "design from existing cloned voice" will let you combine both).
