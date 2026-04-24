"use client";

/**
 * ImportTab — "I already have an audio clip, just persist it as a voice."
 *
 * No synthesis involved. Takes an audio file (WAV/MP3/OGG/M4A/FLAC),
 * transcodes to MP3 server-side, and saves as a kind="uploaded" profile.
 * User chooses one visual representation:
 *   (a) a color mood for the dynamic sphere — same palette system as the
 *       Design tab; or
 *   (b) a profile picture — cropped to a circle in Reader's gallery.
 *
 * Result gets auto-synced to Reader if the Settings gear is configured.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { importVoice, type Profile } from "@/lib/api";
import { DropZone } from "@/components/DropZone";
import { MoodPicker } from "@/components/MoodPicker";
import { VoiceSphere } from "@/components/VoiceSphere";
import { MOODS, moodPalette, type Palette } from "@/lib/moodPalettes";

type VisualMode = "sphere" | "picture";

type Props = {
  onCreated: (p: Profile) => void;
};

export function ImportTab({ onCreated }: Props) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [audio, setAudio] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);

  const [visual, setVisual] = useState<VisualMode>("sphere");
  const [moodId, setMoodId] = useState<string>(MOODS[0].id);
  // Seed starts at 0 (SSR-safe) and randomizes post-mount — a `Math.random()`
  // initializer here would diverge between server and client renders and
  // trigger React's hydration warning on VoiceSphere's colors prop. Same
  // pattern as CloneTab / DesignTab.
  const [paletteSeed, setPaletteSeed] = useState<number>(0);
  useEffect(() => {
    setPaletteSeed(Math.random());
  }, []);
  const [cover, setCover] = useState<File | null>(null);
  const [coverUrl, setCoverUrl] = useState<string | null>(null);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Profile | null>(null);

  // Keep object-URL previews in sync with the uploaded file state.
  useEffect(() => {
    if (!audio) {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
      return;
    }
    const u = URL.createObjectURL(audio);
    setAudioUrl(u);
    return () => URL.revokeObjectURL(u);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [audio]);

  useEffect(() => {
    if (!cover) {
      if (coverUrl) URL.revokeObjectURL(coverUrl);
      setCoverUrl(null);
      return;
    }
    const u = URL.createObjectURL(cover);
    setCoverUrl(u);
    return () => URL.revokeObjectURL(u);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cover]);

  const livePalette = useMemo(
    () => moodPalette(moodId, paletteSeed),
    [moodId, paletteSeed]
  );

  const canSubmit = !!audio && !!name.trim() && !busy;

  const handleSubmit = useCallback(async () => {
    if (!audio || !name.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const p = await importVoice({
        name: name.trim(),
        description: description.trim() || undefined,
        audio,
        cover: visual === "picture" ? cover : null,
        colors: visual === "sphere" ? (livePalette as unknown as string[]) : null,
      });
      setResult(p);
      onCreated(p);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [audio, name, description, busy, cover, livePalette, visual, onCreated]);

  return (
    <div className="grid md:grid-cols-[1fr_340px] gap-8 items-start">
      {/* Left: metadata + file inputs */}
      <div className="space-y-5">
        <div>
          <h3 className="text-base font-semibold mb-1">Import a voice</h3>
          <p className="text-sm text-[color:var(--muted)]">
            Already have an audio clip you want to save as a voice in Reader?
            Drop it here — no synthesis, just store + ship.
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1.5">
            Name <span className="text-[color:var(--muted)] font-normal">(the person)</span>
          </label>
          <input
            className="input"
            placeholder="Alex"
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={60}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1.5">
            Description{" "}
            <span className="text-[color:var(--muted)] font-normal">
              shown alongside the name in Reader
            </span>
          </label>
          <input
            className="input"
            placeholder="Friendly, inviting and balanced"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            maxLength={120}
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1.5">
            Audio file
            <span className="ml-2 text-xs text-[color:var(--muted)] font-normal">
              WAV/MP3/M4A/OGG/FLAC · auto-converted to MP3
            </span>
          </label>
          <DropZone
            accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
            file={audio}
            onFile={setAudio}
            label="Drag & drop an audio file here"
            hint="WAV · MP3 · M4A · OGG · FLAC — up to 20 MB"
            minHeight={120}
            preview={
              audioUrl ? (
                <audio controls className="w-full" src={audioUrl} />
              ) : null
            }
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1.5">
            Visual representation
          </label>
          <div className="seg flex gap-1 rounded-full p-1" style={{ background: "var(--surface-2)" }}>
            <button
              type="button"
              className={`tab flex-1 ${visual === "sphere" ? "tab-active" : ""}`}
              onClick={() => setVisual("sphere")}
            >
              Dynamic sphere
            </button>
            <button
              type="button"
              className={`tab flex-1 ${visual === "picture" ? "tab-active" : ""}`}
              onClick={() => setVisual("picture")}
            >
              Profile picture
            </button>
          </div>

          {visual === "picture" && (
            <div className="mt-3">
              <DropZone
                accept="image/png,image/jpeg,image/webp,image/gif"
                file={cover}
                onFile={setCover}
                label="Drag & drop a profile picture"
                hint="PNG · JPEG · WebP · GIF — up to 5 MB"
                minHeight={100}
                preview={
                  coverUrl ? (
                    /* eslint-disable-next-line @next/next/no-img-element */
                    <img
                      src={coverUrl}
                      alt="cover preview"
                      className="w-16 h-16 rounded-full object-cover border border-[color:var(--border)]"
                    />
                  ) : null
                }
              />
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 pt-2 flex-wrap">
          <button
            className="btn btn-primary"
            onClick={handleSubmit}
            disabled={!canSubmit}
          >
            {busy ? "Saving & shipping…" : "Save & send to Reader"}
          </button>
          {error && (
            <span className="text-sm text-red-500 break-all">{error}</span>
          )}
        </div>

        {result && (
          <div className="text-xs text-[color:var(--muted)]">
            Saved as <strong>{result.name}</strong> —{" "}
            {result.synced ? "synced to Reader." : "local only (Reader not configured)."}
          </div>
        )}
      </div>

      {/* Right: live preview of the visual representation */}
      <div className="flex flex-col items-center">
        {visual === "picture" && coverUrl ? (
          <div
            className="voice-cover"
            style={{ width: 220, height: 220 }}
            aria-label="Cover preview"
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={coverUrl} alt="cover preview" />
          </div>
        ) : (
          <VoiceSphere
            seed={result?.id ?? `import:${name}:${moodId}`}
            size={220}
            colors={livePalette}
          />
        )}

        {visual === "sphere" && (
          <div className="mt-5 w-full max-w-[260px]">
            <MoodPicker
              moodId={moodId}
              onMoodChange={(id) => {
                setMoodId(id);
                setPaletteSeed(Math.random());
              }}
              onRegenerate={() => setPaletteSeed(Math.random())}
              palette={livePalette}
              compact
            />
          </div>
        )}
      </div>
    </div>
  );
}
