"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  cloneVoice,
  DEFAULT_PREVIEW_TEXT,
  listEngines,
  previewClone,
  sampleUrl,
  type Engine,
  type Profile,
} from "@/lib/api";
import { EnginePicker, useEngineChoice } from "@/components/EnginePicker";
import { GenerationProgress } from "@/components/GenerationProgress";
import { MoodPicker } from "@/components/MoodPicker";
import { SessionHistory, type SessionItem } from "@/components/SessionHistory";
import { VoiceSphere } from "@/components/VoiceSphere";
import { MOODS, moodPalette } from "@/lib/moodPalettes";

type Props = {
  onCreated: (p: Profile) => void;
};

/** Min reference-clip duration in seconds. Per Spike D (April 2026),
 *  zero-shot voice identity transfer degrades noticeably on prompts
 *  shorter than ~8 s, but the flow still produces usable output down
 *  to 5 s. We enforce 5 s as the floor so short test clips aren't
 *  rejected; the backend (MIN_PROMPT_DURATION_S) re-validates. */
const MIN_DURATION_S = 5;

/** Probe an audio file's duration client-side via a hidden <audio>
 *  element. Avoids reading the whole file into memory — <audio>
 *  streams it and fires `durationchange`. Returns NaN if the browser
 *  can't decode the container (very rare for wav/mp3). */
function probeDurationFromFile(file: File): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const el = document.createElement("audio");
    el.preload = "metadata";
    el.src = url;
    const cleanup = () => {
      URL.revokeObjectURL(url);
      el.remove();
    };
    el.addEventListener("loadedmetadata", () => {
      const d = el.duration;
      cleanup();
      resolve(Number.isFinite(d) ? d : NaN);
    });
    el.addEventListener("error", () => {
      cleanup();
      reject(new Error("Could not decode audio"));
    });
  });
}

export function CloneTab({ onCreated }: Props) {
  const [name, setName] = useState("");
  const [previewText, setPreviewText] = useState(DEFAULT_PREVIEW_TEXT);
  const [refText, setRefText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [fileDuration, setFileDuration] = useState<number | null>(null);
  const [fileDurationError, setFileDurationError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<"preview" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Profile | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [sessionItems, setSessionItems] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sphere colour mood — same palette system as the Import + Design tabs.
  // Persisted into profile.design.colors on Save so Reader's voice picker
  // reproduces the exact mood the user chose here.
  //
  // NOTE on the seed: we can't seed with `Math.random()` in useState because
  // Clone is the default tab and gets SSR'd; a random initializer produces
  // different values on server vs client and blows up hydration with a
  // "server rendered HTML didn't match the client" warning on <VoiceSphere>'s
  // colors prop. Seed with 0 (deterministic, SSR-safe) and randomize in a
  // post-mount effect so the user still gets a fresh palette per visit.
  const [moodId, setMoodId] = useState<string>(MOODS[0].id);
  const [paletteSeed, setPaletteSeed] = useState<number>(0);
  useEffect(() => {
    setPaletteSeed(Math.random());
  }, []);
  const livePalette = useMemo(
    () => moodPalette(moodId, paletteSeed),
    [moodId, paletteSeed]
  );

  const durationOk = fileDuration != null && fileDuration >= MIN_DURATION_S;
  const durationTooShort = fileDuration != null && fileDuration < MIN_DURATION_S;

  // When a new file is dropped, probe its duration client-side so we
  // can gate Save without a round-trip. The backend re-validates.
  useEffect(() => {
    if (!file) {
      setFileDuration(null);
      setFileDurationError(null);
      return;
    }
    let cancelled = false;
    setFileDuration(null);
    setFileDurationError(null);
    probeDurationFromFile(file)
      .then((d) => {
        if (cancelled) return;
        if (Number.isNaN(d)) {
          setFileDurationError("Could not read audio length");
        } else {
          setFileDuration(d);
        }
      })
      .catch((e) => {
        if (!cancelled) setFileDurationError((e as Error).message);
      });
    return () => {
      cancelled = true;
    };
  }, [file]);

  // Clean up all blob URLs when the tab unmounts. previewUrl aliases one
  // of the session items during its lifetime, so revoking the list here
  // covers both cases.
  useEffect(() => {
    return () => {
      sessionItems.forEach((it) => URL.revokeObjectURL(it.blobUrl));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const [engines, setEngines] = useState<Engine[] | null>(null);
  const [backendDefault, setBackendDefault] = useState<string | null>(null);
  const [engine, setEngine] = useEngineChoice(engines, backendDefault);
  const [language, setLanguage] = useState("en");

  useEffect(() => {
    listEngines()
      .then((d) => {
        setEngines(d.engines);
        setBackendDefault(d.default);
      })
      .catch((e) => setError((e as Error).message));
  }, []);

  // Reset language if the chosen engine doesn't support it.
  useEffect(() => {
    if (!engines || !engine) return;
    const e = engines.find((x) => x.id === engine);
    if (e && !e.languages.includes(language)) {
      setLanguage(e.languages[0] ?? "en");
    }
  }, [engine, engines, language]);

  const onDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  }, []);

  const onPreview = useCallback(async () => {
    if (!file || !engine || busy) return;
    setBusy("preview");
    setError(null);
    try {
      const url = await previewClone({
        file,
        engine,
        language,
        previewText: previewText.trim() || undefined,
        refText: refText.trim() || undefined,
      });
      const label = `${name || file.name.replace(/\.[^.]+$/, "")} · ${engine}`;
      const newItem: SessionItem = {
        id: `sess-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        blobUrl: url,
        colors: null, // clone uses seed-derived palette
        label,
        createdAt: Date.now(),
      };
      setSessionItems((xs) => [newItem, ...xs].slice(0, 12));
      setActiveSessionId(newItem.id);
      setPreviewUrl(url);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }, [file, name, engine, language, previewText, refText, busy]);

  const selectSessionItem = useCallback((item: SessionItem) => {
    setActiveSessionId(item.id);
    setPreviewUrl(item.blobUrl);
  }, []);

  const clearSession = useCallback(() => {
    sessionItems.forEach((it) => URL.revokeObjectURL(it.blobUrl));
    setSessionItems([]);
    setActiveSessionId(null);
    setPreviewUrl(null);
  }, [sessionItems]);

  const onSave = useCallback(async () => {
    if (!file || !name.trim() || !engine || busy) return;
    setBusy("save");
    setError(null);
    setResult(null);
    try {
      const p = await cloneVoice({
        name: name.trim(),
        file,
        engine,
        language,
        previewText: previewText.trim() || undefined,
        refText: refText.trim() || undefined,
        colors: livePalette as unknown as string[],
      });
      setResult(p);
      onCreated(p);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }, [file, name, engine, language, previewText, refText, busy, onCreated, livePalette]);

  return (
    <div className="grid md:grid-cols-[1fr_320px] gap-8 items-start">
      <div className="space-y-5">
        <div>
          <label className="block text-sm font-medium mb-1.5">Voice name</label>
          <input
            className="input"
            placeholder="e.g. Mom, Attenborough, My narrator"
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={60}
          />
        </div>

        {engines && engine && (
          <EnginePicker
            engines={engines}
            value={engine}
            onChange={setEngine}
            includeLanguage
            language={language}
            onLanguageChange={setLanguage}
          />
        )}

        <div>
          <label className="block text-sm font-medium mb-1.5">
            Reference audio
            <span className="ml-2 text-xs text-[color:var(--muted)] font-normal">
              ≥{MIN_DURATION_S}s of clean speech (10–20s is ideal, 5s works for testing)
            </span>
          </label>
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className={`card flex flex-col items-center justify-center text-center cursor-pointer transition-colors ${
              dragging ? "border-[color:var(--accent)]" : ""
            } ${durationTooShort ? "border-red-500/60" : ""}`}
            style={{ minHeight: 160 }}
          >
            <input
              ref={inputRef}
              type="file"
              accept="audio/wav,audio/mpeg,audio/mp3,audio/x-wav,.wav,.mp3"
              className="hidden"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            {file ? (
              <>
                <div className="text-base font-medium">{file.name}</div>
                <div className="text-xs text-[color:var(--muted)] mt-1">
                  {(file.size / 1024).toFixed(0)} KB
                  {fileDuration != null && (
                    <>
                      {" · "}
                      <span className={durationOk ? "text-[color:var(--foreground)]" : "text-red-400"}>
                        {fileDuration.toFixed(1)}s
                      </span>
                    </>
                  )}
                  {" · click to replace"}
                </div>
                {durationTooShort && (
                  <div className="text-xs text-red-400 mt-2 max-w-sm">
                    ⚠ Too short for good zero-shot quality. Voices cloned from
                    clips under {MIN_DURATION_S}s sound muffled / off-mic.
                    Record or pick a longer clip.
                  </div>
                )}
                {fileDurationError && (
                  <div className="text-xs text-red-400 mt-2">
                    Could not read audio length — upload may still work if the
                    backend can decode it.
                  </div>
                )}
              </>
            ) : (
              <>
                <div className="text-4xl mb-2 opacity-60">⇡</div>
                <div className="text-sm font-medium">Drop a WAV or MP3, or click to browse</div>
                <div className="text-xs text-[color:var(--muted)] mt-1">
                  Min {MIN_DURATION_S}s · resampled to 24 kHz mono
                </div>
              </>
            )}
          </div>
        </div>

        <details className="text-sm">
          <summary className="cursor-pointer text-[color:var(--muted)] hover:text-[color:var(--foreground)]">
            Advanced: transcript + preview text
          </summary>
          <div className="space-y-3 pt-3">
            <div>
              <label className="block text-xs font-medium mb-1 text-[color:var(--muted)]">
                What&apos;s being said in the reference clip?{" "}
                <span className="text-[color:var(--muted)]">
                  (leave blank — Whisper auto-transcribes on save)
                </span>
              </label>
              <textarea
                className="input"
                rows={2}
                placeholder="Leave blank to auto-transcribe on save…"
                value={refText}
                onChange={(e) => setRefText(e.target.value)}
              />
            </div>
            <div>
              <label className="block text-xs font-medium mb-1 text-[color:var(--muted)]">
                Preview text (what the new voice will say)
              </label>
              <textarea
                className="input"
                rows={2}
                value={previewText}
                onChange={(e) => setPreviewText(e.target.value)}
              />
            </div>
          </div>
        </details>

        <div className="flex items-center gap-3 pt-2 flex-wrap">
          <button
            className="btn"
            onClick={onPreview}
            disabled={!!busy || !file}
            title="Synthesize without saving — listen before committing"
          >
            {busy === "preview" ? "Generating preview…" : "▶ Preview"}
          </button>
          <button
            className="btn btn-primary"
            onClick={onSave}
            disabled={!!busy || !file || !name.trim() || durationTooShort}
            title={
              durationTooShort
                ? `Reference clip must be at least ${MIN_DURATION_S}s`
                : undefined
            }
          >
            {busy === "save" ? "Saving…" : "Save voice"}
          </button>
          {error && <span className="text-sm text-red-500">{error}</span>}
        </div>

        {busy && <GenerationProgress kind={busy} estimateSeconds={12} />}
      </div>

      <div className="flex flex-col items-center">
        <VoiceSphere
          seed={activeSessionId ?? result?.id ?? name ?? "preview"}
          size={220}
          speaking={audioPlaying}
          ready={(!!previewUrl || !!result) && !audioPlaying}
          withPlayIcon={false}
          colors={livePalette}
        />
        {/* Once saved, the server-side palette is the source of truth and
            changing the mood here would just be decorative — hide the
            picker in that state. Before save, it drives both the live
            sphere colour and the `colors` payload sent to /api/clone. */}
        {!result && (
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
        {result ? (
          <div className="mt-5 w-full space-y-3">
            <div className="text-center">
              <div className="font-medium">{result.name}</div>
              <div className="text-xs text-[color:var(--muted)] mt-0.5">
                Cloned
                {result.duration_s != null && ` · ${result.duration_s.toFixed(1)}s prompt`}
                {" · "}
                {result.synced ? "synced to Reader" : "local only"}
              </div>
            </div>
            <audio
              controls
              className="w-full"
              onPlay={() => setAudioPlaying(true)}
              onPause={() => setAudioPlaying(false)}
              onEnded={() => setAudioPlaying(false)}
              src={sampleUrl(result.id)}
            />
            {result.prompt_text && (
              <details className="text-xs text-[color:var(--muted)]">
                <summary className="cursor-pointer hover:text-[color:var(--foreground)]">
                  Auto-transcribed prompt
                </summary>
                <div className="mt-2 p-2 rounded bg-[color:var(--surface-2)] text-[color:var(--foreground)] leading-relaxed">
                  &ldquo;{result.prompt_text}&rdquo;
                </div>
              </details>
            )}
          </div>
        ) : previewUrl ? (
          <audio
            controls
            autoPlay
            className="w-full mt-4"
            src={previewUrl}
            onPlay={() => setAudioPlaying(true)}
            onPause={() => setAudioPlaying(false)}
            onEnded={() => setAudioPlaying(false)}
          />
        ) : (
          <div className="mt-5 text-center text-xs text-[color:var(--muted)] max-w-xs">
            After cloning, your preview will appear here and the sphere will get
            a unique palette.
          </div>
        )}

        {/* Session history: throwaway previews to A/B between. */}
        <div className="w-full">
          <SessionHistory
            items={sessionItems}
            activeId={activeSessionId}
            onSelect={selectSessionItem}
            onClear={sessionItems.length > 0 ? clearSession : undefined}
          />
        </div>
      </div>
    </div>
  );
}
