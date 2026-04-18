"use client";

import { useCallback, useEffect, useRef, useState } from "react";
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
import { SessionHistory, type SessionItem } from "@/components/SessionHistory";
import { VoiceSphere } from "@/components/VoiceSphere";

type Props = {
  onCreated: (p: Profile) => void;
};

export function CloneTab({ onCreated }: Props) {
  const [name, setName] = useState("");
  const [previewText, setPreviewText] = useState(DEFAULT_PREVIEW_TEXT);
  const [refText, setRefText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState<"preview" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Profile | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [sessionItems, setSessionItems] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

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
      });
      setResult(p);
      onCreated(p);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }, [file, name, engine, language, previewText, refText, busy, onCreated]);

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
              10–20s of clean speech works best
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
            }`}
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
                  {(file.size / 1024).toFixed(0)} KB · click to replace
                </div>
              </>
            ) : (
              <>
                <div className="text-4xl mb-2 opacity-60">⇡</div>
                <div className="text-sm font-medium">Drop a WAV or MP3, or click to browse</div>
                <div className="text-xs text-[color:var(--muted)] mt-1">
                  Will be resampled to 24 kHz mono
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
                What&apos;s being said in the reference clip? (optional — speeds up
                cloning and improves prosody)
              </label>
              <textarea
                className="input"
                rows={2}
                placeholder="Leave blank to auto-transcribe…"
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
            disabled={!!busy || !file || !name.trim()}
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
        />
        {result ? (
          <div className="mt-5 w-full space-y-3">
            <div className="text-center">
              <div className="font-medium">{result.name}</div>
              <div className="text-xs text-[color:var(--muted)] mt-0.5">
                Cloned · {result.synced ? "synced to Reader" : "local only"}
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
