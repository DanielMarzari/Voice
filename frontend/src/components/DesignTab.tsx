"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  DEFAULT_PREVIEW_TEXT,
  designVoice,
  listEngines,
  listPresets,
  listReferences,
  listXttsSpeakers,
  previewDesign,
  sampleUrl,
  uploadReference,
  type Engine,
  type Preset,
  type Profile,
  type Reference,
  type XttsSpeaker,
} from "@/lib/api";
import { EnginePicker, useEngineChoice } from "@/components/EnginePicker";
import { GenerationProgress } from "@/components/GenerationProgress";
import { ManageReferencesModal } from "@/components/ManageReferencesModal";
import { MoodPicker } from "@/components/MoodPicker";
import { SessionHistory, type SessionItem } from "@/components/SessionHistory";
import { UploadReferenceModal } from "@/components/UploadReferenceModal";
import { VoiceSphere } from "@/components/VoiceSphere";
import { MOODS, moodPalette, type Palette } from "@/lib/moodPalettes";

type Props = {
  onCreated: (p: Profile) => void;
};

type Sliders = {
  pitch: number;      // -6..6 semitones
  speed: number;      // 0.5..2.0
  temperature: number; // 0.1..1.5
};

const DEFAULT_SLIDERS: Sliders = { pitch: 0, speed: 1, temperature: 0.7 };

export function DesignTab({ onCreated }: Props) {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [baseVoice, setBaseVoice] = useState("");
  const [name, setName] = useState("");
  const [previewText, setPreviewText] = useState(DEFAULT_PREVIEW_TEXT);
  const [sliders, setSliders] = useState<Sliders>(DEFAULT_SLIDERS);
  const [busy, setBusy] = useState<"preview" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Profile | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [audioPlaying, setAudioPlaying] = useState(false);

  // Color mood: dropdown of curated palettes + a seed we bump on
  // "Regenerate" to randomize within the chosen mood.
  const [moodId, setMoodId] = useState<string>(MOODS[0].id);
  const [paletteSeed, setPaletteSeed] = useState<number>(() => Math.random());

  // In-memory list of previews generated during this tab session. Lets the
  // user A/B between unsaved generations without re-synthesizing. Cleared
  // on tab unmount / page reload (intentionally throwaway).
  const [sessionItems, setSessionItems] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Clean up all blob URLs when the tab unmounts. previewUrl aliases one
  // of the sessionItems' urls during its lifetime, so we revoke the
  // session list exhaustively on unmount — covers both.
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

  // XTTS built-in speakers (female / male options) — fetched when XTTS is chosen.
  const [xttsSpeakers, setXttsSpeakers] = useState<XttsSpeaker[]>([]);
  const [speakerName, setSpeakerName] = useState<string | null>(null);

  // User-uploaded reference clips (the "bring your own voice" path).
  const [references, setReferences] = useState<Reference[]>([]);
  const [refModalOpen, setRefModalOpen] = useState(false);
  const [manageOpen, setManageOpen] = useState(false);

  useEffect(() => {
    listPresets()
      .then((ps) => {
        setPresets(ps);
        if (ps.length && !baseVoice) setBaseVoice(ps[0].id);
      })
      .catch((e) => setError((e as Error).message));
    listEngines()
      .then((d) => {
        setEngines(d.engines);
        setBackendDefault(d.default);
      })
      .catch(() => {});
    listReferences().then(setReferences).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleReferenceUpload(args: {
    name: string;
    description: string;
    file: File;
  }) {
    const added = await uploadReference(args);
    setReferences((r) => [...r.filter((x) => x.id !== added.id), added]);
    // Auto-select the newly uploaded reference.
    setBaseVoice(added.id);
  }

  function handleReferencesChanged(next: Reference[]) {
    setReferences(next);
    // If the active base voice just got deleted, revert to the first preset.
    if (
      baseVoice.startsWith("user:") &&
      !next.some((r) => r.id === baseVoice) &&
      presets.length > 0
    ) {
      selectPreset(presets[0].id);
    }
  }

  // Keep language valid for the chosen engine.
  useEffect(() => {
    if (!engines || !engine) return;
    const e = engines.find((x) => x.id === engine);
    if (e && !e.languages.includes(language)) {
      setLanguage(e.languages[0] ?? "en");
    }
  }, [engine, engines, language]);

  // Fetch XTTS built-in speakers lazily when XTTS becomes the chosen engine.
  useEffect(() => {
    if (engine !== "xtts") {
      setSpeakerName(null);
      return;
    }
    if (xttsSpeakers.length > 0) return;
    listXttsSpeakers()
      .then((sp) => {
        setXttsSpeakers(sp);
        // Default to the first female speaker so the baseline isn't "male."
        const firstFemale = sp.find((s) => s.gender === "female");
        setSpeakerName((firstFemale ?? sp[0])?.name ?? null);
      })
      .catch(() => {
        // Non-fatal — fall back to reference-clip cloning.
      });
  }, [engine, xttsSpeakers.length]);

  const update = <K extends keyof Sliders>(key: K, val: Sliders[K]) =>
    setSliders((s) => ({ ...s, [key]: val }));

  const selectPreset = (id: string) => {
    setBaseVoice(id);
    // Only built-in mood presets carry slider defaults. User references
    // (id prefixed with "user:") leave the sliders untouched so the
    // user can keep whatever they were fine-tuning.
    const p = presets.find((x) => x.id === id);
    if (p) {
      setSliders({
        pitch: p.default_pitch,
        speed: p.default_speed,
        temperature: p.default_temperature,
      });
    }
  };

  // Curated color mood. Picking a mood gives a stable "emotional register"
  // (Vibrant / Calm / Sunset / Ocean / etc.); Regenerate reshuffles
  // within that mood so the user can nudge without leaving the vibe.
  // Declared up here so both the sphere prop and onSave() dep list see it.
  const livePalette = useMemo(
    () => moodPalette(moodId, paletteSeed),
    [moodId, paletteSeed]
  );

  const onPreview = useCallback(async () => {
    if (!baseVoice || !engine || busy) return;
    setBusy("preview");
    setError(null);
    try {
      const url = await previewDesign({
        baseVoice,
        engine,
        language,
        pitch: sliders.pitch,
        speed: sliders.speed,
        temperature: sliders.temperature,
        speakerName: engine === "xtts" ? speakerName : null,
        previewText: previewText.trim() || undefined,
      });
      // Build a short, readable caption for the session list.
      const baseLabel = baseVoice.startsWith("user:")
        ? baseVoice.slice(5)
        : baseVoice;
      const voiceLabel =
        engine === "xtts" && speakerName ? speakerName.split(" ")[0] : baseLabel;
      const newItem: SessionItem = {
        id: `sess-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
        blobUrl: url,
        colors: livePalette as Palette,
        label: `${voiceLabel} · p${sliders.pitch.toFixed(1)} s${sliders.speed.toFixed(2)}`,
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
  }, [baseVoice, engine, language, sliders, previewText, speakerName, livePalette, busy]);

  const selectSessionItem = useCallback((item: SessionItem) => {
    setActiveSessionId(item.id);
    setPreviewUrl(item.blobUrl);
    // Update the palette so the sphere matches that preview's look.
    // (We don't restore the moodId — these are transient comparisons.)
  }, []);

  const clearSession = useCallback(() => {
    sessionItems.forEach((it) => URL.revokeObjectURL(it.blobUrl));
    setSessionItems([]);
    setActiveSessionId(null);
    setPreviewUrl(null);
  }, [sessionItems]);

  const onSave = useCallback(async () => {
    if (!name.trim() || !baseVoice || !engine || busy) return;
    setBusy("save");
    setError(null);
    setResult(null);
    try {
      const p = await designVoice({
        name: name.trim(),
        baseVoice,
        engine,
        language,
        pitch: sliders.pitch,
        speed: sliders.speed,
        temperature: sliders.temperature,
        speakerName: engine === "xtts" ? speakerName : null,
        // Ship the exact slider-derived palette so Reader's sphere matches.
        colors: livePalette as unknown as string[],
        previewText: previewText.trim() || undefined,
      });
      setResult(p);
      onCreated(p);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
    }
  }, [name, baseVoice, engine, language, sliders, previewText, speakerName, livePalette, busy, onCreated]);

  // Seed for preview sphere — static so the sphere identity doesn't jump
  // every time a slider moves. The colors (livePalette above) are what change.
  const previewSeed = `${engine}:${baseVoice}:${speakerName ?? ""}`;

  return (
    <div className="grid md:grid-cols-[1fr_320px] gap-8 items-start">
      <div className="space-y-5">
        <div>
          <label className="block text-sm font-medium mb-1.5">Voice name</label>
          <input
            className="input"
            placeholder="e.g. Narrator Soft, Calm Reader, Warm Alto"
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

        {/* XTTS built-in speakers — 58 pre-trained voices, explicitly
            gendered, no reference clip needed. Only shown when XTTS is
            the selected engine. When XTTS isn't installed the list is
            empty and we show a hint pointing to the Upload alternative. */}
        {engine === "xtts" && xttsSpeakers.length > 0 && (
          <div>
            <label className="block text-sm font-medium mb-1.5">
              Speaker voice
              <span className="ml-2 text-xs text-[color:var(--muted)] font-normal">
                XTTS built-in — pick gender / tone
              </span>
            </label>
            <select
              className="select"
              value={speakerName ?? ""}
              onChange={(e) => setSpeakerName(e.target.value || null)}
            >
              <optgroup label="Female">
                {xttsSpeakers
                  .filter((s) => s.gender === "female")
                  .map((s) => (
                    <option key={s.name} value={s.name}>{s.name}</option>
                  ))}
              </optgroup>
              <optgroup label="Male">
                {xttsSpeakers
                  .filter((s) => s.gender === "male")
                  .map((s) => (
                    <option key={s.name} value={s.name}>{s.name}</option>
                  ))}
              </optgroup>
              {xttsSpeakers.some((s) => s.gender === "unknown") && (
                <optgroup label="Other">
                  {xttsSpeakers
                    .filter((s) => s.gender === "unknown")
                    .map((s) => (
                      <option key={s.name} value={s.name}>{s.name}</option>
                    ))}
                </optgroup>
              )}
            </select>
            <div className="text-xs text-[color:var(--muted)] mt-1">
              When a speaker is selected, XTTS uses it directly and ignores
              the Base voice reference clip below.
            </div>
          </div>
        )}
        {engine === "xtts" && xttsSpeakers.length === 0 && (
          <div className="card !p-3 text-xs text-[color:var(--muted)]">
            XTTS built-in speaker list unavailable
            {engines?.find((e) => e.id === "xtts" && !e.available)
              ? ": the TTS package isn't installed. Run `pip install TTS>=0.22` in backend/ and restart, or use the Upload button below to bring your own reference clip."
              : ". Wait for the first synthesis to load XTTS, then refresh. In the meantime, Upload a reference clip below to get any voice you want."}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium mb-1.5">
            Base voice
            <span className="ml-2 text-xs text-[color:var(--muted)] font-normal">
              built-in mood OR your own upload
            </span>
          </label>
          <div className="flex gap-2 items-stretch">
            <select
              className="select flex-1"
              value={baseVoice}
              onChange={(e) => selectPreset(e.target.value)}
              disabled={!presets.length}
            >
              <optgroup label="Built-in moods">
                {presets.length === 0 && <option>Loading presets…</option>}
                {presets.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </optgroup>
              {references.length > 0 && (
                <optgroup label="Your uploaded voices">
                  {references.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label}
                    </option>
                  ))}
                </optgroup>
              )}
            </select>
            <button
              type="button"
              className="btn"
              onClick={() => setRefModalOpen(true)}
              title="Upload a reference audio clip — any voice, any gender"
            >
              ⇡ Upload
            </button>
            {references.length > 0 && (
              <button
                type="button"
                className="btn"
                onClick={() => setManageOpen(true)}
                title="Manage your uploaded reference voices"
              >
                Manage
              </button>
            )}
          </div>
          <div className="text-xs text-[color:var(--muted)] mt-1">
            Drop a ~10s clean speech clip. Works with both F5-TTS and XTTS.
          </div>
        </div>

        <Slider
          label="Pitch"
          unit="semitones"
          min={-6}
          max={6}
          step={0.5}
          value={sliders.pitch}
          onChange={(v) => update("pitch", v)}
        />
        <Slider
          label="Speed"
          unit="x"
          min={0.5}
          max={2}
          step={0.05}
          value={sliders.speed}
          onChange={(v) => update("speed", v)}
        />
        <Slider
          label="Temperature"
          unit=""
          min={0.1}
          max={1.5}
          step={0.05}
          value={sliders.temperature}
          onChange={(v) => update("temperature", v)}
          hint="Higher = more expressive, more variance"
        />

        <div>
          <label className="block text-sm font-medium mb-1.5">Preview text</label>
          <textarea
            className="input"
            rows={2}
            value={previewText}
            onChange={(e) => setPreviewText(e.target.value)}
          />
        </div>

        <div className="flex items-center gap-3 pt-2 flex-wrap">
          <button
            className="btn"
            onClick={onPreview}
            disabled={!!busy || !baseVoice}
            title="Synthesize without saving — listen before committing"
          >
            {busy === "preview" ? "Generating preview…" : "▶ Preview"}
          </button>
          <button
            className="btn btn-primary"
            onClick={onSave}
            disabled={!!busy || !name.trim() || !baseVoice}
          >
            {busy === "save" ? "Saving…" : "Save voice"}
          </button>
          <button
            className="btn btn-ghost"
            onClick={() => setSliders(DEFAULT_SLIDERS)}
            disabled={!!busy}
          >
            Reset sliders
          </button>
          {error && <span className="text-sm text-red-500">{error}</span>}
        </div>

        {busy && <GenerationProgress kind={busy} estimateSeconds={12} />}
      </div>

      {/* Right column: sphere + mood picker + session history */}
      <div className="flex flex-col items-center">
        <VoiceSphere
          seed={activeSessionId ?? result?.id ?? previewSeed}
          size={220}
          speaking={audioPlaying}
          ready={(!!previewUrl || !!result) && !audioPlaying}
          colors={
            (sessionItems.find((s) => s.id === activeSessionId)?.colors as
              | Palette
              | undefined) ?? livePalette
          }
        />

        {result ? (
          <div className="mt-5 w-full space-y-3">
            <div className="text-center">
              <div className="font-medium">{result.name}</div>
              <div className="text-xs text-[color:var(--muted)] mt-0.5">
                Designed · {result.synced ? "synced to Reader" : "local only"}
              </div>
            </div>
            <audio
              controls
              className="w-full"
              src={sampleUrl(result.id)}
              onPlay={() => setAudioPlaying(true)}
              onPause={() => setAudioPlaying(false)}
              onEnded={() => setAudioPlaying(false)}
            />
          </div>
        ) : previewUrl ? (
          // Active session preview — hidden controls auto-play, user can
          // still drag the scrubber. Distinct from the saved audio above.
          <audio
            controls
            autoPlay
            className="w-full mt-4"
            src={previewUrl}
            onPlay={() => setAudioPlaying(true)}
            onPause={() => setAudioPlaying(false)}
            onEnded={() => setAudioPlaying(false)}
          />
        ) : null}

        {/* Color mood moved here per user request — directly under the orb. */}
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

      <UploadReferenceModal
        open={refModalOpen}
        onClose={() => setRefModalOpen(false)}
        onSubmit={handleReferenceUpload}
      />
      <ManageReferencesModal
        open={manageOpen}
        onClose={() => setManageOpen(false)}
        references={references}
        onChange={handleReferencesChanged}
      />
    </div>
  );
}

function Slider(props: {
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <div>
      <div className="flex items-baseline justify-between mb-1.5">
        <label className="block text-sm font-medium">{props.label}</label>
        <span className="text-xs text-[color:var(--muted)] tabular-nums">
          {props.value.toFixed(props.step < 0.1 ? 2 : 1)}
          {props.unit ? ` ${props.unit}` : ""}
        </span>
      </div>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
        className="w-full accent-[color:var(--accent)]"
      />
      {props.hint && (
        <div className="text-xs text-[color:var(--muted)] mt-1">{props.hint}</div>
      )}
    </div>
  );
}
