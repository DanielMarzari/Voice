"use client";

/**
 * DeepCloneTab — ElevenLabs-equivalent Professional Voice Cloning.
 *
 * Zero-shot (the Clone tab) gets you 80% of the way there from 10-20 s
 * of audio. This tab gets you the remaining 20%: fine-tune a model on
 * ~30 min of clean speech so the resulting voice is stable over long
 * reads.
 *
 * Two-stage pipeline:
 *
 *   Stage 1 — Prepare (this UI, synchronous, <few min):
 *     User drops one or more audio files. We ship them to
 *     /api/deep-clone/prepare which segments the audio into 3-10 s
 *     chunks, transcribes each with whisper, and writes the dataset
 *     to backend/data/training/<id>/.
 *
 *   Stage 2 — Train (out-of-process, hours):
 *     Backend renders a train.sh the user runs in their terminal.
 *     This UI shows the exact command + a "Copy command" button.
 *     We poll the training status every 30 s; once a checkpoint
 *     appears on disk, the job flips to "Ready" and a "Use in Design
 *     tab" button appears.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  deleteDeepCloneTraining,
  getDeepCloneScript,
  listDeepCloneTrainings,
  listEngines,
  prepareDeepClone,
  registerDeepClone,
  type DeepCloneManifest,
  type Engine,
} from "@/lib/api";
import { DropZone } from "@/components/DropZone";
import { EnginePicker, useEngineChoice } from "@/components/EnginePicker";

type LocalFile = {
  file: File;
  durationS: number | null; // populated lazily via an <audio> probe
};

type Props = {
  onRegistered?: (deepId: string) => void;
};

export function DeepCloneTab({ onRegistered }: Props) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [files, setFiles] = useState<LocalFile[]>([]);
  const [pending, setPending] = useState<File | null>(null); // DropZone's "current" file

  const [engines, setEngines] = useState<Engine[] | null>(null);
  const [backendDefault, setBackendDefault] = useState<string | null>(null);
  const [engine, setEngine] = useEngineChoice(engines, backendDefault);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<DeepCloneManifest | null>(null);
  const [scriptCache, setScriptCache] = useState<Record<string, string>>({});
  const [trainings, setTrainings] = useState<DeepCloneManifest[]>([]);

  useEffect(() => {
    listEngines()
      .then((d) => {
        setEngines(d.engines);
        setBackendDefault(d.default);
      })
      .catch(() => {});
    refreshTrainings();
  }, []);

  const refreshTrainings = useCallback(() => {
    listDeepCloneTrainings()
      .then(setTrainings)
      .catch(() => {});
  }, []);

  // Poll training status every 30 s whenever any job is in "training".
  // Stops once all jobs are in a terminal state (prepared / ready / failed).
  useEffect(() => {
    const anyTraining = trainings.some((t) => t.status === "training");
    if (!anyTraining) return;
    const id = setInterval(refreshTrainings, 30_000);
    return () => clearInterval(id);
  }, [trainings, refreshTrainings]);

  // DropZone is single-file by design — we catch its onFile calls and
  // append to our list so the user can build up a multi-file dataset.
  useEffect(() => {
    if (!pending) return;
    const f = pending;
    setPending(null);
    // Probe duration in the background.
    const placeholder: LocalFile = { file: f, durationS: null };
    setFiles((xs) => [...xs, placeholder]);
    probeDuration(f).then((d) => {
      setFiles((xs) =>
        xs.map((x) => (x.file === f ? { ...x, durationS: d } : x))
      );
    });
  }, [pending]);

  const totalDuration = useMemo(
    () =>
      files.reduce((acc, f) => (f.durationS != null ? acc + f.durationS : acc), 0),
    [files]
  );

  const unknownDurationCount = files.filter((f) => f.durationS == null).length;

  const canSubmit = files.length > 0 && !!name.trim() && !busy && !!engine;

  const handlePrepare = useCallback(async () => {
    if (!canSubmit) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const manifest = await prepareDeepClone({
        name: name.trim(),
        description: description.trim() || undefined,
        engine: (engine as "f5" | "xtts") ?? "f5",
        files: files.map((f) => f.file),
      });
      setResult(manifest);
      refreshTrainings();
      // Prefetch script body for the "copy command" button.
      try {
        const text = await getDeepCloneScript(manifest.id);
        setScriptCache((c) => ({ ...c, [manifest.id]: text }));
      } catch {
        /* non-fatal */
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }, [canSubmit, name, description, engine, files, refreshTrainings]);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-base font-semibold mb-1">Deep Clone</h3>
        <p className="text-sm text-[color:var(--muted)]">
          Fine-tune a model on <strong>~30 min of clean speech</strong> for
          ElevenLabs-grade consistency. The zero-shot{" "}
          <em>Clone a voice</em> tab is faster but less stable over long
          reads — use Deep Clone when you want a voice that stays tight
          across a 30 min audiobook chapter.
        </p>
        <div className="text-xs text-[color:var(--muted)] mt-2">
          Training runs out-of-process (a shell script you run yourself)
          — takes hours on CPU / MPS. The UI prepares the dataset here,
          then polls for the checkpoint.
        </div>
      </div>

      <div className="grid md:grid-cols-[1fr_340px] gap-8 items-start">
        {/* Left: inputs */}
        <div className="space-y-5">
          <div>
            <label className="block text-sm font-medium mb-1.5">
              Voice name{" "}
              <span className="text-[color:var(--muted)] font-normal">
                (the person)
              </span>
            </label>
            <input
              className="input"
              placeholder="Morgan Freeman, Grandma, My narrator"
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
              placeholder="Warm, authoritative, slow-paced"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              maxLength={120}
            />
          </div>

          {engines && engine && (
            <EnginePicker
              engines={engines.filter((e) => e.id === "f5" || e.id === "xtts")}
              value={engine}
              onChange={setEngine}
            />
          )}

          <div>
            <label className="block text-sm font-medium mb-1.5">
              Training audio{" "}
              <span className="text-xs text-[color:var(--muted)] font-normal">
                aim for ~30 min total of clean, consistent speech
              </span>
            </label>
            <DropZone
              accept="audio/*,.wav,.mp3,.m4a,.ogg,.flac"
              file={pending}
              onFile={setPending}
              label="Drag & drop audio file(s) here"
              hint="WAV · MP3 · M4A · OGG · FLAC — add files one at a time, build up to ~30 min"
              minHeight={140}
            />
            {files.length > 0 && (
              <div className="mt-3 border border-[color:var(--border)] rounded-md divide-y divide-[color:var(--border)]">
                {files.map((f, i) => (
                  <div
                    key={`${f.file.name}-${i}`}
                    className="flex items-center justify-between px-3 py-2 text-xs"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="truncate font-medium" title={f.file.name}>
                        {f.file.name}
                      </div>
                      <div className="text-[color:var(--muted)]">
                        {formatSize(f.file.size)}
                        {f.durationS != null
                          ? ` · ${formatDuration(f.durationS)}`
                          : " · measuring…"}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn !py-1 !px-2 !text-red-500"
                      onClick={() =>
                        setFiles((xs) => xs.filter((x) => x !== f))
                      }
                      aria-label={`Remove ${f.file.name}`}
                    >
                      ✕
                    </button>
                  </div>
                ))}
                <div className="px-3 py-2 text-xs text-[color:var(--muted)] flex justify-between items-center bg-[color:var(--surface-2)]">
                  <span>
                    {files.length} file{files.length === 1 ? "" : "s"}
                    {unknownDurationCount > 0
                      ? ` · ${unknownDurationCount} still measuring`
                      : ""}
                  </span>
                  <span>
                    Total:{" "}
                    <strong>{formatDuration(totalDuration)}</strong>
                    {totalDuration < 60 * 15 && files.length > 0 && (
                      <span className="text-yellow-500 ml-2">
                        (aim for ≥15 min)
                      </span>
                    )}
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="flex items-center gap-3 pt-2 flex-wrap">
            <button
              className="btn btn-primary"
              onClick={handlePrepare}
              disabled={!canSubmit}
              title="Segment + transcribe the uploaded audio into a training dataset"
            >
              {busy ? "Preparing dataset…" : "Prepare dataset"}
            </button>
            {error && (
              <span className="text-sm text-red-500 break-all">{error}</span>
            )}
          </div>
          {busy && (
            <div className="text-xs text-[color:var(--muted)] flex items-center gap-2">
              <Spinner />
              Segmenting + transcribing… can take 1-3 min for ~30 min of
              audio on the first call (whisper loads on first use).
            </div>
          )}

          {result && (
            <DeepCloneResult
              manifest={result}
              scriptBody={scriptCache[result.id]}
            />
          )}
        </div>

        {/* Right: explainer panel */}
        <aside className="card text-xs space-y-3 text-[color:var(--muted)]">
          <div className="font-medium text-[color:var(--foreground)]">
            How Deep Clone works
          </div>
          <ol className="space-y-2 list-decimal list-inside">
            <li>
              <strong className="text-[color:var(--foreground)]">
                Prepare
              </strong>{" "}
              (right now): we split your audio into 3–10 s segments,
              transcribe each with Whisper, and write a training
              manifest.
            </li>
            <li>
              <strong className="text-[color:var(--foreground)]">
                Train
              </strong>{" "}
              (in your terminal): copy the generated{" "}
              <code>train.sh</code> command. Takes hours on Apple
              Silicon — days on small machines.
            </li>
            <li>
              <strong className="text-[color:var(--foreground)]">
                Register
              </strong>
              : once the checkpoint lands on disk, this UI auto-detects
              it. Click <em>Use in Design tab</em> to make the voice
              selectable as <code>deep:&lt;slug&gt;</code>.
            </li>
          </ol>
          <div className="pt-2 border-t border-[color:var(--border)]">
            <div className="font-medium text-[color:var(--foreground)]">
              Tips
            </div>
            <ul className="mt-1.5 space-y-1 list-disc list-inside">
              <li>
                Ship one style at a time (no mixing audiobook + podcast
                reads).
              </li>
              <li>Denoise first — garbage in, garbage out.</li>
              <li>
                Review <code>metadata.csv</code> before training if the
                transcripts look off.
              </li>
            </ul>
          </div>
        </aside>
      </div>

      {/* Bottom: existing training jobs */}
      <section className="mt-10">
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-sm font-semibold">My deep clones</h4>
          <button
            className="btn !py-1 !px-2 text-xs"
            onClick={refreshTrainings}
            title="Re-fetch training status"
          >
            Refresh
          </button>
        </div>
        {trainings.length === 0 ? (
          <div className="text-xs text-[color:var(--muted)]">
            No deep clones yet. Prepare one above.
          </div>
        ) : (
          <div className="space-y-3">
            {trainings.map((t) => (
              <TrainingRow
                key={t.id}
                manifest={t}
                onDelete={async () => {
                  if (
                    !confirm(
                      `Delete deep clone "${t.name}"? This removes the dataset and any checkpoint.`
                    )
                  )
                    return;
                  await deleteDeepCloneTraining(t.id);
                  refreshTrainings();
                }}
                onRegister={async () => {
                  try {
                    const reg = await registerDeepClone(t.id);
                    alert(
                      `Registered as ${reg.id}. It's now selectable in the Design tab's base voice dropdown.`
                    );
                    onRegistered?.(reg.id);
                    refreshTrainings();
                  } catch (e) {
                    alert(`Register failed: ${(e as Error).message}`);
                  }
                }}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sub-components                                                      */
/* ------------------------------------------------------------------ */

function DeepCloneResult({
  manifest,
  scriptBody,
}: {
  manifest: DeepCloneManifest;
  scriptBody?: string;
}) {
  const [copied, setCopied] = useState(false);

  const command = `bash ${manifest.train_script}`;
  async function copy() {
    try {
      await navigator.clipboard.writeText(command);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* no-op */
    }
  }

  return (
    <div className="card space-y-3 border-[color:var(--accent)]">
      <div className="text-sm font-semibold">Dataset ready</div>
      <div className="text-xs text-[color:var(--muted)] space-y-0.5">
        <div>
          <strong>{manifest.segment_count}</strong> segments · total{" "}
          <strong>{formatDuration(manifest.total_duration_s)}</strong> ·
          engine <strong>{manifest.engine}</strong>
        </div>
        <div>
          Dataset dir:{" "}
          <code className="text-[color:var(--foreground)]">
            {manifest.train_script.replace(/\/train\.sh$/, "")}
          </code>
        </div>
      </div>

      <div className="text-xs">
        <div className="font-medium mb-1">Run training</div>
        <div className="flex gap-2 items-center">
          <code className="flex-1 px-2 py-1.5 rounded bg-[color:var(--surface-2)] font-mono text-[11px] truncate">
            {command}
          </code>
          <button className="btn !py-1 !px-2 text-xs" onClick={copy}>
            {copied ? "Copied!" : "Copy command"}
          </button>
        </div>
        <div className="text-[color:var(--muted)] mt-2">
          Training runs in your terminal and writes checkpoints back to
          the dataset dir. This UI polls every 30 s; when a checkpoint
          appears, the job below flips to <em>Ready</em>.
        </div>
      </div>

      {scriptBody && (
        <details className="text-xs">
          <summary className="cursor-pointer text-[color:var(--muted)]">
            View <code>train.sh</code>
          </summary>
          <pre className="mt-2 p-2 bg-[color:var(--surface-2)] rounded text-[11px] overflow-x-auto max-h-64">
            {scriptBody}
          </pre>
        </details>
      )}
    </div>
  );
}

function TrainingRow({
  manifest,
  onDelete,
  onRegister,
}: {
  manifest: DeepCloneManifest;
  onDelete: () => void;
  onRegister: () => void;
}) {
  const statusColor =
    manifest.status === "ready"
      ? "text-green-500"
      : manifest.status === "training"
      ? "text-yellow-500"
      : manifest.status === "failed"
      ? "text-red-500"
      : "text-[color:var(--muted)]";

  return (
    <div className="card flex items-start gap-3">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="font-medium text-sm truncate">{manifest.name}</span>
          <span className={`text-xs uppercase tracking-wider ${statusColor}`}>
            ● {manifest.status}
          </span>
        </div>
        {manifest.description && (
          <div className="text-xs text-[color:var(--muted)] mt-0.5 truncate">
            {manifest.description}
          </div>
        )}
        <div className="text-xs text-[color:var(--muted)] mt-1">
          {manifest.segment_count} segments ·{" "}
          {formatDuration(manifest.total_duration_s)} · engine{" "}
          {manifest.engine} ·{" "}
          {new Date(manifest.created_at).toLocaleString()}
        </div>
        {manifest.checkpoint_path && (
          <div className="text-xs text-[color:var(--muted)] mt-0.5 truncate">
            Checkpoint: <code>{manifest.checkpoint_path}</code>
          </div>
        )}
        {manifest.error && (
          <div className="text-xs text-red-500 mt-1">{manifest.error}</div>
        )}
      </div>
      <div className="flex flex-col gap-1.5 items-end shrink-0">
        {manifest.status === "ready" && (
          <button
            className="btn btn-primary !py-1 !px-2 text-xs"
            onClick={onRegister}
          >
            Use in Design tab
          </button>
        )}
        <button
          className="btn !py-1 !px-2 text-xs !text-red-500"
          onClick={onDelete}
        >
          Delete
        </button>
      </div>
    </div>
  );
}

function Spinner() {
  return (
    <span
      className="inline-block w-3 h-3 border-2 border-[color:var(--muted)] border-t-transparent rounded-full animate-spin"
      aria-hidden
    />
  );
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDuration(seconds: number): string {
  if (!isFinite(seconds) || seconds < 0) return "0 s";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  if (m < 60) return `${m}m ${s.toString().padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  const mm = m - h * 60;
  return `${h}h ${mm.toString().padStart(2, "0")}m`;
}

/**
 * Probe an audio file's duration using an offscreen <audio> element.
 * Falls back to null if the browser can't decode it (e.g. uncommon
 * codec) — the backend will still prep the dataset fine.
 */
function probeDuration(file: File): Promise<number | null> {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const audio = new Audio();
    audio.preload = "metadata";
    audio.src = url;
    const cleanup = () => URL.revokeObjectURL(url);
    audio.onloadedmetadata = () => {
      const d = audio.duration;
      cleanup();
      resolve(isFinite(d) ? d : null);
    };
    audio.onerror = () => {
      cleanup();
      resolve(null);
    };
    // Timebomb in case the browser never fires metadata events.
    setTimeout(() => {
      cleanup();
      resolve(null);
    }, 10_000);
  });
}
