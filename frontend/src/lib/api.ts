// Thin typed wrapper over the FastAPI backend.
//
// We fetch the backend DIRECTLY (http://127.0.0.1:8000) rather than going
// through Next.js's rewrites proxy. Reason: Next's internal HTTP proxy has
// an undocumented ~30s idle timeout, but F5-TTS / XTTS synthesis regularly
// takes 30-60s (longer on first call when weights are loading). The proxy
// would reset the socket mid-synth, causing a 500 on the browser even though
// the backend is still working. Going direct avoids the middleman entirely.
//
// CORS for localhost:3007 is already set up in backend/main.py.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  (typeof window !== "undefined" && window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "");

export function u(path: string): string {
  return `${API_BASE}${path}`;
}

export const DEFAULT_PREVIEW_TEXT =
  "Hi there! I'm your new voice. Do I sound natural, clear, and " +
  "expressive? There's plenty more where this came from.";

export type VoiceKind = "cloned" | "designed" | "uploaded";

export type Profile = {
  id: string;
  name: string;
  kind: VoiceKind;
  engine: string;
  created_at: string;
  design: {
    base_voice?: string;
    pitch?: number;
    speed?: number;
    temperature?: number;
    [k: string]: unknown;
  };
  synced: boolean;
  /** Reference-clip transcript (Whisper-generated at clone time, or user-
   *  supplied). Required for zero-shot inference in Reader. */
  prompt_text?: string | null;
  /** Reference-clip duration in seconds. The Clone tab enforces >= 10 s
   *  client-side; the backend re-validates. */
  duration_s?: number | null;
  /** Number of mel frames in the prompt spectrogram (prompt_mel.f32 on
   *  disk; ~num_frames = duration_s × 24000 / 256). Set by the clone
   *  endpoint via mel_features.compute_prompt_mel. Presence on a synced
   *  profile means Reader's browser-inference path can use this voice. */
  prompt_mel_frames?: number | null;
};

export type Preset = {
  id: string;
  label: string;
  default_pitch: number;
  default_speed: number;
  default_temperature: number;
};

export type Engine = {
  id: string;
  label: string;
  license: string;
  languages: string[];
  loaded: boolean;
  available: boolean;
  device: string;
};

export type Health = {
  ok: boolean;
  reader_configured: boolean;
  reader_base_url: string | null;
  reader_token_set: boolean;
  device: string;
  default_engine: string;
  engines: Engine[];
};

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(`${r.status} ${r.statusText}${body ? `: ${body.slice(0, 300)}` : ""}`);
  }
  return r.json();
}

export async function getHealth(): Promise<Health> {
  return j(await fetch(u("/api/health")));
}

export async function listPresets(): Promise<Preset[]> {
  const d = await j<{ presets: Preset[] }>(await fetch(u("/api/presets")));
  return d.presets;
}

export async function listEngines(): Promise<{ default: string; engines: Engine[] }> {
  return j(await fetch(u("/api/engines")));
}

export type XttsSpeaker = { name: string; gender: "female" | "male" | "unknown" };
export async function listXttsSpeakers(): Promise<XttsSpeaker[]> {
  const d = await j<{ speakers: XttsSpeaker[] }>(await fetch(u("/api/xtts/speakers")));
  return d.speakers;
}

// User-uploaded reference clips (Design tab "Upload reference audio").
// `label` is the composite display string — "Name (description)" when a
// description is present, otherwise just the name.
export type Reference = {
  id: string;
  label: string;
  name: string;
  description: string;
  path: string;
};

export async function listReferences(): Promise<Reference[]> {
  const d = await j<{ references: Reference[] }>(await fetch(u("/api/references")));
  return d.references;
}

export async function uploadReference(args: {
  name: string;
  description?: string;
  file: File;
  transcript?: string;
}): Promise<Reference> {
  const fd = new FormData();
  fd.append("name", args.name);
  fd.append("audio_file", args.file);
  if (args.description) fd.append("description", args.description);
  if (args.transcript) fd.append("transcript", args.transcript);
  const d = await j<{ reference: Reference }>(
    await fetch(u("/api/references"), { method: "POST", body: fd })
  );
  return d.reference;
}

export async function deleteReference(slug: string): Promise<void> {
  await j(
    await fetch(u(`/api/references/${encodeURIComponent(slug)}`), {
      method: "DELETE",
    })
  );
}

export async function listProfiles(): Promise<Profile[]> {
  const d = await j<{ profiles: Profile[] }>(await fetch(u("/api/profiles")));
  return d.profiles;
}

export async function deleteProfile(id: string): Promise<void> {
  await j(await fetch(u(`/api/profiles/${id}`), { method: "DELETE" }));
}

export async function syncProfile(id: string): Promise<Profile> {
  return j(await fetch(u(`/api/profiles/${id}/sync`), { method: "POST" }));
}

export function sampleUrl(id: string): string {
  return u(`/api/profiles/${id}/sample`);
}

function cloneFormData(args: {
  name?: string;
  file: File;
  engine: string;
  language?: string;
  previewText?: string;
  refText?: string;
}): FormData {
  const fd = new FormData();
  if (args.name) fd.append("name", args.name);
  fd.append("audio_file", args.file);
  fd.append("engine_id", args.engine);
  fd.append("language", args.language ?? "en");
  if (args.previewText) fd.append("preview_text", args.previewText);
  if (args.refText) fd.append("ref_text", args.refText);
  return fd;
}

export async function cloneVoice(args: {
  name: string;
  file: File;
  engine: string;
  language?: string;
  previewText?: string;
  refText?: string;
  upload?: boolean;
}): Promise<Profile> {
  const fd = cloneFormData(args);
  fd.append("upload", String(args.upload !== false));
  fd.append("save", "true");
  return j(await fetch(u("/api/clone"), { method: "POST", body: fd }));
}

/** Preview a cloned voice without persisting. Returns an object URL for
 * an <audio> element — revoke it (URL.revokeObjectURL) when done. */
export async function previewClone(args: {
  file: File;
  engine: string;
  language?: string;
  previewText?: string;
  refText?: string;
}): Promise<string> {
  const fd = cloneFormData(args);
  fd.append("save", "false");
  const r = await fetch(u("/api/clone"), { method: "POST", body: fd });
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(
      `${r.status} ${r.statusText}${body ? `: ${body.slice(0, 300)}` : ""}`
    );
  }
  const blob = await r.blob();
  return URL.createObjectURL(blob);
}

function designBody(args: {
  name?: string;
  baseVoice: string;
  engine: string;
  language?: string;
  pitch: number;
  speed: number;
  temperature: number;
  speakerName?: string | null;
  colors?: string[] | null;
  previewText?: string;
}): string {
  return JSON.stringify({
    // The /preview endpoint ignores `name` but we still send something so
    // server-side pydantic doesn't reject it (min_length=1).
    name: args.name || "Preview",
    base_voice: args.baseVoice,
    engine: args.engine,
    language: args.language ?? "en",
    pitch: args.pitch,
    speed: args.speed,
    temperature: args.temperature,
    speaker_name: args.speakerName ?? null,
    colors: args.colors ?? null,
    preview_text: args.previewText ?? DEFAULT_PREVIEW_TEXT,
  });
}

export async function designVoice(args: {
  name: string;
  baseVoice: string;
  engine: string;
  language?: string;
  pitch: number;
  speed: number;
  temperature: number;
  speakerName?: string | null;
  colors?: string[] | null;
  previewText?: string;
  upload?: boolean;
}): Promise<Profile> {
  const url = u(`/api/design?upload=${args.upload !== false}`);
  return j(
    await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: designBody(args),
    })
  );
}

/** Import — store a user-provided audio clip as a voice profile directly
 * (no synthesis). Can optionally ship a cover image OR a mood palette for
 * the sphere. Both are persisted locally AND pushed to Reader when
 * connected. */
export async function importVoice(args: {
  name: string;
  description?: string;
  audio: File;
  cover?: File | null;
  colors?: string[] | null;
}): Promise<Profile> {
  const fd = new FormData();
  fd.append("name", args.name);
  fd.append("audio_file", args.audio);
  if (args.description) fd.append("description", args.description);
  if (args.cover) fd.append("cover_image", args.cover);
  if (args.colors && args.colors.length === 4) {
    fd.append("colors", JSON.stringify(args.colors));
  }
  return j(await fetch(u("/api/import"), { method: "POST", body: fd }));
}

/* ------------------------ Deep Clone ------------------------ */

/**
 * A "Deep Clone" training job — prepared on the backend by segmenting
 * + transcribing ~30 min of user-provided audio. The heavy lifting
 * (actual fine-tuning) happens in a shell script the user runs
 * themselves; the UI polls `status` every 30 s while a job is running.
 */
export type DeepCloneStatus =
  | "prepared"
  | "training"
  | "ready"
  | "failed";

export type DeepCloneManifest = {
  id: string;
  name: string;
  description: string;
  engine: "f5" | "xtts";
  created_at: string;
  segment_count: number;
  total_duration_s: number;
  train_script: string;
  checkpoint_glob: string;
  status: DeepCloneStatus;
  error?: string | null;
  checkpoint_path?: string | null;
  notes: string[];
};

export async function prepareDeepClone(args: {
  name: string;
  description?: string;
  engine: "f5" | "xtts";
  files: File[];
}): Promise<DeepCloneManifest> {
  const fd = new FormData();
  fd.append("name", args.name);
  if (args.description) fd.append("description", args.description);
  fd.append("engine", args.engine);
  for (const f of args.files) fd.append("audio_file", f);
  return j(await fetch(u("/api/deep-clone/prepare"), { method: "POST", body: fd }));
}

export async function listDeepCloneTrainings(): Promise<DeepCloneManifest[]> {
  const d = await j<{ trainings: DeepCloneManifest[] }>(
    await fetch(u("/api/deep-clone"))
  );
  return d.trainings;
}

export async function getDeepCloneTraining(id: string): Promise<DeepCloneManifest> {
  return j(await fetch(u(`/api/deep-clone/${encodeURIComponent(id)}`)));
}

export async function deleteDeepCloneTraining(id: string): Promise<void> {
  await j(
    await fetch(u(`/api/deep-clone/${encodeURIComponent(id)}`), {
      method: "DELETE",
    })
  );
}

export async function getDeepCloneScript(id: string): Promise<string> {
  const r = await fetch(u(`/api/deep-clone/${encodeURIComponent(id)}/script`));
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(
      `${r.status} ${r.statusText}${body ? `: ${body.slice(0, 300)}` : ""}`
    );
  }
  return r.text();
}

export async function registerDeepClone(id: string): Promise<{
  id: string;
  name: string;
  description: string;
  engine: "f5" | "xtts";
  checkpoint_path: string;
  ref_audio_path: string;
}> {
  return j(
    await fetch(u(`/api/deep-clone/${encodeURIComponent(id)}/register`), {
      method: "POST",
    })
  );
}

/** Preview a designed voice without persisting. Returns a blob object URL. */
export async function previewDesign(args: {
  baseVoice: string;
  engine: string;
  language?: string;
  pitch: number;
  speed: number;
  temperature: number;
  speakerName?: string | null;
  previewText?: string;
}): Promise<string> {
  const r = await fetch(u("/api/design/preview"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: designBody(args),
  });
  if (!r.ok) {
    const body = await r.text().catch(() => "");
    throw new Error(
      `${r.status} ${r.statusText}${body ? `: ${body.slice(0, 300)}` : ""}`
    );
  }
  const blob = await r.blob();
  return URL.createObjectURL(blob);
}
