"use client";

import { useEffect, useMemo, useState } from "react";
import {
  deleteProfile,
  listProfiles,
  sampleUrl,
  syncProfile,
  type Profile,
} from "@/lib/api";
import { VoiceSphere } from "@/components/VoiceSphere";

type Props = {
  refreshKey: number;
};

export function LibraryTab({ refreshKey }: Props) {
  const [profiles, setProfiles] = useState<Profile[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  const [syncingId, setSyncingId] = useState<string | null>(null);
  const [syncAllBusy, setSyncAllBusy] = useState(false);
  const [syncAllStatus, setSyncAllStatus] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    listProfiles()
      .then((ps) => alive && setProfiles(ps))
      .catch((e) => alive && setError((e as Error).message));
    return () => {
      alive = false;
    };
  }, [refreshKey]);

  const unsyncedCount = useMemo(
    () => (profiles ?? []).filter((p) => !p.synced).length,
    [profiles]
  );

  async function handleDelete(id: string) {
    if (!confirm("Delete this voice locally and on the server?")) return;
    try {
      await deleteProfile(id);
      setProfiles((ps) => ps?.filter((p) => p.id !== id) ?? null);
    } catch (e) {
      alert((e as Error).message);
    }
  }

  async function handleSync(id: string) {
    setSyncingId(id);
    try {
      const updated = await syncProfile(id);
      setProfiles((ps) =>
        ps ? ps.map((p) => (p.id === id ? updated : p)) : ps
      );
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setSyncingId(null);
    }
  }

  async function handleSyncAll() {
    if (!profiles || syncAllBusy) return;
    const unsynced = profiles.filter((p) => !p.synced);
    if (unsynced.length === 0) return;
    setSyncAllBusy(true);
    setSyncAllStatus(null);
    let success = 0;
    const errors: string[] = [];
    for (let i = 0; i < unsynced.length; i++) {
      const p = unsynced[i];
      setSyncAllStatus(`Syncing ${p.name} (${i + 1}/${unsynced.length})…`);
      try {
        const updated = await syncProfile(p.id);
        setProfiles((ps) =>
          ps ? ps.map((x) => (x.id === p.id ? updated : x)) : ps
        );
        success++;
      } catch (e) {
        errors.push(`${p.name}: ${(e as Error).message.slice(0, 120)}`);
      }
    }
    setSyncAllBusy(false);
    setSyncAllStatus(
      errors.length
        ? `Synced ${success}/${unsynced.length}. ${errors.length} failed.`
        : `Synced all ${success} voices.`
    );
    if (errors.length) {
      console.error("Sync all errors:", errors);
    }
    // Clear the status line after a few seconds.
    setTimeout(() => setSyncAllStatus(null), 5000);
  }

  if (error) return <div className="text-sm text-red-500">{error}</div>;
  if (profiles === null)
    return <div className="text-sm text-[color:var(--muted)]">Loading…</div>;

  if (profiles.length === 0) {
    return (
      <div className="text-center py-16">
        <div className="text-5xl opacity-40 mb-3">🎙</div>
        <div className="text-base font-medium mb-1">No voices yet</div>
        <div className="text-sm text-[color:var(--muted)]">
          Head to the Clone or Design tab to make your first voice.
        </div>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-5">
        <div className="text-sm text-[color:var(--muted)]">
          {profiles.length} voice{profiles.length === 1 ? "" : "s"}
          {unsyncedCount > 0 && (
            <span> · {unsyncedCount} not yet synced to Reader</span>
          )}
        </div>
        <div className="flex items-center gap-3">
          {syncAllStatus && (
            <span className="text-xs text-[color:var(--muted)]">{syncAllStatus}</span>
          )}
          <button
            className="btn"
            onClick={handleSyncAll}
            disabled={unsyncedCount === 0 || syncAllBusy}
            title="Push every local-only voice to Reader"
          >
            {syncAllBusy ? "Syncing…" : `⇪ Sync all${unsyncedCount ? ` (${unsyncedCount})` : ""}`}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
        {profiles.map((p) => (
          <div
            key={p.id}
            className="card flex flex-col items-center gap-3 text-center"
          >
            <VoiceSphere
              seed={p.id}
              size={120}
              speaking={playing === p.id}
              withPlayIcon={playing !== p.id}
              onClick={() => setPlaying(playing === p.id ? null : p.id)}
            />
            <div>
              <div className="font-medium">{p.name}</div>
              <div className="text-xs text-[color:var(--muted)] mt-0.5">
                {p.kind === "cloned"
                  ? "Cloned"
                  : p.kind === "uploaded"
                  ? "Imported"
                  : "Designed"}
                {p.duration_s != null && ` · ${p.duration_s.toFixed(0)}s`}
                {" · "}
                {p.synced ? "synced" : "local only"}
              </div>
              {p.prompt_text && (
                <div
                  className="text-[11px] text-[color:var(--muted)] italic mt-1.5 line-clamp-2"
                  title={p.prompt_text}
                >
                  &ldquo;{p.prompt_text}&rdquo;
                </div>
              )}
            </div>
            {playing === p.id && (
              <audio
                autoPlay
                controls
                className="w-full"
                src={sampleUrl(p.id)}
                onEnded={() => setPlaying(null)}
              />
            )}
            <div className="flex items-center gap-1 mt-auto">
              {!p.synced && (
                <button
                  className="btn btn-ghost text-xs"
                  onClick={() => handleSync(p.id)}
                  disabled={syncingId === p.id}
                  title="Push this voice to Reader"
                >
                  {syncingId === p.id ? "Syncing…" : "Sync"}
                </button>
              )}
              <button
                className="btn btn-ghost !p-1.5"
                onClick={() => handleDelete(p.id)}
                title="Delete this voice (local + Reader)"
                aria-label="Delete"
              >
                <TrashIcon />
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TrashIcon() {
  // Inline SVG so we don't pull in an icon dep. currentColor = foreground;
  // the parent sets color on hover for the red-500 affordance.
  return (
    <svg
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
      className="text-[color:var(--muted)] hover:text-red-500 transition-colors"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}
