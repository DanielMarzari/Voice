"use client";

import { useEffect, useState } from "react";
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

  useEffect(() => {
    let alive = true;
    listProfiles()
      .then((ps) => alive && setProfiles(ps))
      .catch((e) => alive && setError((e as Error).message));
    return () => {
      alive = false;
    };
  }, [refreshKey]);

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
    try {
      const updated = await syncProfile(id);
      setProfiles((ps) =>
        ps ? ps.map((p) => (p.id === id ? updated : p)) : ps
      );
    } catch (e) {
      alert((e as Error).message);
    }
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
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-6">
      {profiles.map((p) => (
        <div key={p.id} className="card flex flex-col items-center gap-3 text-center">
          <VoiceSphere
            seed={p.id}
            size={120}
            withPlayIcon={playing !== p.id}
            onClick={() => setPlaying(playing === p.id ? null : p.id)}
          />
          <div>
            <div className="font-medium">{p.name}</div>
            <div className="text-xs text-[color:var(--muted)] mt-0.5">
              {p.kind === "cloned" ? "Cloned" : "Designed"} ·{" "}
              {p.synced ? "synced" : "local only"}
            </div>
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
          <div className="flex items-center gap-2 mt-auto">
            {!p.synced && (
              <button className="btn btn-ghost text-xs" onClick={() => handleSync(p.id)}>
                Sync
              </button>
            )}
            <button
              className="btn btn-ghost text-xs text-red-500"
              onClick={() => handleDelete(p.id)}
            >
              Delete
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
