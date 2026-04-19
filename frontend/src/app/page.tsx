"use client";

import { useCallback, useEffect, useState } from "react";
import { getHealth, type Health, type Profile } from "@/lib/api";
import { CloneTab } from "@/components/CloneTab";
import { DeepCloneTab } from "@/components/DeepCloneTab";
import { DesignTab } from "@/components/DesignTab";
import { ImportTab } from "@/components/ImportTab";
import { LibraryTab } from "@/components/LibraryTab";
import { SettingsModal } from "@/components/SettingsModal";
import { VoiceSphere } from "@/components/VoiceSphere";

type TabKey = "clone" | "design" | "import" | "deep-clone" | "library";

export default function VoiceStudioPage() {
  const [tab, setTab] = useState<TabKey>("clone");
  const [health, setHealth] = useState<Health | null>(null);
  const [libraryKey, setLibraryKey] = useState(0);
  const [settingsOpen, setSettingsOpen] = useState(false);

  const refreshHealth = useCallback(() => {
    getHealth().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    refreshHealth();
  }, [refreshHealth]);

  const onCreated = useCallback(
    (_p: Profile) => {
      setLibraryKey((k) => k + 1);
      refreshHealth();
    },
    [refreshHealth]
  );

  return (
    <div className="min-h-screen">
      <header className="border-b border-[color:var(--border)] bg-[color:var(--background)]/90 backdrop-blur sticky top-0 z-20">
        <div className="max-w-[1100px] mx-auto px-6 py-4 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <SphereLogo />
            <div>
              <div className="font-semibold">Voice Studio</div>
              <div className="text-xs text-[color:var(--muted)]">
                Local cloning & design for Reader
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <HealthPill health={health} />
            <button
              className="btn btn-ghost"
              onClick={() => setSettingsOpen(true)}
              title="Reader connection settings"
              aria-label="Settings"
            >
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden
              >
                <circle cx="12" cy="12" r="3" />
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
              </svg>
            </button>
          </div>
        </div>
        <div className="max-w-[1100px] mx-auto px-6 pb-3 flex gap-1">
          <TabBtn active={tab === "clone"} onClick={() => setTab("clone")} label="Clone a voice" />
          <TabBtn active={tab === "design"} onClick={() => setTab("design")} label="Design a voice" />
          <TabBtn active={tab === "import"} onClick={() => setTab("import")} label="Import a voice" />
          <TabBtn active={tab === "deep-clone"} onClick={() => setTab("deep-clone")} label="Deep Clone" />
          <TabBtn active={tab === "library"} onClick={() => setTab("library")} label="Library" />
        </div>
      </header>

      <main className="max-w-[1100px] mx-auto px-6 py-8">
        {tab === "clone" && <CloneTab onCreated={onCreated} />}
        {tab === "design" && <DesignTab onCreated={onCreated} />}
        {tab === "import" && <ImportTab onCreated={onCreated} />}
        {tab === "deep-clone" && <DeepCloneTab />}
        {tab === "library" && <LibraryTab refreshKey={libraryKey} />}
      </main>

      <SettingsModal
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        onSaved={refreshHealth}
        initialBaseUrl={health?.reader_base_url ?? null}
        tokenAlreadySet={!!health?.reader_token_set}
      />
    </div>
  );
}

function TabBtn({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      className={`tab ${active ? "tab-active" : ""}`}
      onClick={onClick}
      aria-pressed={active}
    >
      {label}
    </button>
  );
}

function HealthPill({ health }: { health: Health | null }) {
  if (!health) {
    return <span className="chip">⚠ Backend offline</span>;
  }
  const available = health.engines.filter((e) => e.available);
  const loaded = health.engines.filter((e) => e.loaded);
  const chipText =
    loaded.length > 0
      ? `● ${loaded.map((e) => e.id).join(", ")} on ${health.device}`
      : `${available.length}/${health.engines.length} engines available`;
  const tooltip = [
    health.reader_base_url ?? "Reader not connected",
    `Engines: ${health.engines
      .map((e) => `${e.id}${e.available ? "" : " (missing)"}${e.loaded ? " [loaded]" : ""}`)
      .join(", ")}`,
  ].join(" · ");

  return (
    <div className="flex items-center gap-2">
      {!health.reader_configured && (
        <span
          className="chip"
          title="Set READER_BASE_URL + READER_AUTH_TOKEN in .env.local"
        >
          ⚠ Reader not configured
        </span>
      )}
      <span className="chip" title={tooltip}>
        {chipText}
      </span>
    </div>
  );
}

function SphereLogo() {
  // Tiny sphere for the header — reuses the animated component.
  return <VoiceSphere seed="voice-studio-logo" size={32} />;
}
