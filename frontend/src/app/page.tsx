"use client";

import { useCallback, useEffect, useState } from "react";
import { getHealth, type Health, type Profile } from "@/lib/api";
import { CloneTab } from "@/components/CloneTab";
import { DesignTab } from "@/components/DesignTab";
import { LibraryTab } from "@/components/LibraryTab";

type TabKey = "clone" | "design" | "library";

export default function VoiceStudioPage() {
  const [tab, setTab] = useState<TabKey>("clone");
  const [health, setHealth] = useState<Health | null>(null);
  const [libraryKey, setLibraryKey] = useState(0);

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
          <HealthPill health={health} />
        </div>
        <div className="max-w-[1100px] mx-auto px-6 pb-3 flex gap-1">
          <TabBtn active={tab === "clone"} onClick={() => setTab("clone")} label="Clone a voice" />
          <TabBtn active={tab === "design"} onClick={() => setTab("design")} label="Design a voice" />
          <TabBtn active={tab === "library"} onClick={() => setTab("library")} label="Library" />
        </div>
      </header>

      <main className="max-w-[1100px] mx-auto px-6 py-8">
        {tab === "clone" && <CloneTab onCreated={onCreated} />}
        {tab === "design" && <DesignTab onCreated={onCreated} />}
        {tab === "library" && <LibraryTab refreshKey={libraryKey} />}
      </main>
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
  // Little static version of the sphere for the header.
  return (
    <div
      className="voice-sphere voice-sphere-static"
      style={
        {
          width: "32px",
          height: "32px",
          "--c1": "#f6b4a6",
          "--c2": "#d58cff",
          "--c3": "#6190ff",
          "--c4": "#9effd8",
        } as React.CSSProperties & Record<string, string>
      }
      aria-hidden
    />
  );
}
