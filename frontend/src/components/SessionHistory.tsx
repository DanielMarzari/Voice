"use client";

/**
 * SessionHistory — in-memory list of previews generated during this tab's
 * lifetime. Lets the user A/B between unsaved generations without having
 * to re-synthesize each one.
 *
 * Items are just blob URLs + metadata; they live only in component state
 * and go away on tab switch / page reload. (That's intentional — these
 * are meant to be throwaway comparisons, not long-term storage. If the
 * user wants to keep one, they hit "Save voice".)
 */

import { VoiceSphere } from "@/components/VoiceSphere";
import type { Palette } from "@/lib/moodPalettes";

export type SessionItem = {
  id: string;                // uuid-like, stable per entry
  blobUrl: string;           // object URL for <audio src>
  colors: Palette | null;    // palette used at generation time
  label: string;             // brief caption (e.g. "pitch +2, calm mood")
  createdAt: number;         // ms timestamp
};

type Props = {
  items: SessionItem[];
  activeId: string | null;
  onSelect: (item: SessionItem) => void;
  onClear?: () => void;
};

export function SessionHistory({ items, activeId, onSelect, onClear }: Props) {
  if (items.length === 0) return null;
  return (
    <div className="mt-4 w-full">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-medium text-[color:var(--muted)] uppercase tracking-wider">
          Session · {items.length} preview{items.length === 1 ? "" : "s"}
        </div>
        {onClear && (
          <button
            type="button"
            onClick={onClear}
            className="text-[11px] text-[color:var(--muted)] hover:text-red-500"
          >
            Clear
          </button>
        )}
      </div>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {items.map((it) => {
          const isActive = it.id === activeId;
          return (
            <button
              key={it.id}
              onClick={() => onSelect(it)}
              className={[
                "flex-shrink-0 flex flex-col items-center gap-1 p-2 rounded-lg transition-colors",
                isActive
                  ? "bg-[color:var(--surface-2)] ring-2 ring-[color:var(--accent)]"
                  : "hover:bg-[color:var(--surface-2)]",
              ].join(" ")}
              style={{ width: 84 }}
              title={`${it.label} · ${new Date(it.createdAt).toLocaleTimeString()}`}
            >
              <VoiceSphere
                seed={it.id}
                size={52}
                colors={it.colors ?? undefined}
                speaking={isActive}
              />
              <div className="text-[10px] leading-tight text-center text-[color:var(--muted)] line-clamp-2 w-full">
                {it.label}
              </div>
            </button>
          );
        })}
      </div>
      <div className="text-[11px] text-[color:var(--muted)] mt-1">
        Throwaway previews — hit <em>Save voice</em> above to keep one.
      </div>
    </div>
  );
}
