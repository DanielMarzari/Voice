"use client";

/**
 * EnginePicker — shared control for both Clone and Design tabs.
 *
 * Shows each registered engine as a card with its license + language count,
 * and whether it's currently loaded in memory on the backend. User choice
 * is persisted in localStorage so it sticks across tabs.
 */

import { useEffect, useState } from "react";
import type { Engine } from "@/lib/api";

const STORAGE_KEY = "voice-studio:engine";

type Props = {
  engines: Engine[];
  value: string;
  onChange: (engineId: string) => void;
  includeLanguage?: boolean;
  language?: string;
  onLanguageChange?: (lang: string) => void;
};

// Minimal country-style labels for the language dropdown. XTTS supports
// all of these; F5 only EN + ZH — we filter based on chosen engine.
const LANG_LABELS: Record<string, string> = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  pl: "Polish",
  tr: "Turkish",
  ru: "Russian",
  nl: "Dutch",
  cs: "Czech",
  ar: "Arabic",
  zh: "Chinese",
  ja: "Japanese",
  hu: "Hungarian",
  ko: "Korean",
  hi: "Hindi",
};

export function EnginePicker({
  engines,
  value,
  onChange,
  includeLanguage = false,
  language = "en",
  onLanguageChange,
}: Props) {
  const chosen = engines.find((e) => e.id === value);

  return (
    <div>
      <label className="block text-sm font-medium mb-1.5">TTS engine</label>
      <div className="grid grid-cols-2 gap-3">
        {engines.map((e) => (
          <EngineCard
            key={e.id}
            engine={e}
            selected={value === e.id}
            onSelect={() => {
              onChange(e.id);
              try {
                localStorage.setItem(STORAGE_KEY, e.id);
              } catch {
                /* no-op */
              }
            }}
          />
        ))}
      </div>

      {includeLanguage && chosen && (
        <div className="mt-3">
          <label className="block text-xs font-medium text-[color:var(--muted)] mb-1">
            Language
          </label>
          <select
            className="select"
            value={language}
            onChange={(e) => onLanguageChange?.(e.target.value)}
            disabled={chosen.languages.length <= 1}
          >
            {chosen.languages.map((l) => (
              <option key={l} value={l}>
                {LANG_LABELS[l] ?? l.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
      )}
    </div>
  );
}

function EngineCard({
  engine,
  selected,
  onSelect,
}: {
  engine: Engine;
  selected: boolean;
  onSelect: () => void;
}) {
  const unavailable = !engine.available;
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={unavailable}
      className={[
        "card text-left transition-colors",
        selected ? "!border-[color:var(--accent)]" : "",
        unavailable ? "opacity-50 cursor-not-allowed" : "hover:border-[color:var(--muted-2)]",
      ].join(" ")}
      style={{
        cursor: unavailable ? "not-allowed" : "pointer",
        borderWidth: 2,
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="font-medium text-sm">{engine.label}</div>
        {engine.loaded && (
          <span
            className="text-[10px] uppercase tracking-wider text-[color:var(--muted)] whitespace-nowrap"
            title="Currently loaded in memory on the backend"
          >
            ● live
          </span>
        )}
      </div>
      <div className="text-xs text-[color:var(--muted)] mt-1">{engine.license}</div>
      <div className="text-xs text-[color:var(--muted)] mt-0.5">
        {engine.languages.length} language{engine.languages.length === 1 ? "" : "s"}
      </div>
      {unavailable && (
        <div className="text-xs text-red-500 mt-1">
          Not installed (see backend/requirements.txt)
        </div>
      )}
    </button>
  );
}

// Utility hook: load the persisted engine choice, falling back to backend default.
export function useEngineChoice(
  engines: Engine[] | null,
  backendDefault: string | null
): [string, (e: string) => void] {
  const [value, setValue] = useState<string>("");

  useEffect(() => {
    if (!engines || value) return;
    let initial: string | undefined;
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored && engines.some((e) => e.id === stored && e.available)) {
        initial = stored;
      }
    } catch {
      /* no-op */
    }
    if (!initial) {
      // Prefer backend default if available, otherwise first available engine.
      if (
        backendDefault &&
        engines.some((e) => e.id === backendDefault && e.available)
      ) {
        initial = backendDefault;
      } else {
        initial = engines.find((e) => e.available)?.id ?? engines[0]?.id;
      }
    }
    if (initial) setValue(initial);
  }, [engines, backendDefault, value]);

  return [value, setValue];
}
