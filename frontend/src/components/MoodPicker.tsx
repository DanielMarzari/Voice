"use client";

/**
 * MoodPicker — compact mood dropdown + regenerate button + live color chips.
 * Designed to sit beneath the preview sphere on the right side of a tab
 * (narrow width), so the chip row wraps instead of the dropdown.
 */

import { MOODS, type Palette } from "@/lib/moodPalettes";

type Props = {
  moodId: string;
  onMoodChange: (id: string) => void;
  onRegenerate: () => void;
  palette: Palette;
  compact?: boolean;
};

export function MoodPicker({
  moodId,
  onMoodChange,
  onRegenerate,
  palette,
  compact,
}: Props) {
  return (
    <div className={compact ? "w-full" : ""}>
      <label className="block text-xs font-medium text-[color:var(--muted)] mb-1 uppercase tracking-wider">
        Color mood
      </label>
      <div className="flex gap-1.5">
        <select
          className="select flex-1 !py-1.5 text-sm"
          value={moodId}
          onChange={(e) => onMoodChange(e.target.value)}
        >
          {MOODS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          className="btn !py-1.5 !px-2.5 text-sm"
          onClick={onRegenerate}
          title="Reshuffle colors within this mood"
          aria-label="Regenerate palette"
        >
          ⟳
        </button>
      </div>
      <div className="flex gap-1 mt-2 items-center">
        {palette.map((c, i) => (
          <span
            key={i}
            className="inline-block w-4 h-4 rounded-full border border-[color:var(--border)]"
            style={{ background: c }}
            aria-hidden
          />
        ))}
      </div>
      <div className="text-[11px] text-[color:var(--muted)] mt-1.5 leading-snug">
        {MOODS.find((m) => m.id === moodId)?.description}
      </div>
    </div>
  );
}
