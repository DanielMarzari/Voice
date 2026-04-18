"use client";

/**
 * GenerationProgress — indeterminate progress bar + elapsed-time counter
 * shown while a preview or save synthesis is in flight.
 *
 * We can't show a real percentage because F5-TTS and XTTS don't stream
 * progress events; we just know synthesis is running. So: a sweeping
 * indeterminate bar, a live "elapsed / estimated" line, and a nudge
 * after the estimate passes ("taking longer than usual…") so the user
 * knows the app isn't frozen.
 */

import { useEffect, useState } from "react";

type Props = {
  /** "preview" | "save" — controls the wording. */
  kind: "preview" | "save";
  /** Rough expected duration in seconds. On M2 MPS: preview ~8-15s,
   *  save ~10-20s (extra round-trip to Reader). First call after boot
   *  is slower due to model load. */
  estimateSeconds?: number;
};

export function GenerationProgress({ kind, estimateSeconds = 15 }: Props) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const start = performance.now();
    const id = setInterval(() => {
      setElapsed(Math.floor((performance.now() - start) / 1000));
    }, 250);
    return () => clearInterval(id);
  }, []);

  const overdue = elapsed > estimateSeconds * 1.5;
  const verb = kind === "preview" ? "Generating preview" : "Saving voice";

  return (
    <div className="mt-3 w-full max-w-sm mx-auto">
      <div
        className="h-1.5 w-full rounded-full overflow-hidden"
        style={{ background: "var(--surface-2)" }}
        role="progressbar"
        aria-valuetext={`${elapsed}s elapsed`}
      >
        <div
          className="gen-progress-sweep h-full rounded-full"
          style={{
            width: "40%",
            background:
              "linear-gradient(90deg, transparent, var(--accent), transparent)",
          }}
        />
      </div>
      <div className="text-xs text-[color:var(--muted)] mt-2 flex justify-between items-baseline">
        <span>
          {verb}… <span className="tabular-nums">{elapsed}s</span>
        </span>
        <span className="tabular-nums text-[color:var(--muted-2)]">
          {overdue ? "taking longer than usual…" : `est ${estimateSeconds}s`}
        </span>
      </div>
    </div>
  );
}
