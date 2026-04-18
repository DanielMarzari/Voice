"use client";

/**
 * VoiceSphere — ElevenLabs-style soft planetary orb.
 *
 * Rendering model: 4 large, heavily-blurred radial-gradient blobs drift
 * independently inside a circular mask. Prime-number durations keep them
 * from ever resyncing, so the motion reads organic like clouds/water.
 * When `speaking` is true the blobs pull toward the center and a ripple
 * emanates outward — that's the "disruption pattern."
 *
 * IMPORTANT: Duplicated verbatim between Reader and Voice repos. When you
 * change one, copy to the other. CSS for .voice-sphere* lives in each
 * app's globals.css.
 */

import { useMemo } from "react";

type Props = {
  seed: string;
  size?: number;                // pixels
  speaking?: boolean;            // when true → disruption/pulse animation
  withPlayIcon?: boolean;
  className?: string;
  onClick?: () => void;
  ariaLabel?: string;
};

// Hand-tuned pastel palettes, inspired by the ElevenLabs sample spheres:
// warm blended hues (coral/peach/lavender/mint) rather than saturated primaries.
const PALETTES: Array<[string, string, string, string]> = [
  ["#ffb3a3", "#ffd9a0", "#e4b8ff", "#a0c4ff"],   // coral + peach + lavender + sky
  ["#ffd1a0", "#ff9aa8", "#c4a8ff", "#a8e0ff"],   // amber + pink + violet + ice
  ["#ffc5c5", "#ffe0b3", "#c8b6ff", "#b3e5d1"],   // blush + cream + iris + mint
  ["#ff9aaf", "#ffb38a", "#e0a8ff", "#8ec9ff"],   // rose + tangerine + lilac + powder
  ["#ffd1dc", "#c8f0d4", "#b3d1ff", "#e8c8ff"],   // cotton-candy pastel
  ["#ffbaa0", "#ffd694", "#ffc1f0", "#b8ecff"],   // sunset warm
  ["#c0e7d9", "#a8c8ff", "#d4b3ff", "#ffc0d4"],   // cool → warm spectrum
  ["#fff0c2", "#ffb3a3", "#d1a8ff", "#a3e0ff"],   // butter + coral + iris + aqua
  ["#e8c8ff", "#ff9ec7", "#ffcba8", "#ffe8a8"],   // pink-gold arcade
  ["#b3f0e0", "#a8d4ff", "#c8a8ff", "#ffb3d9"],   // aurora
];

function pickPalette(seed: string): [string, string, string, string] {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const idx = Math.abs(h) % PALETTES.length;
  return PALETTES[idx];
}

// Each blob gets its own drift "seed" so four spheres in a row don't
// all start in identical positions. We derive an offset per (seed, index).
function driftSeed(seed: string, index: number): number {
  let h = 2166136261;
  const s = `${seed}:${index}`;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h);
}

export function VoiceSphere({
  seed,
  size = 160,
  speaking = false,
  withPlayIcon = false,
  className = "",
  onClick,
  ariaLabel,
}: Props) {
  const colors = useMemo(() => pickPalette(seed), [seed]);

  // Blob params: each drifts at a different prime-durated tempo starting
  // from a different phase so they never align.
  const blobs = useMemo(
    () =>
      colors.map((color, i) => {
        const s = driftSeed(seed, i);
        // Negative animation-delay starts each blob mid-cycle so we don't
        // see a "sync bloom" on first paint.
        return {
          color,
          delay: -((s % 100) / 100) * 20,  // up to -20s
          originX: 30 + ((s >> 3) % 40),   // 30..70 %
          originY: 30 + ((s >> 7) % 40),
        };
      }),
    [seed, colors]
  );

  const cls = [
    "voice-sphere",
    speaking ? "voice-sphere--speaking" : "",
    onClick ? "cursor-pointer" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  const outerStyle: React.CSSProperties = {
    width: `${size}px`,
    height: `${size}px`,
    // Background under the blobs so there's no flash before they render.
    background: `radial-gradient(circle at 40% 40%, ${colors[0]}, ${colors[3]})`,
  };

  return (
    <div
      className={cls}
      style={outerStyle}
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      aria-label={ariaLabel}
      onKeyDown={(e) => {
        if (onClick && (e.key === "Enter" || e.key === " ")) {
          e.preventDefault();
          onClick();
        }
      }}
    >
      {blobs.map((b, i) => (
        <div
          key={i}
          className={`voice-sphere-blob voice-sphere-blob-${i + 1}`}
          style={
            {
              "--blob-color": b.color,
              "--blob-x": `${b.originX}%`,
              "--blob-y": `${b.originY}%`,
              animationDelay: `${b.delay}s`,
            } as React.CSSProperties & Record<string, string>
          }
        />
      ))}

      {/* Specular highlight — the glossy "top-left" sheen that sells it as 3D. */}
      <div className="voice-sphere-gloss" aria-hidden />

      {/* Disruption ring — only visible while speaking. */}
      {speaking && <div className="voice-sphere-ripple" aria-hidden />}

      {withPlayIcon && !speaking && (
        <div className="voice-sphere-play" aria-hidden>
          <svg width="38%" height="38%" viewBox="0 0 24 24" fill="currentColor" aria-hidden>
            <path d="M8 5v14l11-7z" />
          </svg>
        </div>
      )}
    </div>
  );
}
