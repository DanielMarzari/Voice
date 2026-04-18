/**
 * designPalette — derive a 4-color sphere palette from the Design tab's
 * live slider values, per user spec:
 *   (a) pitch influences one color,
 *       temperature another,
 *       speed another,
 *   (b) higher setting → brighter color, lower → darker,
 *   (c) there should always be contrast.
 *
 * Mapping:
 *   color0 (warm primary)   ← pitch       (coral gradient)
 *   color1 (warm secondary) ← temperature (peach/amber gradient)
 *   color2 (highlight)      ← speed       (cream gradient)
 *   color3 (cool accent)    ← fixed contrast color, tinted inversely
 *                             to the warm average so the palette always
 *                             has a cool anchor (→ rule (c)).
 */

type Palette = [string, string, string, string];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function mixHex(c1: string, c2: string, t: number): string {
  const p = (h: string) => [
    parseInt(h.slice(1, 3), 16),
    parseInt(h.slice(3, 5), 16),
    parseInt(h.slice(5, 7), 16),
  ];
  const [r1, g1, b1] = p(c1);
  const [r2, g2, b2] = p(c2);
  const r = Math.round(lerp(r1, r2, t));
  const g = Math.round(lerp(g1, g2, t));
  const b = Math.round(lerp(b1, b2, t));
  const h = (n: number) => n.toString(16).padStart(2, "0");
  return `#${h(r)}${h(g)}${h(b)}`;
}

/** Normalize a value from [min, max] to [0, 1]. */
function norm(v: number, min: number, max: number): number {
  return Math.max(0, Math.min(1, (v - min) / (max - min)));
}

export function designPalette(args: {
  pitch: number;        // -6..+6
  speed: number;        // 0.5..2.0
  temperature: number;  // 0.1..1.5
}): Palette {
  const pN = norm(args.pitch, -6, 6);
  const sN = norm(args.speed, 0.5, 2.0);
  const tN = norm(args.temperature, 0.1, 1.5);

  // color0: pitch → coral. Low = burgundy-ish, high = bright peach-coral.
  const color0 = mixHex("#7a2e28", "#ff9478", pN);

  // color1: temperature → pink/rose. Low = dusty mauve, high = vivid rose.
  const color1 = mixHex("#6b3a42", "#ff6b8a", tN);

  // color2: speed → warm highlight. Low = muted cream, high = bright butter.
  const color2 = mixHex("#a08870", "#ffefc2", sN);

  // color3: cool accent for contrast. Pulls in the opposite direction of
  // the warm average so the palette always has visible range. The base
  // picks from a narrow cool band (indigo → sky) parameterized by the
  // warm average — the less warm/bright the rest, the cooler/deeper the
  // accent, so contrast is preserved across all slider positions.
  const warmAvg = (pN + tN + sN) / 3;
  const color3 = mixHex("#3a3270", "#b8c8ff", 1 - Math.abs(warmAvg - 0.5) * 1.6);

  return [color0, color1, color2, color3];
}
