/**
 * designPalette — derive a 4-color sphere palette from the Design tab's
 * live slider values.
 *
 * Each slider owns a distinct hue family so the resulting palette actually
 * covers the color wheel — red / yellow / green / blue-violet — instead of
 * sitting in the coral-pink-peach corner. Rule (c) "always contrast" is
 * satisfied by the fixed hue assignment: no matter where the sliders sit,
 * you get one warm red, one yellow/gold, one green/teal, and one cool
 * blue-violet. Slider value only controls brightness within each hue.
 *
 * Mapping:
 *   color0 (red family)        ← pitch        (deep crimson → bright coral-red)
 *   color1 (yellow family)     ← speed        (dark amber   → bright butter)
 *   color2 (green family)      ← temperature  (forest teal  → bright mint)
 *   color3 (blue-violet family) ← fixed contrast anchor (indigo → sky-violet),
 *                                parameterized by warm-average so the cool
 *                                stays visible whatever the warms are doing.
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

  // Red family (pitch): crimson → bright coral-red.
  const color0 = mixHex("#5a1820", "#ff6a5a", pN);
  // Yellow family (speed): dark amber → bright butter.
  const color1 = mixHex("#5a3d10", "#ffd860", sN);
  // Green family (temperature): dark forest → bright mint.
  const color2 = mixHex("#0f3a2a", "#80f2b8", tN);
  // Cool anchor (blue-violet) — brightness inversely ties to warm average
  // so when warms are bright the cool becomes deeper (contrast), and vice
  // versa. Never goes fully black — hue is always visible.
  const warmAvg = (pN + sN + tN) / 3;
  const color3 = mixHex("#1d1e4a", "#8ea6ff", 1 - Math.abs(warmAvg - 0.5) * 1.4);

  return [color0, color1, color2, color3];
}
