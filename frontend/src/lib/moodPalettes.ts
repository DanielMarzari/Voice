/**
 * moodPalettes — hand-curated, aesthetically paired color quads that
 * replace the earlier slider-driven palette.
 *
 * Each mood defines a "base" palette and a few variants. Regenerate picks
 * one of the variants (or nudges HSL) so the user gets something fresh
 * without leaving the mood's emotional register.
 *
 * Shape: [warmA, warmB, highlight, coolAccent] — same ordering as the
 * VoiceSphere shader's uColor0..uColor3 uniforms.
 */

export type Palette = [string, string, string, string];

export type Mood = {
  id: string;
  label: string;
  description: string;
  variants: Palette[];
};

export const MOODS: Mood[] = [
  {
    id: "vibrant",
    label: "Vibrant",
    description: "Saturated and energetic — hot pink, electric blue, gold.",
    variants: [
      ["#ff3d6e", "#ffb03a", "#ffe066", "#4a5cff"],
      ["#ff5eb3", "#ff7038", "#9fff6e", "#5a3dff"],
      ["#ff2e89", "#ffa000", "#b3ff4a", "#00d4ff"],
      ["#ff4081", "#ffd166", "#06d6a0", "#8338ec"],
    ],
  },
  {
    id: "calm",
    label: "Calm",
    description: "Soft pastels — lavender, sky, cream, pale mint.",
    variants: [
      ["#d5b4ff", "#b9d3ff", "#ffe8d1", "#b8e8d4"],
      ["#e6c5ff", "#c5d8ff", "#fff0e0", "#a8ddc4"],
      ["#d9c2ff", "#b5cdf0", "#ffefd6", "#c7e4d1"],
      ["#e5d0ff", "#c9deff", "#f5e8d4", "#b0e0cc"],
    ],
  },
  {
    id: "sunset",
    label: "Sunset",
    description: "Coral + amber + deep purple + blush pink.",
    variants: [
      ["#ff6b4a", "#ffad5c", "#ffe3b8", "#7b4dff"],
      ["#ff5e62", "#ff9966", "#ffd29a", "#9f5cff"],
      ["#ff8a6b", "#ffbe76", "#ffd5a8", "#614385"],
      ["#f76b8a", "#ffa872", "#ffdeb5", "#6b4ee0"],
    ],
  },
  {
    id: "ocean",
    label: "Ocean",
    description: "Teal + aqua + navy + mint — cool and fluid.",
    variants: [
      ["#2b9fc8", "#4ad7d4", "#b8f0ff", "#1a3a82"],
      ["#1f8ab5", "#5ce0d4", "#a8e6ff", "#16305a"],
      ["#2a9df4", "#6ae0e0", "#c8f0ff", "#1f4a8a"],
      ["#2e86de", "#48dbfb", "#d5f0ff", "#0a3d62"],
    ],
  },
  {
    id: "forest",
    label: "Forest",
    description: "Moss + amber + deep green + cream.",
    variants: [
      ["#4a8d3f", "#c29d4a", "#ffe4a8", "#2b5a28"],
      ["#5ba04a", "#d4a64e", "#fff0b8", "#1e4a1f"],
      ["#6aa855", "#e0b860", "#fff4c2", "#2d5f33"],
      ["#528e45", "#c7a25b", "#f7e8b5", "#264c27"],
    ],
  },
  {
    id: "mono",
    label: "Monochrome",
    description: "Greys with a single accent. Editorial / minimalist.",
    variants: [
      ["#5a5a5c", "#9a9a9e", "#e0e0e2", "#ff4a6b"],
      ["#424246", "#8a8a90", "#d8d8da", "#4a8cff"],
      ["#6b6b70", "#a8a8ad", "#ebebed", "#ffa840"],
      ["#484850", "#8e8e95", "#d0d0d2", "#7a4aff"],
    ],
  },
  {
    id: "berry",
    label: "Berry",
    description: "Wine, plum, raspberry, cream.",
    variants: [
      ["#8e2751", "#c94278", "#ffd4e0", "#3f1a3a"],
      ["#a53860", "#d85a86", "#fcd7e1", "#4a1f45"],
      ["#9c2e5f", "#e0568c", "#ffe0eb", "#3a1530"],
      ["#b03a66", "#e46591", "#ffecf2", "#2e1128"],
    ],
  },
  {
    id: "citrus",
    label: "Citrus",
    description: "Lime, lemon, tangerine, teal.",
    variants: [
      ["#ffb627", "#c5e655", "#fff5b0", "#00a896"],
      ["#ffc24d", "#b8d633", "#fcf2a3", "#0b8a7e"],
      ["#ff9b2e", "#c2e14f", "#fff0a0", "#008a70"],
      ["#ffac1c", "#a5d94a", "#fce98a", "#0d8888"],
    ],
  },
];

export function moodById(id: string): Mood | undefined {
  return MOODS.find((m) => m.id === id);
}

// Small, deterministic PRNG from a seed so regenerate is repeatable.
function mulberry32(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = seed;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hexToHsl(hex: string): [number, number, number] {
  const r = parseInt(hex.slice(1, 3), 16) / 255;
  const g = parseInt(hex.slice(3, 5), 16) / 255;
  const b = parseInt(hex.slice(5, 7), 16) / 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;
  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    if (max === r) h = (g - b) / d + (g < b ? 6 : 0);
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
  }
  return [h, s, l];
}

function hslToHex(h: number, s: number, l: number): string {
  h = ((h % 360) + 360) % 360;
  s = Math.max(0, Math.min(1, s));
  l = Math.max(0, Math.min(1, l));
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let [r, g, b] = [0, 0, 0];
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const to = (v: number) =>
    Math.round((v + m) * 255).toString(16).padStart(2, "0");
  return `#${to(r)}${to(g)}${to(b)}`;
}

/**
 * Produce a palette for a given mood, seeded so the same (mood, seed)
 * always yields the same result. Pass a fresh Math.random()-derived seed
 * to the "Regenerate" button for variety.
 */
export function moodPalette(moodId: string, seed = 0): Palette {
  const mood = moodById(moodId) ?? MOODS[0];
  const rand = mulberry32(Math.floor(seed * 1e9) || 1);
  // Start from a random variant, then nudge each color in HSL space so
  // no two regenerates look identical.
  const base = mood.variants[Math.floor(rand() * mood.variants.length)];
  const out = base.map((hex) => {
    const [h, s, l] = hexToHsl(hex);
    const nh = h + (rand() - 0.5) * 16;       // ±8° hue
    const ns = s + (rand() - 0.5) * 0.10;     // ±5% saturation
    const nl = l + (rand() - 0.5) * 0.06;     // ±3% lightness
    return hslToHex(nh, ns, nl);
  }) as unknown as Palette;
  return out;
}
