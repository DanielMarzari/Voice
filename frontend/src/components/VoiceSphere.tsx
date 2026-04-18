"use client";

/**
 * VoiceSphere — WebGL fragment shader modeled after ElevenLabs' sphere.
 *
 * What it does (GPU, every frame):
 *   1. Build a UV in [-1, 1] centered at the middle of the canvas.
 *   2. Compute 4-octave FBM noise at two offset positions to get a flow
 *      field, then use that flow to displace the UV. This is the classic
 *      "domain warping" technique — produces fluid-like, watercolor motion.
 *   3. Read a third FBM at the displaced position to drive color mixing
 *      between a per-voice 4-color palette.
 *   4. Apply specular highlight + edge shadow for 3D feel.
 *   5. Use a smoothstep against distance-from-center as the circular
 *      alpha mask. No <div>-shaped edges; the circle lives in the shader.
 *
 * Speaking state ramps `uSpeaking` from 0 → 1 (smoothed), which
 *   - ~doubles the displacement amplitude (the "disruption"),
 *   - speeds up the flow,
 *   - and adds an outward concentric ring that pulses from the center.
 *
 * Performance:
 *   - IntersectionObserver pauses the render loop when the sphere is
 *     scrolled off-screen.
 *   - Resources are disposed on unmount; a `webglcontextlost` handler
 *     is installed so the browser can reclaim memory under pressure.
 *
 * Duplicated verbatim between Reader and Voice repos. Keep in sync.
 */

import { useEffect, useMemo, useRef } from "react";

type Props = {
  seed: string;
  size?: number;
  speaking?: boolean;
  withPlayIcon?: boolean;
  className?: string;
  onClick?: () => void;
  ariaLabel?: string;
};

// Higher-saturation palettes calibrated against the actual ElevenLabs
// sphere video: coral/peach warms + lavender/periwinkle cools + a pale
// highlight. Each tuple is [warm-A, warm-B, highlight, cool].
const PALETTES: Array<[string, string, string, string]> = [
  ["#ff7050", "#ff9878", "#fee4d1", "#b3a8ff"], // coral + lavender (flagship)
  ["#ff8060", "#ffbb88", "#ffe8c8", "#9b9fe8"], // peach + periwinkle
  ["#ff7a95", "#ffaa8a", "#ffe0d0", "#c8a8ff"], // rose + iris
  ["#ff6b7e", "#ffb38a", "#fff0d8", "#7fb3ff"], // pink → sky
  ["#ff9060", "#ffcf9a", "#fff4dc", "#a08eff"], // amber → violet
  ["#ff7aa8", "#ffb0b0", "#ffe8f0", "#8ac8ff"], // blush → powder
  ["#ff8866", "#ffd8a0", "#fff2c8", "#9bdcff"], // sunset
  ["#e47a88", "#ffa890", "#fde0d6", "#8b9cff"], // dusty rose
  ["#ff7a50", "#ffba78", "#ffe8c0", "#d0a8ff"], // tangerine → lilac
  ["#ff6080", "#ff98c0", "#ffe0e8", "#90c0ff"], // hot pink → ice
];

function hexToRgb01(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  const r = parseInt(h.slice(0, 2), 16) / 255;
  const g = parseInt(h.slice(2, 4), 16) / 255;
  const b = parseInt(h.slice(4, 6), 16) / 255;
  return [r, g, b];
}

// FNV-1a hash → deterministic palette pick from the seed string.
function hashSeed(seed: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < seed.length; i++) {
    h ^= seed.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

// ---------------- Shaders ----------------
// WebGL2 / GLSL ES 3.00. Supported in all modern browsers (>95%).

const VERT_SHADER = `#version 300 es
in vec2 aPos;
out vec2 vUv;
void main() {
  vUv = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}
`;

const FRAG_SHADER = `#version 300 es
precision highp float;

in vec2 vUv;
out vec4 outColor;

uniform float uTime;        // seconds
uniform float uSpeaking;    // smoothed 0..1
uniform vec3  uColor0;
uniform vec3  uColor1;
uniform vec3  uColor2;
uniform vec3  uColor3;
uniform float uSeed;        // drift offset per-voice

// ---------- noise ----------
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  f = f*f*(3.0 - 2.0*f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a,b,f.x), mix(c,d,f.x), f.y);
}
float fbm(vec2 p) {
  float v = 0.0;
  float a = 0.5;
  mat2 rot = mat2(0.80, 0.60, -0.60, 0.80);
  for (int i = 0; i < 4; i++) {
    v += a * vnoise(p);
    p = rot * p * 2.02;
    a *= 0.5;
  }
  return v;
}

// ---------- main ----------
void main() {
  // Center UV in [-1, 1]. Distance used for circular mask + shading.
  vec2 uv = vUv * 2.0 - 1.0;
  float r = length(uv);

  // Early discard beyond the circle (saves fragment work on gallery pages
  // with lots of spheres). 1.02 keeps antialiasing headroom.
  if (r > 1.02) {
    outColor = vec4(0.0);
    return;
  }

  // Flow speed gets a 3x boost while speaking — more roiling motion.
  float t = uTime * (0.18 + 0.42 * uSpeaking) + uSeed;

  // Displacement amplitude: 0.35 calm → 0.75 speaking. This is what makes
  // speaking look "disrupted" rather than just "faster."
  float amp = mix(0.35, 0.75, uSpeaking);

  // Domain warping: two passes of FBM where each pass's output perturbs
  // the next pass's input. Produces the fluid/smoke look.
  vec2 q = vec2(
    fbm(uv * 1.3 + vec2(0.00, t * 0.12)),
    fbm(uv * 1.3 + vec2(2.10, t * 0.09))
  );
  vec2 s = vec2(
    fbm(uv * 1.3 + q * 1.8 + vec2(t * 0.14, 0.00)),
    fbm(uv * 1.3 + q * 1.8 + vec2(0.00, t * 0.16))
  );
  float n = fbm(uv * 1.3 + s * amp);

  // Four-way color mix. q.x, s.y, n each drive a different blend, so the
  // result is never just linear between two colors — you get the mottled,
  // multi-hued look of the reference.
  vec3 color = mix(uColor0, uColor1, smoothstep(0.25, 0.85, n));
  color = mix(color, uColor2, smoothstep(0.30, 0.95, s.y));
  color = mix(color, uColor3, smoothstep(0.55, 1.05, q.x * 0.8 + n * 0.6));

  // Slight warm bias — pulls toward the coral dominant in the reference.
  color = pow(color, vec3(0.92));

  // 3D shading: bright top-left highlight, dark bottom-right shadow.
  // Makes the sphere read as a ball, not a flat disc.
  vec2 toHi = uv - vec2(-0.35, -0.45);
  float hi = pow(clamp(1.0 - length(toHi) * 0.85, 0.0, 1.0), 2.4);
  color += vec3(0.30, 0.22, 0.18) * hi;

  float edgeShadow = smoothstep(0.55, 1.0, r);
  color *= mix(1.0, 0.72, edgeShadow);
  // Extra darkening at the very rim for depth.
  color *= mix(1.0, 0.55, smoothstep(0.88, 1.0, r));

  // Subtle grain so it doesn't look plasticky — low amplitude.
  float grain = (hash(vUv * 920.0 + t * 0.1) - 0.5) * 0.035;
  color += grain;

  // Speaking-state disruption ring: a thin bright band that travels from
  // center outward on a ~1.2s cycle while uSpeaking > 0.
  if (uSpeaking > 0.02) {
    float cycle = fract(uTime * 0.8);
    float ring = smoothstep(0.03, 0.0, abs(r - cycle)) *
                 (1.0 - cycle) *                 // fade out as it expands
                 uSpeaking * 0.35;
    color += vec3(1.0) * ring;
  }

  // Circular alpha with 1px smooth edge.
  float aa = fwidth(r) * 0.9;
  float alpha = 1.0 - smoothstep(1.0 - aa, 1.0, r);
  outColor = vec4(color, alpha);
}
`;

// ---------------- WebGL setup helpers ----------------

function makeShader(gl: WebGL2RenderingContext, type: number, src: string) {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("Shader compile failed: " + log);
  }
  return sh;
}

function makeProgram(gl: WebGL2RenderingContext, vs: string, fs: string) {
  const v = makeShader(gl, gl.VERTEX_SHADER, vs);
  const f = makeShader(gl, gl.FRAGMENT_SHADER, fs);
  const p = gl.createProgram()!;
  gl.attachShader(p, v);
  gl.attachShader(p, f);
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p);
    gl.deleteProgram(p);
    throw new Error("Program link failed: " + log);
  }
  gl.deleteShader(v);
  gl.deleteShader(f);
  return p;
}

// ---------------- Component ----------------

export function VoiceSphere({
  seed,
  size = 160,
  speaking = false,
  withPlayIcon = false,
  className = "",
  onClick,
  ariaLabel,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const speakingRef = useRef(speaking);
  speakingRef.current = speaking;

  // Stable per-seed numeric seed (becomes a uniform) + picked palette.
  const { palette, numericSeed } = useMemo(() => {
    const h = hashSeed(seed);
    return { palette: PALETTES[h % PALETTES.length], numericSeed: (h & 0xffff) / 0xffff * 100 };
  }, [seed]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = size * dpr;
    canvas.height = size * dpr;

    const gl = canvas.getContext("webgl2", {
      antialias: true,
      premultipliedAlpha: true,
      alpha: true,
    }) as WebGL2RenderingContext | null;

    if (!gl) {
      // WebGL2 unavailable — render nothing (parent still shows container).
      // Could add a <canvas> → <div> gradient fallback here if needed.
      return;
    }

    let program: WebGLProgram | null;
    try {
      program = makeProgram(gl, VERT_SHADER, FRAG_SHADER);
    } catch (e) {
      console.warn("VoiceSphere shader compile failed:", e);
      return;
    }

    // Full-screen triangle (faster than a quad, no seams).
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW
    );
    const aPos = gl.getAttribLocation(program, "aPos");
    gl.enableVertexAttribArray(aPos);
    gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

    const uTime = gl.getUniformLocation(program, "uTime");
    const uSpeaking = gl.getUniformLocation(program, "uSpeaking");
    const uSeed = gl.getUniformLocation(program, "uSeed");
    const uC0 = gl.getUniformLocation(program, "uColor0");
    const uC1 = gl.getUniformLocation(program, "uColor1");
    const uC2 = gl.getUniformLocation(program, "uColor2");
    const uC3 = gl.getUniformLocation(program, "uColor3");

    gl.useProgram(program);
    gl.uniform3fv(uC0, hexToRgb01(palette[0]));
    gl.uniform3fv(uC1, hexToRgb01(palette[1]));
    gl.uniform3fv(uC2, hexToRgb01(palette[2]));
    gl.uniform3fv(uC3, hexToRgb01(palette[3]));
    gl.uniform1f(uSeed, numericSeed);

    gl.clearColor(0, 0, 0, 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.viewport(0, 0, canvas.width, canvas.height);

    // State for the render loop.
    let rafId = 0;
    let running = true;
    let startTime = performance.now();
    let speakingLerp = speakingRef.current ? 1 : 0;

    function frame(now: number) {
      if (!running || !gl || !program) return;
      const tSec = (now - startTime) / 1000;
      // Lerp speaking toward current ref value so transitions are smooth.
      const target = speakingRef.current ? 1 : 0;
      speakingLerp += (target - speakingLerp) * 0.12;
      gl.useProgram(program);
      gl.uniform1f(uTime, tSec);
      gl.uniform1f(uSpeaking, speakingLerp);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      rafId = requestAnimationFrame(frame);
    }
    rafId = requestAnimationFrame(frame);

    // Pause the loop when scrolled off-screen (perf win for gallery pages).
    const io =
      "IntersectionObserver" in window
        ? new IntersectionObserver(
            (entries) => {
              for (const e of entries) {
                if (e.isIntersecting) {
                  if (!running) {
                    running = true;
                    startTime = performance.now() - 1000; // avoid time jump
                    rafId = requestAnimationFrame(frame);
                  }
                } else {
                  running = false;
                  cancelAnimationFrame(rafId);
                }
              }
            },
            { rootMargin: "50px" }
          )
        : null;
    io?.observe(canvas);

    const onCtxLost = (e: Event) => {
      e.preventDefault();
      running = false;
      cancelAnimationFrame(rafId);
    };
    canvas.addEventListener("webglcontextlost", onCtxLost);

    return () => {
      running = false;
      cancelAnimationFrame(rafId);
      io?.disconnect();
      canvas.removeEventListener("webglcontextlost", onCtxLost);
      gl.deleteProgram(program);
      gl.deleteBuffer(vbo);
      gl.deleteVertexArray(vao);
    };
  }, [size, palette, numericSeed]);

  const cls = [
    "voice-sphere",
    speaking ? "voice-sphere--speaking" : "",
    onClick ? "cursor-pointer" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={cls}
      style={{ width: `${size}px`, height: `${size}px` }}
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
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block" }}
        aria-hidden
      />
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
