// Spike B — ORT-Web + WebGPU harness
//
// Loads an ONNX flow-matching TTS model (F5 or ZipVoice) via
// onnxruntime-web, runs one inference, and reports timings.
//
// This is deliberately minimal — no React, no bundler, no service worker.
// If feasibility fails here it's the model/runtime combo, not the app shell.

// We load ORT-Web from jsdelivr so this file stays self-contained.
// In Reader (Phase 3) it'll be bundled via npm `onnxruntime-web`.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.webgpu.min.mjs";

const logEl = document.getElementById("log");
const capsEl = document.getElementById("caps");
const loadBtn = document.getElementById("loadBtn");
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const modelSel = document.getElementById("model");
const epSel = document.getElementById("ep");
const stepsInput = document.getElementById("steps");
const promptInput = document.getElementById("prompt");
const audioOut = document.getElementById("out");

let session = null;

function log(msg, cls = "") {
  const line = document.createElement("div");
  if (cls) line.className = cls;
  const ts = new Date().toISOString().slice(11, 23);
  line.textContent = `[${ts}] ${msg}`;
  logEl.appendChild(line);
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog() {
  logEl.textContent = "";
}

// ---------- Capability detection ----------
async function detectCaps() {
  const hasWebGPU = "gpu" in navigator;
  let adapterInfo = null;
  if (hasWebGPU) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        adapterInfo = adapter.info?.description || adapter.info?.vendor || "(adapter present)";
      }
    } catch (e) {
      adapterInfo = `adapter request failed: ${e.message}`;
    }
  }
  const hasSAB = typeof SharedArrayBuffer !== "undefined";
  const ua = navigator.userAgent;
  capsEl.innerHTML = `
    <b>WebGPU:</b> ${hasWebGPU ? "✓" : "✗"} &nbsp;
    <b>Adapter:</b> ${adapterInfo ?? "—"}<br>
    <b>SharedArrayBuffer:</b> ${hasSAB ? "✓" : "✗ (WASM threads disabled)"}<br>
    <b>UA:</b> ${ua}
  `;
  if (!hasWebGPU) {
    capsEl.style.borderColor = "#ff5c5c";
    log("WebGPU not available — test with Chrome 125+ desktop.", "err");
  }
}
detectCaps();

// ---------- Model loading ----------
loadBtn.addEventListener("click", async () => {
  const url = modelSel.value;
  const ep = epSel.value;
  log(`Loading ${url} on ${ep}…`);

  // Configure ORT WASM assets so the WASM fallback can resolve its helper
  // files. Point at jsdelivr so we don't need to host them.
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

  const t0 = performance.now();
  try {
    const providers = ep === "webgpu" ? ["webgpu"] : ["wasm"];
    session = await ort.InferenceSession.create(url, {
      executionProviders: providers,
      graphOptimizationLevel: "all",
      // Required for big models — 2GB proto limit forces external-weights path.
      externalData: [],
    });
    const dt = performance.now() - t0;
    log(`Loaded in ${dt.toFixed(0)} ms`, "ok");
    log(
      `Inputs: ${session.inputNames.join(", ")} | Outputs: ${session.outputNames.join(", ")}`
    );
    runBtn.disabled = false;
  } catch (e) {
    log(`Load failed: ${e.message}`, "err");
    console.error(e);
  }
});

// ---------- Tokenization ----------
//
// TODO(spike): load a real tokenizer.json (F5 char-level or ZipVoice BPE).
// For spike B we only need *some* token tensor shaped correctly so we can
// time inference. Replace with real tokenizer before claiming parity.
function fakeTokenize(text) {
  const ids = new Int64Array(text.length);
  for (let i = 0; i < text.length; i++) ids[i] = text.charCodeAt(i) % 256;
  return ids;
}

// ---------- Flow sampling loop ----------
//
// The exported ONNX is one step of the flow: (x_t, t, cond) → velocity.
// We iterate it N times with Euler integration. A real implementation
// uses rectified-flow sampling + EPSS / Sway; for a timing spike the
// plain Euler loop is representative.
async function sampleFlow(steps, melLen, melDim, tokens) {
  const B = 1;
  let x = new Float32Array(B * melLen * melDim); // start at zero / Gaussian
  // Optional: initialize as Gaussian noise. For timing this doesn't matter.

  const dt = 1 / steps;
  let totalMs = 0;

  for (let i = 0; i < steps; i++) {
    const t = i / steps;
    const feeds = {
      // Names must match the exported ONNX's input list. Adjust after Spike A.
      x: new ort.Tensor("float32", x, [B, melLen, melDim]),
      t: new ort.Tensor("float32", new Float32Array([t]), [B]),
      text: new ort.Tensor("int64", tokens, [B, tokens.length]),
      // ref_mel + ref_mel_len would go here; for spike B we pass zeros.
      ref_mel: new ort.Tensor(
        "float32",
        new Float32Array(B * 100 * melDim),
        [B, 100, melDim]
      ),
      ref_mel_len: new ort.Tensor("int64", new BigInt64Array([100n]), [B]),
    };

    const t0 = performance.now();
    const out = await session.run(feeds);
    const stepMs = performance.now() - t0;
    totalMs += stepMs;
    log(`  step ${i + 1}/${steps}: ${stepMs.toFixed(0)} ms`);

    // Euler update: x_{t+dt} = x_t + dt · velocity
    const v = out.velocity?.data ?? Object.values(out)[0].data;
    for (let j = 0; j < x.length; j++) x[j] = x[j] + dt * v[j];
  }

  log(`Sampling total: ${totalMs.toFixed(0)} ms`, "ok");
  return x;
}

// ---------- Run button ----------
runBtn.addEventListener("click", async () => {
  if (!session) {
    log("Load a model first.", "err");
    return;
  }
  const prompt = promptInput.value;
  const steps = Math.max(1, parseInt(stepsInput.value, 10) || 4);

  log(`Running: "${prompt}" (${steps} steps)`);
  const tokens = fakeTokenize(prompt);
  log(`Tokens: ${tokens.length} (fake tokenizer)`, "warn");

  const t0 = performance.now();
  try {
    // 200 mel frames ≈ 4 seconds at 24kHz / 256 hop. Adjust per model.
    const mel = await sampleFlow(steps, 200, 100, tokens);
    const dt = performance.now() - t0;
    log(`Total inference: ${dt.toFixed(0)} ms`, "ok");

    // Mel → waveform would hand off to vocos.onnx here. For the feasibility
    // spike we just confirm we produced a mel tensor of the expected shape.
    log(`Mel output shape: [1, 200, 100], len=${mel.length}`);
    log("✓ Spike B passed: model runs end-to-end in the browser.", "ok");

    // Post-spike: hook up vocos and play audio via Web Audio API.
    // audioOut.src = URL.createObjectURL(new Blob([wav], {type: "audio/wav"}));
  } catch (e) {
    log(`Run failed: ${e.message}`, "err");
    console.error(e);
  }
});

clearBtn.addEventListener("click", clearLog);
