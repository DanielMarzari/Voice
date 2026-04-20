// Spike B — ZipVoice-Distill on ORT-Web + WebGPU
//
// Loads k2-fsa's pre-exported ZipVoice-Distill ONNX files (text_encoder +
// fm_decoder) via onnxruntime-web, runs them end-to-end with dummy tokens,
// and reports per-step + total wall clock.
//
// This is a feasibility spike — we're NOT checking audio quality, only
// that the pipeline loads + runs without crashing and at a reasonable
// speed. Real tokenization + vocoding comes in Phase 3.
//
// Signatures (from Spike A's parity_report_lightning.json):
//
//   text_encoder.onnx
//     inputs:  tokens (N,T) int64,
//              prompt_tokens (N,T) int64,
//              prompt_features_len scalar int64,
//              speed scalar float
//     output:  text_condition (N,T,concat_dim) float
//
//   fm_decoder.onnx
//     inputs:  t scalar float,
//              x (N,T,100) float,
//              text_condition (N,T,100) float,
//              speech_condition (N,T,100) float,
//              guidance_scale scalar float
//     output:  v (N,T,100) float
//
// Note: text_condition from text_encoder is (N,T,C) — but fm_decoder wants
// it at (N,T,100). We slice/project it here for the spike; real ZipVoice
// does an interpolation/expansion step we're not reproducing. This doesn't
// affect timing feasibility, which is what we care about.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/ort.webgpu.min.mjs";

const logEl = document.getElementById("log");
const capsEl = document.getElementById("caps");
const resultsEl = document.getElementById("results");
const resultsTableEl = document.getElementById("resultsTable");
const loadBtn = document.getElementById("loadBtn");
const runBtn = document.getElementById("runBtn");
const clearBtn = document.getElementById("clearBtn");
const precisionSel = document.getElementById("precision");
const epSel = document.getElementById("ep");
const stepsInput = document.getElementById("steps");
const seqlenInput = document.getElementById("seqlen");

const N_MEL = 100;       // ZipVoice mel dim
const NUM_TOKENS_VOCAB = 500; // tokens.txt has ~500 entries; any index < this is valid

let textEncoder = null;
let fmDecoder = null;
let loadedPrecision = null;
let loadedEp = null;

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
  resultsEl.classList.remove("visible");
}

// ---------- Capability detection ----------
async function detectCaps() {
  const hasWebGPU = "gpu" in navigator;
  let adapterInfo = null;
  if (hasWebGPU) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        const info = adapter.info || {};
        adapterInfo = info.description || info.vendor || info.architecture || "(adapter present)";
      } else {
        adapterInfo = "requestAdapter returned null";
      }
    } catch (e) {
      adapterInfo = `adapter request failed: ${e.message}`;
    }
  }
  const hasSAB = typeof SharedArrayBuffer !== "undefined";
  capsEl.innerHTML = `
    <b>WebGPU:</b> ${hasWebGPU ? "✓" : "✗"} &nbsp;
    <b>Adapter:</b> ${adapterInfo ?? "—"}<br>
    <b>SharedArrayBuffer:</b> ${hasSAB ? "✓" : "✗ (WASM threads disabled — add COOP/COEP to enable)"}
  `;
  if (!hasWebGPU) {
    capsEl.style.borderColor = "#ff5c5c";
    log("WebGPU not available — test with Chrome 125+ desktop.", "err");
  }
}
detectCaps();

// ---------- Model loading ----------
loadBtn.addEventListener("click", async () => {
  const prec = precisionSel.value;
  const ep = epSel.value;
  const suffix = prec === "int8" ? "_int8" : "";
  const teUrl = `./models/text_encoder${suffix}.onnx`;
  const fdUrl = `./models/fm_decoder${suffix}.onnx`;

  loadBtn.disabled = true;

  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";

  const providers = ep === "webgpu" ? ["webgpu"] : ["wasm"];
  const sessionOpts = {
    executionProviders: providers,
    graphOptimizationLevel: "all",
  };

  try {
    log(`Loading text_encoder${suffix}.onnx on ${ep}…`);
    const t0 = performance.now();
    textEncoder = await ort.InferenceSession.create(teUrl, sessionOpts);
    const teMs = performance.now() - t0;
    log(
      `  text_encoder loaded in ${teMs.toFixed(0)} ms — inputs: ${textEncoder.inputNames.join(", ")}`,
      "ok"
    );

    log(`Loading fm_decoder${suffix}.onnx on ${ep}…`);
    const t1 = performance.now();
    fmDecoder = await ort.InferenceSession.create(fdUrl, sessionOpts);
    const fdMs = performance.now() - t1;
    log(
      `  fm_decoder loaded in ${fdMs.toFixed(0)} ms — inputs: ${fmDecoder.inputNames.join(", ")}`,
      "ok"
    );

    loadedPrecision = prec;
    loadedEp = ep;
    runBtn.disabled = false;

    log(`Total load time: ${(teMs + fdMs).toFixed(0)} ms`, "ok");
  } catch (e) {
    log(`Load failed: ${e.message}`, "err");
    console.error(e);
    loadBtn.disabled = false;
  }
});

// ---------- Sampling ----------
async function runTextEncoder(seqLen) {
  // Build int64 tokens + prompt_tokens. Use small random values in [1, 99] to
  // exactly mirror Spike A's CPU run (which passed). A short prompt
  // (prompt_features_len=30, < seqLen=50) also mirrors Spike A — it looks
  // like `prompt_features_len` feeds an internal slice that must be < seqLen.
  const tokens = new BigInt64Array(seqLen);
  const promptTokens = new BigInt64Array(seqLen);
  for (let i = 0; i < seqLen; i++) {
    tokens[i] = BigInt(1 + Math.floor(Math.random() * 99));
    promptTokens[i] = BigInt(1 + Math.floor(Math.random() * 99));
  }
  const promptFeaturesLen = Math.min(30, Math.max(1, Math.floor(seqLen / 2)));

  const feeds = {
    tokens: new ort.Tensor("int64", tokens, [1, seqLen]),
    prompt_tokens: new ort.Tensor("int64", promptTokens, [1, seqLen]),
    prompt_features_len: new ort.Tensor("int64", new BigInt64Array([BigInt(promptFeaturesLen)]), []),
    speed: new ort.Tensor("float32", new Float32Array([1.0]), []),
  };

  log(
    `  text_encoder feeds: tokens=[1,${seqLen}] prompt_tokens=[1,${seqLen}] ` +
    `prompt_features_len=${promptFeaturesLen} speed=1.0`
  );

  const t0 = performance.now();
  let out;
  try {
    out = await textEncoder.run(feeds);
  } catch (e) {
    log(`  text_encoder.run FAILED: ${e.message}`, "err");
    throw e;
  }
  const dt = performance.now() - t0;

  const cond = out.text_condition || out[Object.keys(out)[0]];
  return { cond, ms: dt };
}

// Pad/slice text_condition to (N, T, N_MEL) so fm_decoder can consume it.
// Real ZipVoice does a length-regulation / interpolation step that we're
// not reproducing — this spike only cares about timing.
function coerceCondToMel(cond, seqLen) {
  // cond.dims might be [1, T, concat_dim]. We need [1, T, 100].
  const dims = cond.dims;
  const outerT = dims[1];
  const inDim = dims[2];
  const srcData = cond.data; // Float32Array

  const T = Math.min(outerT, seqLen);
  const outData = new Float32Array(1 * T * N_MEL);

  for (let t = 0; t < T; t++) {
    for (let c = 0; c < N_MEL; c++) {
      // Project last-dim slice to the first N_MEL channels (zero-pad if smaller).
      if (c < inDim) {
        outData[t * N_MEL + c] = srcData[t * inDim + c];
      }
    }
  }
  return new ort.Tensor("float32", outData, [1, T, N_MEL]);
}

async function runFmDecoderStep(xTensor, tVal, textCondTensor, speechCondTensor) {
  const feeds = {
    t: new ort.Tensor("float32", new Float32Array([tVal]), []),
    x: xTensor,
    text_condition: textCondTensor,
    speech_condition: speechCondTensor,
    guidance_scale: new ort.Tensor("float32", new Float32Array([1.0]), []),
  };
  const t0 = performance.now();
  const out = await fmDecoder.run(feeds);
  const dt = performance.now() - t0;
  const v = out.v || out[Object.keys(out)[0]];
  return { v, ms: dt };
}

// ---------- Run button ----------
runBtn.addEventListener("click", async () => {
  if (!textEncoder || !fmDecoder) {
    log("Load models first.", "err");
    return;
  }
  const steps = Math.max(1, parseInt(stepsInput.value, 10) || 4);
  const seqLen = Math.max(10, parseInt(seqlenInput.value, 10) || 50);

  runBtn.disabled = true;
  log(`Running end-to-end: seqLen=${seqLen} frames, ${steps} NFE steps…`);

  try {
    const wholeStart = performance.now();

    // 1) text_encoder
    const { cond, ms: teMs } = await runTextEncoder(seqLen);
    log(`  text_encoder: ${teMs.toFixed(0)} ms (output shape [${cond.dims.join(", ")}])`);
    const textCondMel = coerceCondToMel(cond, seqLen);
    const effectiveT = textCondMel.dims[1];

    // 2) init noise x + dummy speech_condition (voice prompt mel)
    const n = 1 * effectiveT * N_MEL;
    let xData = new Float32Array(n);
    for (let i = 0; i < n; i++) xData[i] = (Math.random() - 0.5) * 2;
    let xTensor = new ort.Tensor("float32", xData, [1, effectiveT, N_MEL]);

    const speechData = new Float32Array(n);
    // leave as zeros — zero-shot voice prompt is just a mel-shaped placeholder here
    const speechCondTensor = new ort.Tensor("float32", speechData, [1, effectiveT, N_MEL]);

    // 3) fm_decoder × steps with Euler integration
    const stepTimes = [];
    for (let s = 0; s < steps; s++) {
      const t = s / steps; // flow time 0→1
      const { v, ms } = await runFmDecoderStep(xTensor, t, textCondMel, speechCondTensor);
      stepTimes.push(ms);
      log(`  fm_decoder step ${s + 1}/${steps}: ${ms.toFixed(0)} ms`);

      // Euler: x_{t+dt} = x_t + dt · v
      const dt = 1 / steps;
      const vData = v.data;
      const newX = new Float32Array(xData.length);
      for (let i = 0; i < xData.length; i++) newX[i] = xData[i] + dt * vData[i];
      xData = newX;
      xTensor = new ort.Tensor("float32", xData, [1, effectiveT, N_MEL]);
    }

    const totalMs = performance.now() - wholeStart;
    const fmTotal = stepTimes.reduce((a, b) => a + b, 0);
    const fmAvg = fmTotal / steps;

    // Approx "audio length" — ZipVoice uses 24 kHz with hop 256 → 50 frames ≈ 0.53s.
    const audioSec = (effectiveT * 256) / 24000;
    const rtFactor = audioSec / (totalMs / 1000);

    log(``, "ok");
    log(`✓ End-to-end complete: ${totalMs.toFixed(0)} ms total`, "ok");
    log(`  text_encoder: ${teMs.toFixed(0)} ms`, "ok");
    log(`  fm_decoder:   ${fmTotal.toFixed(0)} ms (${fmAvg.toFixed(0)} ms / step avg)`, "ok");
    log(`  ≈ ${audioSec.toFixed(2)}s audio → ${rtFactor.toFixed(2)}× real-time on ${loadedEp}`, "ok");

    // Populate summary table for easy screenshotting
    resultsTableEl.innerHTML = "";
    const rows = [
      ["Precision", loadedPrecision.toUpperCase()],
      ["Execution provider", loadedEp],
      ["Sequence length", `${effectiveT} frames`],
      ["NFE steps", `${steps}`],
      ["text_encoder", `${teMs.toFixed(0)} ms`],
      ["fm_decoder total", `${fmTotal.toFixed(0)} ms`],
      ["fm_decoder / step", `${fmAvg.toFixed(0)} ms`],
      ["End-to-end", `${totalMs.toFixed(0)} ms`],
      ["Audio / wall-clock", `${rtFactor.toFixed(2)}× real-time`],
    ];
    for (const [k, v] of rows) {
      const tr = document.createElement("tr");
      const tdK = document.createElement("td"); tdK.textContent = k;
      const tdV = document.createElement("td"); tdV.textContent = v;
      tr.appendChild(tdK); tr.appendChild(tdV);
      resultsTableEl.appendChild(tr);
    }
    resultsEl.classList.add("visible");

    // Stash in window for easy copy via DevTools / automation
    window.__SPIKE_B_RESULT__ = {
      precision: loadedPrecision,
      ep: loadedEp,
      seqLen: effectiveT,
      steps,
      textEncoderMs: teMs,
      fmDecoderTotalMs: fmTotal,
      fmDecoderAvgMs: fmAvg,
      totalMs,
      audioSec,
      realTimeFactor: rtFactor,
      stepTimes,
    };
  } catch (e) {
    log(`Run failed: ${e.message}`, "err");
    console.error(e);
  } finally {
    runBtn.disabled = false;
  }
});

clearBtn.addEventListener("click", clearLog);
