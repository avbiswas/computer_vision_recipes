// Play inside the dream: the world model predicts the next 16x16 token grid from past frames + your key,
// the VQ-VAE decoder turns tokens into a 64x64 image. Mirrors the replay loop in export_onnx.py
// (checked there to match PyTorch exactly).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const R2 = "/r2/world-model/";  // model files live in R2
const OFFSETS = [32, 16, 8, 4, 3, 2, 1, 0];  // context frames: t-32, t-16, ..., t-1, and the target t
const NO_ACTION = 5;                          // "no action between these two frames" token
const TOK = 256;                              // 16 x 16 tokens per frame
const KEYS = { ArrowUp: 1, ArrowDown: 2, ArrowLeft: 3, ArrowRight: 4 };  // env.py: forward, backward, turn left, turn right
const MAX_FPS = 15;

const $ = (id) => document.getElementById(id);
const status = (t) => ($("wm-status").textContent = t);
const i64 = (arr, dims) => new ort.Tensor("int64", BigInt64Array.from(arr, (v) => BigInt(v)), dims);
const f32 = (arr, dims) => new ort.Tensor("float32", Float32Array.from(arr), dims);

let wm, dec, rooms, backend;
let hist = [], acts = [], dropped = 0, running = false, frames = 0;
const held = new Set();

// ------------------------------------------------------------------ loading UI (spinner + progress bar, see site.css .loader)
const progress = (frac) => {  // frac in 0..1, or null for "busy, size unknown"
  const i = document.getElementById("bar");
  if (frac == null) { i.classList.remove("det"); i.style.width = ""; } else { i.classList.add("det"); i.style.width = `${(Math.min(1, frac) * 100).toFixed(1)}%`; }
};
const loaded = () => document.getElementById("loader").classList.add("done");
// fetch a file as bytes, reporting progress as (bytes so far, total bytes or 0)
async function fetchBytes(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get("content-length") || 0, reader = r.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}
const mb = (n) => (n / 1e6).toFixed(0);

async function load() {
  rooms = await fetch("rooms.json").then((r) => r.json());
  // download both files first (with progress), then start the sessions
  const sizes = [39.3e6, 5.9e6], files = {};  // approximate, for a smooth bar across the two files
  let done = 0;
  for (const [i, f] of ["world_model.onnx", "decoder.onnx"].entries()) {
    files[f] = await fetchBytes(R2 + f, (got) => {
      const all = sizes[0] + sizes[1];
      status(`Downloading the model: ${mb(done + got)} / ${mb(all)} MB (only the first visit)`);
      progress((done + got) / all);
    });
    done += sizes[i];
  }
  status("Starting the model…"); progress(null);
  const forceWasm = new URLSearchParams(location.search).has("wasm");
  for (const ep of navigator.gpu && !forceWasm ? ["webgpu", "wasm"] : ["wasm"]) {
    try {
      const t0 = performance.now();
      // one at a time: onnxruntime-web can't create two WebGPU sessions in parallel
      wm = await ort.InferenceSession.create(files["world_model.onnx"], { executionProviders: [ep] });
      dec = await ort.InferenceSession.create(files["decoder.onnx"], { executionProviders: [ep] });
      backend = ep;
      newRoom(0);
      await step(0);  // warm-up run (compiles shaders on WebGPU)
      loaded();
      status(`Ready (${ep}, loaded in ${((performance.now() - t0) / 1000).toFixed(1)} s). Click the picture, then use the arrow keys.`);
      $("wm-play").disabled = $("wm-new").disabled = false;
      return;
    } catch (e) { console.warn(`${ep} failed`, e); }
  }
  loaded();
  status("Could not run the model in this browser.");
}

function newRoom(i) {
  const r = rooms[i ?? Math.floor(Math.random() * rooms.length)];
  hist = r.hist.map((h) => Int32Array.from(h));
  acts = [...r.acts];
  frames = 0; dropped = 0;
}

async function step(action) {
  acts.push(action);
  const t = hist.length;
  const idx = OFFSETS.map((o) => Math.max(t - o, 0));
  const fa = idx.map((i, j) => (j > 0 && i >= 1 && idx[j - 1] === i - 1 ? acts[i - 1] : NO_ACTION));
  const ctx = new Int32Array(7 * TOK);
  idx.slice(0, -1).forEach((i, k) => ctx.set(hist[i], k * TOK));
  const out = await wm.run({
    ctx: i64(ctx, [1, 7, TOK]),
    fa: i64(fa, [1, 8]),
    times: f32(idx.map((i) => i - t), [1, 8]),
    anchor: i64(hist[0], [1, TOK]),
    anchor_time: f32([-Math.min(t + dropped, 250)], [1]),  // true distance to the anchor frame
  });
  const next = Int32Array.from(out.next.data, Number);
  hist.push(next);
  if (hist.length > 64) { hist.splice(1, 1); acts.splice(0, 1); dropped++; }  // keep the anchor (frame 0) + the last 63 frames (context needs 33)
  const { rgba } = await dec.run({ ids: i64(next, [1, TOK]) });
  draw(new Uint8ClampedArray(rgba.data));
  frames++;
}

const small = new OffscreenCanvas(64, 64);
function draw(px) {
  small.getContext("2d").putImageData(new ImageData(px, 64, 64), 0, 0);
  const c = $("wm-screen"), g = c.getContext("2d");
  g.imageSmoothingEnabled = false;
  g.drawImage(small, 0, 0, c.width, c.height);
}

function currentAction() {
  for (const [k, a] of Object.entries(KEYS)) if (held.has(k)) return a;
  return 0;
}

async function loop() {
  let last = performance.now(), times = [];
  while (running) {
    const t0 = performance.now();
    const a = currentAction();
    await step(a);
    times.push(performance.now() - t0); if (times.length > 20) times.shift();
    const ms = times.reduce((s, x) => s + x, 0) / times.length;
    $("wm-info").textContent = `${backend} · ${ms.toFixed(0)} ms per frame · ${frames} dreamed frames · key: ${["none", "forward", "backward", "left", "right"][a]}`;
    const wait = 1000 / MAX_FPS - (performance.now() - last);
    await new Promise((r) => setTimeout(r, Math.max(0, wait)));
    last = performance.now();
  }
}

function setRunning(on) {
  running = on;
  $("wm-play").textContent = on ? "Pause" : "Play";
  if (on) loop();
}

$("wm-play").onclick = () => setRunning(!running);
$("wm-new").onclick = async () => { const was = running; setRunning(false); await new Promise((r) => setTimeout(r, 120)); newRoom(); await step(0); if (was) setRunning(true); };

const screen = $("wm-screen");
screen.tabIndex = 0;
screen.addEventListener("keydown", (e) => { if (e.key in KEYS) { held.add(e.key); e.preventDefault(); if (!running && wm) setRunning(true); } });
screen.addEventListener("keyup", (e) => held.delete(e.key));
screen.addEventListener("blur", () => held.clear());
screen.addEventListener("click", () => screen.focus());

// on-screen keys for touch screens
document.querySelectorAll("[data-key]").forEach((b) => {
  const k = b.dataset.key;
  const down = (e) => { e.preventDefault(); held.add(k); if (!running && wm) setRunning(true); };
  const up = () => held.delete(k);
  b.addEventListener("pointerdown", down); b.addEventListener("pointerup", up); b.addEventListener("pointerleave", up); b.addEventListener("pointercancel", up);
});

load().catch((e) => { loaded(); status("Failed to load: " + e.message); console.error(e); });
