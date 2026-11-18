// PDF layout parsing in the browser: pdf.js renders the page and reads its text layer,
// D-FINE nano (ONNX, onnxruntime-web on WebGPU or WASM) finds the layout regions.
// Same post-processing as parse_pdf.py: drop overlapping boxes, column-aware reading order, markdown by class.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
import * as pdfjs from "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.min.mjs";

pdfjs.GlobalWorkerOptions.workerSrc = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.5.438/build/pdf.worker.min.mjs";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

const S = 1024;              // model input size (page is resized to S x S, like in training)
const RENDER_SCALE = 2;      // pdf.js render scale (144 DPI, same as parse_pdf.py)
const COLORS = { Caption: "#e6194b", Footnote: "#911eb4", Formula: "#f58231", "List-item": "#3cb44b",
  "Page-footer": "#808080", "Page-header": "#808080", Picture: "#4363d8", "Section-header": "#f032e6",
  Table: "#ffe119", Text: "#42d4f4", Title: "#000075" };

const $ = (id) => document.getElementById(id);
const status = (t) => ($("status").textContent = t);
let session, classes, backend, doc, pageNo = 1, last = null;

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

// ------------------------------------------------------------------ model
async function loadModel() {
  classes = await fetch("/r2/pdf-parser/models/classes.json").then((r) => r.json());
  const bytes = await fetchBytes("/r2/pdf-parser/models/layout_fp16.onnx", (got, total) => {
    status(`Downloading the layout model: ${mb(got)}${total ? ` / ${mb(total)}` : ""} MB`);
    progress(total ? got / total : null);
  });
  status("Starting the model…"); progress(null);
  const forceWasm = new URLSearchParams(location.search).has("wasm");  // ?wasm to test the CPU path
  for (const ep of (navigator.gpu && !forceWasm ? ["webgpu", "wasm"] : ["wasm"])) {
    try {
      const t0 = performance.now();
      session = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });
      backend = ep;
      loaded();
      status(`Model ready (${ep}, ${((performance.now() - t0) / 1000).toFixed(1)}s). Open a PDF or try a sample.`);
      return;
    } catch (e) { console.warn(`${ep} failed`, e); }
  }
  loaded();
  status("Could not load the model in this browser.");
}

function toTensor(canvas) {  // page canvas -> [1,3,S,S] float32, RGB in 0-1 (no normalisation, like training)
  const c = new OffscreenCanvas(S, S), g = c.getContext("2d");
  g.drawImage(canvas, 0, 0, S, S);
  const px = g.getImageData(0, 0, S, S).data, n = S * S, x = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) { x[i] = px[4 * i] / 255; x[n + i] = px[4 * i + 1] / 255; x[2 * n + i] = px[4 * i + 2] / 255; }
  return new ort.Tensor("float32", x, [1, 3, S, S]);
}

async function detect(canvas) {  // -> [{type, box:[x0,y0,x1,y1] in canvas px, conf}] for every query
  const out = await session.run({ pixel_values: toTensor(canvas) });
  const L = out.logits.data, B = out.pred_boxes.data, C = classes.length, Q = L.length / C;
  const W = canvas.width, H = canvas.height, regs = [];
  for (let q = 0; q < Q; q++) {
    let best = -Infinity, c = 0;
    for (let k = 0; k < C; k++) if (L[q * C + k] > best) { best = L[q * C + k]; c = k; }
    const [cx, cy, w, h] = B.subarray(4 * q, 4 * q + 4);
    regs.push({ type: classes[c], conf: 1 / (1 + Math.exp(-best)),
      box: [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H] });
  }
  return regs;
}

// ------------------------------------------------------------------ layout post-processing (ported from parse_pdf.py)
function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = ix * iy, ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua ? inter / ua : 0;
}

function dedupe(regs, thr = 0.6) {
  const keep = [];
  for (const r of [...regs].sort((a, b) => b.conf - a.conf)) if (keep.every((k) => iou(r.box, k.box) < thr)) keep.push(r);
  return keep;
}

function readingOrder(regs, pageW) {  // full-width blocks split the page into bands; left column before right in a band
  const mid = pageW / 2, out = [], band = [];
  const isFull = (r) => r.box[0] < mid - pageW * 0.05 && r.box[2] > mid + pageW * 0.05;
  const flush = () => {
    const left = band.filter((r) => (r.box[0] + r.box[2]) / 2 < mid), right = band.filter((r) => !left.includes(r));
    out.push(...left.sort((a, b) => a.box[1] - b.box[1]), ...right.sort((a, b) => a.box[1] - b.box[1]));
    band.length = 0;
  };
  for (const r of [...regs].sort((a, b) => a.box[1] - b.box[1])) { if (isFull(r)) { flush(); out.push(r); } else band.push(r); }
  flush();
  return out;
}

// ------------------------------------------------------------------ text layer
function textItems(content, viewport) {  // pdf.js text items -> boxes in canvas px
  return content.items.filter((it) => it.str !== undefined).map((it) => {
    const [x, y] = viewport.convertToViewportPoint(it.transform[4], it.transform[5]);  // baseline start
    const h = Math.hypot(it.transform[2], it.transform[3]) * viewport.scale, w = it.width * viewport.scale;
    return { str: it.str, eol: it.hasEOL, box: [x, y - h, x + w, y] };
  });
}

function textIn(box, items) {  // items whose centre is inside the box, joined like parse_pdf.py's text_from_layer
  const inside = items.filter((t) => { const cx = (t.box[0] + t.box[2]) / 2, cy = (t.box[1] + t.box[3]) / 2;
    return t.str.trim() && cx >= box[0] && cx <= box[2] && cy >= box[1] && cy <= box[3]; });
  let out = "";
  inside.forEach((t, i) => {
    out += t.str;
    const nxt = inside[i + 1];
    if (!nxt) return;
    const lineBreak = t.eol || Math.abs(nxt.box[3] - t.box[3]) > (t.box[3] - t.box[1]) * 0.5;
    if (lineBreak && t.str.endsWith("-") && t.str.length > 1 && /^[a-z]/.test(nxt.str)) out = out.slice(0, -1);  // conver- + gence
    else if (!(lineBreak && t.str.endsWith("-")) && !out.endsWith(" ")) out += " ";
  });
  return out.replace(/\s+/g, " ").trim();
}

function toMarkdown(r) {
  const c = r.content;
  switch (r.type) {
    case "Page-header": case "Page-footer": return null;
    case "Title": return `# ${c}`;
    case "Section-header": return `## ${c}`;
    case "List-item": return "- " + c.replace(/^[•·▪◦\-*]\s*/, "");
    case "Formula": return c ? `$$\n${c}\n$$` : "*[formula]*";
    case "Picture": return "*[figure]*";
    case "Table": return c ? `*[table]*\n${c}` : "*[table]*";
    case "Caption": return `*${c}*`;
    case "Footnote": return `<sub>${c}</sub>`;
    default: return c;
  }
}

// ------------------------------------------------------------------ page
async function showPage(n) {
  pageNo = n;
  $("pageinfo").textContent = `page ${n} / ${doc.numPages}`;
  $("prev").disabled = n <= 1; $("next").disabled = n >= doc.numPages;
  const page = await doc.getPage(n), viewport = page.getViewport({ scale: RENDER_SCALE });
  const canvas = $("page");
  canvas.width = viewport.width; canvas.height = viewport.height;
  await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
  const pageImg = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height);
  const t0 = performance.now();
  const all = await detect(canvas);
  const ms = performance.now() - t0;
  const items = textItems(await page.getTextContent(), viewport);
  last = { all, items, pageImg, ms, w: canvas.width };
  render();
}

function render() {
  const { all, items, pageImg, ms, w } = last, thr = +$("thr").value;
  const regs = readingOrder(dedupe(all.filter((r) => r.conf >= thr)), w);
  regs.forEach((r) => (r.content = ["Picture"].includes(r.type) ? "" : textIn(r.box, items)));
  const g = $("page").getContext("2d");
  g.putImageData(pageImg, 0, 0);
  g.lineWidth = 3; g.font = "bold 20px system-ui";
  regs.forEach((r, i) => {
    const [x0, y0, x1, y1] = r.box, col = COLORS[r.type] || "#fff";
    g.strokeStyle = col; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
    const label = `${i + 1} ${r.type} ${r.conf.toFixed(2)}`, tw = g.measureText(label).width;
    g.fillStyle = col; g.fillRect(x0, Math.max(0, y0 - 24), tw + 10, 24);
    g.fillStyle = "#fff"; g.fillText(label, x0 + 5, Math.max(18, y0 - 6));
  });
  const md = regs.map(toMarkdown).filter((m) => m).join("\n\n");
  $("md").textContent = md || "(no text layer on this page: scanned PDFs need OCR, which this demo doesn't include)";
  $("copy").disabled = !md;
  const noText = !items.some((t) => t.str.trim());
  status(`${regs.length} regions in ${ms.toFixed(0)} ms (${backend})${noText ? " · no text layer on this page" : ""}`);
  $("legend").innerHTML = Object.entries(COLORS).map(([k, v]) => `<span><i style="background:${v}"></i>${k}</span>`).join("");
}

async function openPdf(data) {
  if (!session) return;
  doc = await pdfjs.getDocument({ data }).promise;
  await showPage(1);
}

$("file").onchange = async (e) => { const f = e.target.files[0]; if (f) await openPdf(new Uint8Array(await f.arrayBuffer())); };
$("sample").onclick = async () => openPdf(new Uint8Array(await fetch("/r2/pdf-parser/samples/sample.pdf").then((r) => r.arrayBuffer())));
$("prev").onclick = () => showPage(pageNo - 1);
$("next").onclick = () => showPage(pageNo + 1);
$("thr").oninput = () => { $("thrv").textContent = (+$("thr").value).toFixed(2); if (last) render(); };
$("copy").onclick = () => navigator.clipboard.writeText($("md").textContent);
loadModel().catch((e) => { loaded(); status("Failed to load: " + e.message); console.error(e); });
