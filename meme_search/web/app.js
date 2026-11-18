// Meme search in the browser: fine-tuned CLIP text encoder (transformers.js) vs precomputed template embeddings.
// Everything runs in the tab: no server calls except fetching static files.
import { AutoTokenizer, CLIPTextModelWithProjection, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "/r2/meme-search/";  // model files are in R2 (over the 25 MiB static-asset limit)

const K = 6;          // results shown
const DUP_COS = 0.95; // hide near-identical templates (reposts)
const STOP = new Set(("a an the and or but of to in on at for with my me i im i'm is are was were be been it its this that so " +
  "when you your we our they their he she his her just about after before from by as do does did have has had").split(" "));

const $ = (id) => document.getElementById(id);
const status = (t) => ($("status").textContent = t);

let tok, model, emb, meta, D = 512;

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
  const t0 = performance.now();
  [tok, model, meta, emb] = await Promise.all([
    AutoTokenizer.from_pretrained("meme-clip"),
    CLIPTextModelWithProjection.from_pretrained("meme-clip", { dtype: "fp16", progress_callback: (p) => {
      if (p.status === "progress" && p.file.endsWith(".onnx")) {
        status(`Downloading the text encoder: ${mb(p.loaded)} / ${mb(p.total)} MB (only the first visit)`);
        progress(p.loaded / p.total);
      } else if (p.status === "done" && p.file.endsWith(".onnx")) { status("Starting the text encoder…"); progress(null); }
    } }),
    fetch("/r2/meme-search/data/templates.json").then((r) => r.json()),
    fetch("/r2/meme-search/data/embeddings.bin").then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)),
  ]);
  D = emb.length / meta.length;
  loaded();
  status(`Ready: ${meta.length} templates, loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s.`);
  $("go").disabled = false;
}

async function embed(texts) {
  const inputs = tok(texts, { padding: true, truncation: true });
  const { text_embeds } = await model(inputs);
  const out = [];
  for (let i = 0; i < texts.length; i++) {
    const v = text_embeds.data.slice(i * D, (i + 1) * D);
    let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
    out.push(v.map((x) => x / n));
  }
  return out;
}

const dotRow = (q, j) => { let s = 0; const o = j * D; for (let k = 0; k < D; k++) s += q[k] * emb[o + k]; return s; };

function topK(q, k) {
  const scores = new Float32Array(meta.length);
  for (let j = 0; j < meta.length; j++) scores[j] = dotRow(q, j);
  const order = [...scores.keys()].sort((a, b) => scores[b] - scores[a]);
  const picked = [];
  for (const j of order) {
    if (picked.length === k) break;
    const dup = picked.some((p) => { let s = 0; for (let d = 0; d < D; d++) s += emb[p * D + d] * emb[j * D + d]; return s > DUP_COS; });
    if (!dup) picked.push(j);
  }
  return picked.map((j) => ({ j, score: scores[j] }));
}

// Leave-one-out: how much does the match drop when each word is removed?
// vs = embeddings of [query, query-without-word-0, query-without-word-1, ...] (computed once per search)
function wordImportance(vs, words, j) {
  const base = dotRow(vs[0], j);
  return words.map((w, i) => ({ w, d: base - dotRow(vs[i + 1], j) }));
}

const FONT = (size) => `bold ${size}px Impact, "Arial Black", sans-serif`;
const measureCtx = document.createElement("canvas").getContext("2d");

// Largest font that fits the slot box; "readable" = fits without shrinking to tiny text.
function fitText(ctx, text, s) {
  const [x0, y0, x1, y1] = s.bbox, w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
  let size = Math.max(10, Math.min(s.line_h * 1.2, h)), lines, fits;
  for (;;) {
    ctx.font = FONT(size);
    lines = wrap(ctx, text.toUpperCase(), w);
    fits = lines.every((l) => ctx.measureText(l).width <= w) && lines.length * size * 1.15 <= Math.max(h, s.line_h * 1.3);
    if (fits || size <= 10) break;
    size -= 2;
  }
  return { size, lines, readable: fits && size >= Math.max(12, 0.5 * s.line_h) };
}

// One box: the full query if it fits; else drop stopwords, then the least important words (CLIP leave-one-out).
function shortenToFit(query, slot, imp) {
  const words = query.split(/\s+/).filter(Boolean);
  const keep = words.map((_, i) => i);
  const isStop = (i) => STOP.has(words[i].toLowerCase().replace(/\W/g, ""));
  const order = [...keep].sort((a, b) => (isStop(b) - isStop(a)) || ((imp[a]?.d ?? 0) - (imp[b]?.d ?? 0)));
  for (const drop of [null, ...order]) {
    if (drop !== null) keep.splice(keep.indexOf(drop), 1);
    const text = keep.map((i) => words[i]).join(" ");
    if (!keep.length || fitText(measureCtx, text, slot).readable) return text;
  }
  return query;
}

// Multi-box templates: main (largest) box = query, other boxes = the most important keywords.
function keywordFill(query, slots, imp) {
  const area = (s) => (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]);
  const main = slots.reduce((m, s, i) => (area(s) > area(slots[m]) ? i : m), 0);
  const ranked = [...imp].sort((a, b) => b.d - a.d).map((x) => x.w.replace(/[.,!?]/g, ""))
    .filter((w) => w && !STOP.has(w.toLowerCase().replace(/\W/g, "")));
  let k = 0;
  return slots.map((s, i) => (i === main ? shortenToFit(query, s, imp) : ranked[k++] ?? ""));
}

function wrap(ctx, text, width) {
  const lines = []; let cur = "";
  for (const w of text.split(/\s+/).filter(Boolean)) {
    const t = cur ? cur + " " + w : w;
    if (ctx.measureText(t).width <= width || !cur) cur = t; else { lines.push(cur); cur = w; }
  }
  return cur ? [...lines, cur] : lines;
}

function draw(canvas, img, slots, texts) {
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  slots.forEach((s, i) => {
    const t = texts[i] || ""; if (!t) return;
    const [x0, y0, x1, y1] = s.bbox, w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
    const { size, lines } = fitText(ctx, t, s);
    ctx.font = FONT(size); ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.lineJoin = "round";
    let y = y0 + (h - lines.length * size * 1.15) / 2;
    for (const l of lines) {
      if (!s.light) { ctx.lineWidth = Math.max(2, size / 7); ctx.strokeStyle = "#000"; ctx.strokeText(l, x0 + w / 2, y); }
      ctx.fillStyle = s.light ? "#000" : "#fff"; ctx.fillText(l, x0 + w / 2, y);
      y += size * 1.15;
    }
  });
}

function card({ j, score }, query, imp) {
  const m = meta[j];
  // no detected text: one caption box across the top
  const slots = m.slots.length ? m.slots : [{ bbox: [m.w * 0.05, m.h * 0.02, m.w * 0.95, m.h * 0.2], line_h: m.h * 0.08, light: false }];
  const multi = slots.length > 1;
  const texts = multi ? keywordFill(query, slots, imp) : [shortenToFit(query, slots[0], imp)];
  const el = document.createElement("div"); el.className = "card";
  const canvas = document.createElement("canvas"); canvas.width = m.w; canvas.height = m.h;
  const img = new Image(); img.src = `/r2/meme-search/data/templates/${m.file}`;
  const redraw = () => img.complete && draw(canvas, img, slots, texts);
  img.onload = redraw;
  el.append(canvas);
  const top = [...imp].filter((x) => x.d > 0 && !STOP.has(x.w.toLowerCase().replace(/\W/g, ""))).sort((a, b) => b.d - a.d).slice(0, 4);
  el.insertAdjacentHTML("beforeend", `<div class="meta">match ${score.toFixed(3)} · ${slots.length} text box(es) ·
    <span class="how">${multi ? "keywords in the small boxes" : texts[0] === query ? "full query" : "shortened to fit"}</span></div>
    <div class="chips">${top.map((x) => `<span>${x.w} +${x.d.toFixed(3)}</span>`).join("")}</div>`);
  const inputs = slots.map((_, i) => {
    const row = document.createElement("div"); row.className = "slot";
    row.innerHTML = `<label>box ${i + 1}</label>`;
    const inp = document.createElement("input"); inp.value = texts[i];
    inp.oninput = () => { texts[i] = inp.value; inp.dataset.edited = "1"; redraw(); };
    row.append(inp); el.append(row);
    return inp;
  });
  const row = document.createElement("div"); row.className = "row";
  const dl = document.createElement("button"); dl.textContent = "Download PNG";
  dl.onclick = () => { const a = document.createElement("a"); a.download = `meme_${m.file.replace(".webp", "")}.png`; a.href = canvas.toDataURL("image/png"); a.click(); };
  const clr = document.createElement("button"); clr.className = "ghost"; clr.textContent = "Clear text";
  clr.onclick = () => { texts.fill(""); inputs.forEach((i) => { i.value = ""; i.dataset.edited = "1"; }); redraw(); };
  row.append(dl, clr); el.append(row);
  return el;
}

$("f").onsubmit = async (e) => {
  e.preventDefault();
  const query = $("q").value.trim() || $("q").placeholder;
  $("go").disabled = true; status("Searching…");
  const t0 = performance.now();
  const words = query.split(/\s+/).filter(Boolean);
  const variants = words.length > 1 ? words.map((_, i) => words.filter((_, k) => k !== i).join(" ")) : [];
  const vs = await embed([query, ...variants]);
  const hits = topK(vs[0], K);
  const imps = hits.map((h) => (variants.length ? wordImportance(vs, words, h.j) : [{ w: query, d: 0 }]));
  $("results").replaceChildren(...hits.map((h, i) => card(h, query, imps[i])));
  status(`${hits.length} templates in ${(performance.now() - t0).toFixed(0)} ms`);
  $("go").disabled = false;
};

load().catch((err) => { loaded(); status("Failed to load: " + err.message); console.error(err); });
