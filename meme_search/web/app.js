// Meme search in the browser: fine-tuned CLIP text encoder (transformers.js) vs precomputed template embeddings.
import { AutoTokenizer, CLIPTextModelWithProjection, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = "./models/";

const K = 6;          // results shown
const DUP_COS = 0.95; // hide near-identical templates (reposts)
const STOP = new Set(("a an the and or but of to in on at for with my me i im i'm is are was were be been it its this that so " +
  "when you your we our they their he she his her just about after before from by as do does did have has had").split(" "));

const $ = (id) => document.getElementById(id);
const status = (t) => ($("status").textContent = t);

let tok, model, emb, meta, D = 512;

async function load() {
  const t0 = performance.now();
  [tok, model, meta, emb] = await Promise.all([
    AutoTokenizer.from_pretrained("meme-clip"),
    CLIPTextModelWithProjection.from_pretrained("meme-clip", { dtype: "fp32" }),
    fetch("data/templates.json").then((r) => r.json()),
    fetch("data/embeddings.bin").then((r) => r.arrayBuffer()).then((b) => new Float32Array(b)),
  ]);
  D = emb.length / meta.length;
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

function fillSlots(query, slots, imp) {
  if (!slots.length) return [];
  const area = (s) => (s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1]);
  const main = slots.reduce((m, s, i) => (area(s) > area(slots[m]) ? i : m), 0);
  const ranked = [...imp].sort((a, b) => b.d - a.d).map((x) => x.w.replace(/[.,!?]/g, ""))
    .filter((w) => w && !STOP.has(w.toLowerCase().replace(/\W/g, "")));
  let k = 0;
  return slots.map((_, i) => (i === main ? query : ranked[k++] ?? ""));
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
    const t = (texts[i] || "").toUpperCase(); if (!t) return;
    const [x0, y0, x1, y1] = s.bbox, w = Math.max(1, x1 - x0), h = Math.max(1, y1 - y0);
    let size = Math.max(10, Math.min(s.line_h * 1.2, h)), lines;
    for (;;) {
      ctx.font = `bold ${size}px Impact, "Arial Black", sans-serif`;
      lines = wrap(ctx, t, w);
      const fits = lines.every((l) => ctx.measureText(l).width <= w) && lines.length * size * 1.15 <= Math.max(h, s.line_h * 1.3);
      if (fits || size <= 10) break;
      size -= 2;
    }
    ctx.textAlign = "center"; ctx.textBaseline = "top"; ctx.lineJoin = "round";
    let y = y0 + (h - lines.length * size * 1.15) / 2;
    for (const l of lines) {
      if (!s.light) { ctx.lineWidth = Math.max(2, size / 7); ctx.strokeStyle = "#000"; ctx.strokeText(l, x0 + w / 2, y); }
      ctx.fillStyle = s.light ? "#000" : "#fff"; ctx.fillText(l, x0 + w / 2, y);
      y += size * 1.15;
    }
  });
}

function card({ j, score }, query, imp) {
  const m = meta[j], texts = fillSlots(query, m.slots, imp);
  const el = document.createElement("div"); el.className = "card";
  const canvas = document.createElement("canvas"); canvas.width = m.w; canvas.height = m.h;
  const img = new Image(); img.src = `data/templates/${m.file}`;
  const redraw = () => img.complete && draw(canvas, img, m.slots, texts);
  img.onload = redraw;
  el.append(canvas);
  const top = [...imp].filter((x) => x.d > 0 && !STOP.has(x.w.toLowerCase().replace(/\W/g, ""))).sort((a, b) => b.d - a.d).slice(0, 4);
  el.insertAdjacentHTML("beforeend", `<div class="meta">match ${score.toFixed(3)} · ${m.slots.length} text slot(s)</div>
    <div class="chips">${top.map((x) => `<span>${x.w} +${x.d.toFixed(3)}</span>`).join("")}</div>`);
  // no detected slots: offer a top caption
  if (!m.slots.length) { m.slots = [{ bbox: [m.w * 0.05, 0, m.w * 0.95, m.h * 0.2], line_h: m.h * 0.08, light: false }]; texts.push(query); }
  m.slots.forEach((_, i) => {
    const row = document.createElement("div"); row.className = "slot";
    row.innerHTML = `<label>slot ${i + 1}</label>`;
    const inp = document.createElement("input"); inp.value = texts[i];
    inp.oninput = () => { texts[i] = inp.value; redraw(); };
    row.append(inp); el.append(row);
  });
  const row = document.createElement("div"); row.className = "row";
  const dl = document.createElement("button"); dl.textContent = "Download PNG";
  dl.onclick = () => { const a = document.createElement("a"); a.download = `meme_${m.file.replace(".webp", "")}.png`; a.href = canvas.toDataURL("image/png"); a.click(); };
  const clr = document.createElement("button"); clr.className = "ghost"; clr.textContent = "Clear text";
  clr.onclick = () => { texts.fill(""); el.querySelectorAll(".slot input").forEach((i) => (i.value = "")); redraw(); };
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

load().catch((err) => { status("Failed to load: " + err.message); console.error(err); });
