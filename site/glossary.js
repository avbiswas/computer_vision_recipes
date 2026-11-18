/* Glossary tooltips for cv.neuralavb.com (same idea as the mechinterp book).

   Automatic: the first time a glossary word appears in the article text (div.prose, not the paper list),
   it gets a dotted underline and an (i) icon. Hover or click it to see the definition.
   Manual: <span class="term" data-term="homography">this word</span> marks any word by hand.

   To add a word: add an entry below. "match" lists the exact spellings to look for (case-sensitive).
   Definitions use simple words (ASD-STE100 style). "link" is optional ("Learn more").
   Some entries also get a small diagram (FIGS, below), shown above the definition. */
const GLOSSARY = {
  // ---------- general ML
  "fine-tune": { name: "Fine-tuning", match: ["fine-tuned", "fine-tune"], def: "Training a model that is already trained, for a short time, on new data. The model keeps what it knows and learns the new task." },
  "cnn": { name: "CNN (convolutional neural network)", match: ["CNN"], def: "A network made of small filters that slide over an image. It finds edges first, then shapes, then objects.", link: "https://en.wikipedia.org/wiki/Convolutional_neural_network" },
  "backbone": { name: "Backbone", match: ["backbone"], def: "The first part of a vision model. It turns the image into feature maps. Other parts (heads) use these features for the task." },
  "transformer": { name: "Transformer", match: ["transformer"], def: "A network that works on a list of tokens. In each layer, every token looks at the other tokens (attention) and updates itself.", link: "https://arxiv.org/abs/1706.03762" },
  "encoder": { name: "Encoder", match: ["text encoder", "hybrid encoder"], def: "The part of a model that turns an input (an image or text) into numbers that the rest of the model uses." },
  "vector": { name: "Vector (embedding)", match: ["512-number vector"], def: "A list of numbers that describes an input. Inputs with similar meaning get vectors that point in similar directions." },
  "dot product": { name: "Dot product", match: ["dot product"], def: "Multiply two vectors number by number and add the results. A large dot product means the two vectors point in the same direction: a good match.", link: "https://en.wikipedia.org/wiki/Dot_product" },
  "vector database": { name: "Vector database", match: ["vector database"], def: "A database that finds the nearest vectors fast, with an index. It is useful for millions of vectors. For 2,000 vectors, a plain loop is fast enough." },
  "k-means": { name: "k-means", match: ["k-means"], def: "A method that puts points into k groups (clusters). Each point goes to the nearest group center. The centers move to the middle of their points, and this repeats.", link: "https://en.wikipedia.org/wiki/K-means_clustering" },
  "near-duplicates": { name: "Near-duplicates", match: ["near-duplicates"], def: "Two templates whose vectors are almost the same (cosine similarity above 0.95), usually the same image posted twice." },
  "heatmap": { name: "Heatmap", match: ["heatmaps"], def: "An image the model outputs where bright pixels mean \"the object is here\". The brightest point (the peak) gives the position." },
  "distillation": { name: "Distillation", match: ["distillation"], def: "A large, slow model (the teacher) makes labels or outputs. A small, fast model (the student) learns to copy them.", link: "https://arxiv.org/abs/1503.02531" },
  "keypoints": { name: "Keypoints", match: ["keypoints"], def: "Special points that the model must find, for example the corners and line crossings of a tennis court." },
  "ocr": { name: "OCR (optical character recognition)", match: ["OCR"], def: "Reading text from the pixels of an image. A normal PDF already stores its text, so it does not need OCR. A scanned PDF does." },

  // ---------- browser
  "webgpu": { name: "WebGPU", match: ["WebGPU"], def: "A browser feature that lets a web page use the graphics card (GPU) for computation. It makes the models here run many times faster than on the CPU.", link: "https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API" },
  "onnxruntime-web": { name: "ONNX Runtime Web", match: ["onnxruntime-web"], def: "A library that runs neural networks in the browser. We export each PyTorch model to an ONNX file, and this library runs it with WebGPU or on the CPU.", link: "https://onnxruntime.ai/docs/tutorials/web/" },
  "pdf.js": { name: "pdf.js", match: ["pdf.js"], def: "Mozilla's PDF reader, written in JavaScript. Here it draws each page as an image and gives the text with its positions.", link: "https://mozilla.github.io/pdf.js/" },

  // ---------- PDF parser
  "d-fine": { name: "D-FINE", match: ["D-FINE nano"], def: "A fast object detector (2024). It predicts each box edge as a probability distribution and refines it in each decoder layer. The nano size has 3.7M parameters.", link: "https://arxiv.org/abs/2410.13842" },
  "detr": { name: "DETR-style detector", match: ["DETR-style"], def: "A detector built on a transformer. A fixed set of learned queries each output one box. It needs no anchor boxes and no non-maximum suppression.", link: "https://arxiv.org/abs/2005.12872" },
  "queries": { name: "Learned queries", match: ["learned queries"], def: "A fixed set of vectors that the detector learns. Each query looks at the image features with attention and outputs one box and one class." },
  "doclaynet": { name: "DocLayNet", match: ["DocLayNet"], def: "A dataset of about 80,000 document pages from reports, manuals, papers, laws, patents and tenders. People drew a box and a class for each region.", link: "https://arxiv.org/abs/2206.01062" },
  "text layer": { name: "PDF text layer", match: ["PDF text layer"], def: "The characters a PDF stores, with their positions on the page. Copy and paste in a PDF reader uses this layer." },
  "reading order": { name: "Reading order", match: ["reading order"], def: "The order a person reads the regions of a page in: for example the full left column, then the right column." },

  // ---------- meme search
  "clip": { name: "CLIP", match: ["CLIP"], def: "A model from OpenAI with an image encoder and a text encoder. It learned from 400 million image-caption pairs to give an image and its caption similar vectors.", link: "https://arxiv.org/abs/2103.00020" },
  "vit": { name: "ViT-B/16", match: ["ViT-B/16"], def: "A Vision Transformer, base size. It cuts the image into 16 × 16 pixel patches and treats each patch as a token.", link: "https://arxiv.org/abs/2010.11929" },
  "memecap": { name: "MemeCap", match: ["MemeCap"], def: "A dataset of 6,416 memes from Reddit. Each meme has a title, a literal description of the picture and an explanation of what the meme means.", link: "https://arxiv.org/abs/2305.13703" },
  "florence-2": { name: "Florence-2", match: ["Florence-2"], def: "A small vision-language model from Microsoft. Here it reads the text on each meme and gives a box for each line.", link: "https://arxiv.org/abs/2311.06242" },
  "lama": { name: "LaMa (inpainting)", match: ["LaMa"], def: "An inpainting model: it fills a masked area of an image with a background that matches the rest. Here it removes the text from memes.", link: "https://arxiv.org/abs/2109.07161" },

  // ---------- world model
  "world model": { name: "World model", match: ["world model"], def: "A model that predicts the next state of an environment from the current state and an action. You can then \"play\" inside the model.", link: "https://arxiv.org/abs/1803.10122" },
  "vq-vae": { name: "VQ-VAE", match: ["VQ-VAE"], def: "An autoencoder with a codebook. The encoder turns an image into a grid of vectors. Each vector becomes the id of the nearest codebook entry. The decoder turns the ids back into an image.", link: "https://arxiv.org/abs/1711.00937" },
  "codebook": { name: "Codebook", match: ["codebook"], def: "A learned list of vectors (here 512). Each part of an image is replaced by the id of its nearest codebook vector, so an image becomes a grid of integers." },
  "tokens": { name: "Tokens", match: ["tokens"], def: "The integers the transformer reads and writes. Here one frame is 16 × 16 = 256 tokens, each an id from the codebook." },
  "genie": { name: "Genie", match: ["Genie-style"], def: "A world model from Google DeepMind (2024). A tokenizer turns frames into tokens, and a spatio-temporal transformer predicts the next frame's tokens from past frames and actions.", link: "https://arxiv.org/abs/2402.15391" },
  "spatial attention": { name: "Spatial attention", match: ["spatial attention"], def: "Attention between the tokens of one frame. It lets each part of the image look at the other parts." },
  "temporal attention": { name: "Causal temporal attention", match: ["causal temporal attention"], def: "Attention between the same grid position in different frames. Causal means a frame can only look at earlier frames, never at the future." },
  "rope": { name: "Rotary position embeddings (RoPE)", match: ["rotary position embeddings"], def: "A way to tell attention where each token is. It rotates the query and key vectors by an angle that depends on the position. Here there are three positions: row, column and time.", link: "https://arxiv.org/abs/2104.09864" },

  // ---------- tennis
  "resnet": { name: "ResNet-18", match: ["ResNet-18"], def: "A classic 18-layer CNN with skip connections. We use its first stages as the encoder and remove the last stage, which has most of the weights.", link: "https://arxiv.org/abs/1512.03385" },
  "u-net": { name: "U-Net decoder", match: ["U-Net"], def: "A decoder that makes the feature maps larger again, step by step. Skip connections bring back fine detail from the encoder.", link: "https://arxiv.org/abs/1505.04597" },
  "homography": { name: "Homography", match: ["homography"], def: "A 3 × 3 matrix that maps points on one flat surface to another, for example the court in the video to a top-down court diagram. 4 or more matching points give the matrix.", link: "https://en.wikipedia.org/wiki/Homography_(computer_vision)" },
  "grounding dino": { name: "Grounding DINO", match: ["Grounding DINO"], def: "An open-set object detector: you give it a text prompt such as \"a person.\" and it finds boxes for it.", link: "https://arxiv.org/abs/2303.05499" },
  "channels": { name: "Frames as channels", match: ["stacked as channels"], def: "A color image has 3 channels (red, green, blue). Here the 3 channels are 3 gray frames in a row, so the network sees motion in one image." },
};

// ---------- small diagrams shown inside some tooltips (light colors: the tooltip card is light)
const F = (body, h = 110) => `<svg viewBox="0 0 300 ${h}" xmlns="http://www.w3.org/2000/svg"><defs><marker id="ga" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#555"/></marker></defs>${body}</svg>`;
const B = (x, y, w, h, t, fill = "#fff", stroke = "#8a8f98") => `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}" stroke="${stroke}"/><text x="${x + w / 2}" y="${y + h / 2 + 4}" text-anchor="middle">${t}</text>`;
const A = (x1, y1, x2, y2) => `<path d="M${x1},${y1} L${x2},${y2}" stroke="#555" stroke-width="1.3" fill="none" marker-end="url(#ga)"/>`;
const grid = (x, y, n, s, f = () => "#e6e3dc") => { let o = ""; for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) o += `<rect x="${x + c * s}" y="${y + r * s}" width="${s - 1}" height="${s - 1}" fill="${f(r, c)}"/>`; return o; };
const BLUE = "#1f6fb2", ORANGE = "#d9862b", GREEN = "#3c8d4f";

const FIGS = {
  "clip": F(B(10, 12, 60, 34, "🖼 image", "#eaf2fb", BLUE) + B(10, 64, 60, 34, '"a cat"', "#fbf1e4", ORANGE) +
    B(92, 12, 84, 34, "image encoder") + B(92, 64, 84, 34, "text encoder") + A(70, 29, 90, 29) + A(70, 81, 90, 81) +
    A(176, 29, 214, 48) + A(176, 81, 214, 62) + `<circle cx="230" cy="55" r="14" fill="#e9f5ec" stroke="${GREEN}"/><text x="230" y="60" text-anchor="middle">✓</text><text x="252" y="59">match</text>`),
  "dot product": F(`<g stroke-width="2.5" fill="none"><path d="M30,95 L110,25" stroke="${BLUE}" marker-end="url(#ga)"/><path d="M30,95 L125,45" stroke="${ORANGE}" marker-end="url(#ga)"/>
    <path d="M175,95 L255,30" stroke="${BLUE}" marker-end="url(#ga)"/><path d="M175,95 L262,88" stroke="${ORANGE}" marker-end="url(#ga)"/></g>
    <text x="40" y="18">same direction</text><text x="70" y="113" text-anchor="middle" class="s">large → good match</text><text x="196" y="18">different</text><text x="215" y="113" text-anchor="middle" class="s">small → poor match</text>`, 118),
  "k-means": F([[60, 40, BLUE], [70, 70, BLUE], [45, 60, BLUE], [80, 50, BLUE], [150, 30, ORANGE], [170, 50, ORANGE], [140, 55, ORANGE], [165, 25, ORANGE], [240, 75, GREEN], [255, 55, GREEN], [225, 60, GREEN], [250, 85, GREEN]]
    .map(([x, y, c]) => `<circle cx="${x}" cy="${y}" r="4.5" fill="${c}"/>`).join("") +
    [[64, 55, BLUE], [156, 40, ORANGE], [243, 69, GREEN]].map(([x, y, c]) => `<path d="M${x - 6},${y - 6} L${x + 6},${y + 6} M${x + 6},${y - 6} L${x - 6},${y + 6}" stroke="${c}" stroke-width="3"/>`).join("") +
    `<text x="150" y="104" text-anchor="middle" class="s">✕ = group center (k = 3)</text>`),
  "vq-vae": F(`<rect x="8" y="30" width="44" height="44" fill="#2b2f38"/><circle cx="24" cy="46" r="7" fill="#f6d77a"/>` + A(54, 52, 68, 52) +
    B(70, 36, 52, 32, "encoder") + A(122, 52, 136, 52) + grid(138, 28, 4, 12, (r, c) => ["#d8e6f5", "#f5e3cc", "#e2efe4", "#ece9e2"][(r * 3 + c) % 4]) +
    `<text x="162" y="94" text-anchor="middle" class="s">grid of ids</text>` + A(188, 52, 202, 52) + B(204, 36, 52, 32, "decoder") + A(256, 52, 268, 52) +
    `<rect x="270" y="38" width="28" height="28" fill="#2b2f38"/><circle cx="280" cy="48" r="4.5" fill="#f6d77a"/>`),
  "codebook": F(`<text x="10" y="22" class="s">one image patch</text><rect x="10" y="30" width="60" height="16" fill="#eaf2fb" stroke="${BLUE}"/><text x="40" y="42" text-anchor="middle" class="s">vector</text>` +
    A(72, 38, 120, 38) + `<text x="80" y="30" class="s">nearest</text>` +
    [0, 1, 2, 3].map((i) => `<rect x="125" y="${12 + i * 22}" width="70" height="17" fill="${i === 2 ? "#fbf1e4" : "#fff"}" stroke="${i === 2 ? ORANGE : "#8a8f98"}"/><text x="132" y="${25 + i * 22}" class="s">${["code 0", "code 1", "code 37", "… 511"][i]}</text>`).join("") +
    A(197, 64, 232, 64) + `<text x="240" y="69" font-weight="700">37</text>`),
  "tokens": F(grid(20, 10, 4, 22) + [[12, 12, 37, 37], [12, 5, 5, 37], [8, 5, 5, 9], [8, 8, 9, 9]].map((row, r) => row.map((v, c) => `<text x="${31 + c * 22}" y="${26 + r * 22}" text-anchor="middle" class="s">${v}</text>`).join("")).join("") +
    `<text x="130" y="45">one frame → a grid of</text><text x="130" y="63">integer ids (16 × 16 here)</text>`),
  "spatial attention": F(spatTemp(true)),
  "temporal attention": F(spatTemp(false)),
  "rope": F([0, 1, 2].map((i) => {
    const cx = 50 + i * 100, a = -Math.PI / 2 + (i + 1) * 0.75, c = [BLUE, ORANGE, GREEN][i];
    return `<circle cx="${cx}" cy="48" r="30" fill="none" stroke="#d6d2ca"/><path d="M${cx},48 L${cx},20" stroke="#bbb" stroke-dasharray="3 3"/>` +
      `<path d="M${cx},48 L${cx + 26 * Math.cos(a)},${48 + 26 * Math.sin(a)}" stroke="${c}" stroke-width="2.5" marker-end="url(#ga)"/>` +
      `<text x="${cx}" y="96" text-anchor="middle" class="s">position ${i + 1}</text>`;
  }).join("") + `<text x="150" y="110" text-anchor="middle" class="s">the vector turns more at each position</text>`, 116),
  "heatmap": F(grid(20, 8, 8, 12, (r, c) => { const d = Math.hypot(r - 3, c - 5); const v = Math.max(0, 1 - d / 3); return `rgba(217,134,43,${(0.08 + v * 0.92).toFixed(2)})`; }) +
    `<circle cx="86" cy="50" r="4" fill="none" stroke="#000" stroke-width="1.5"/>` + A(122, 50, 160, 50) + `<text x="168" y="46">peak = ball at (x, y)</text><text x="168" y="64" class="s">brighter = more likely</text>`),
  "homography": F(`<path d="M30,95 L80,20 L140,20 L190,95 Z" fill="#e2efe4" stroke="${GREEN}"/><circle cx="110" cy="70" r="4" fill="${ORANGE}"/><text x="88" y="110" class="s">camera view</text>` +
    A(196, 58, 218, 58) + `<text x="207" y="48" text-anchor="middle" class="s">H</text><rect x="224" y="12" width="56" height="86" fill="#e2efe4" stroke="${GREEN}"/><circle cx="252" cy="70" r="4" fill="${ORANGE}"/><text x="230" y="110" class="s">top-down</text>`, 114),
  "channels": F(["t−2", "t−1", "t"].map((t, i) => `<rect x="${12 + i * 12}" y="${14 + i * 10}" width="56" height="56" fill="#${["d4d4d4", "bdbdbd", "a6a6a6"][i]}" stroke="#666"/><text x="${40 + i * 12}" y="${92 + i * 0}" class="s"></text>`).join("") +
    `<text x="20" y="104" class="s">3 gray frames</text>` + A(100, 52, 140, 52) + `<rect x="150" y="24" width="56" height="56" fill="#fff" stroke="#666"/><circle cx="164" cy="38" r="4" fill="#d4d4d4" stroke="#999"/><circle cx="178" cy="49" r="4" fill="#bdbdbd" stroke="#888"/><circle cx="192" cy="60" r="4" fill="#555"/>` +
    `<text x="214" y="48">one 3-channel</text><text x="214" y="64">input: motion</text><text x="214" y="80">is visible</text>`),
  "distillation": F(B(10, 30, 80, 44, "big teacher", "#fbf1e4", ORANGE) + `<text x="50" y="92" text-anchor="middle" class="s">slow, run once</text>` + A(90, 52, 120, 52) +
    B(122, 36, 60, 32, "labels") + A(182, 52, 212, 52) + B(214, 36, 76, 32, "small student", "#eaf2fb", BLUE) + `<text x="252" y="88" text-anchor="middle" class="s">fast, 200 fps</text>`),
  "queries": F([0, 1, 2, 3].map((i) => `<circle cx="22" cy="${18 + i * 24}" r="7" fill="#eaf2fb" stroke="${BLUE}"/>` + A(31, 18 + i * 24, 70, 18 + i * 24)).join("") +
    `<rect x="78" y="8" width="120" height="96" fill="#fff" stroke="#8a8f98"/>` + [[86, 14, 104, 14, ORANGE], [86, 34, 60, 26, BLUE], [152, 34, 38, 50, GREEN], [86, 66, 60, 32, BLUE]].map(([x, y, w, h, c]) => `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${c}" stroke-width="2"/>`).join("") +
    `<text x="206" y="48">each query →</text><text x="206" y="66">one box + class</text>`),
};
function spatTemp(spatial) {
  let o = "";
  for (let f = 0; f < 3; f++) o += grid(20 + f * 90, 20, 3, 18) + `<text x="${47 + f * 90}" y="100" text-anchor="middle" class="s">frame ${["t−2", "t−1", "t"][f]}</text>`;
  const cell = (f, r, c) => [20 + f * 90 + c * 18 + 8.5, 20 + r * 18 + 8.5];
  const ln = ([x1, y1], [x2, y2], col) => `<path d="M${x1},${y1} L${x2},${y2}" stroke="${col}" stroke-width="2"/>`;
  if (spatial) { const c0 = cell(2, 1, 1); for (const [r, c] of [[0, 0], [0, 2], [2, 0], [2, 2], [1, 0]]) o += ln(c0, cell(2, r, c), BLUE); o += `<circle cx="${c0[0]}" cy="${c0[1]}" r="4" fill="${BLUE}"/>`; }
  else {
    const c2 = cell(2, 1, 1), arc = ([x1, y1], [x2, y2], h) => `<path d="M${x1},${y1} Q${(x1 + x2) / 2},${y1 - h} ${x2},${y2}" stroke="${ORANGE}" stroke-width="2" fill="none"/>`;
    o += arc(cell(0, 1, 1), c2, 44) + arc(cell(1, 1, 1), c2, 24);
    for (const f of [0, 1, 2]) { const [x, y] = cell(f, 1, 1); o += `<circle cx="${x}" cy="${y}" r="4" fill="${ORANGE}"/>`; }
  }
  return o;
}
for (const [k, fig] of Object.entries(FIGS)) if (GLOSSARY[k]) GLOSSARY[k].fig = fig;

(() => {
  const keys = Object.keys(GLOSSARY);
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");

  // ---------- 1. mark the first use of each word in the article text
  const used = new Set();
  const roots = document.querySelectorAll("article div.prose");
  const skip = (n) => n.parentElement.closest("a, code, pre, .term, .papers, h1, h2, h3, svg, button, label");
  for (const root of roots) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = []; for (let n; (n = walker.nextNode());) if (!skip(n)) nodes.push(n);
    for (let node of nodes) {
      // repeatedly wrap the earliest unused glossary word in this text node, then continue after it
      for (;;) {
        let best = null;
        for (const k of keys) {
          if (used.has(k)) continue;
          for (const m of GLOSSARY[k].match) {
            const hit = new RegExp(`(^|[^\\w-])(${esc(m)})(?![\\w-])`).exec(node.nodeValue);
            if (hit && (!best || hit.index + hit[1].length < best.start)) best = { k, m, start: hit.index + hit[1].length };
          }
        }
        if (!best) break;
        const word = node.splitText(best.start), rest = word.splitText(best.m.length);
        const span = document.createElement("span");
        span.className = "term"; span.dataset.term = best.k;
        word.replaceWith(span); span.append(word);
        used.add(best.k); node = rest;
      }
    }
  }

  // ---------- 2. (i) icon + tooltip (hover on desktop, click/tap to pin)
  const tip = document.createElement("div");
  tip.className = "tip"; tip.setAttribute("role", "tooltip");
  document.body.append(tip);
  let pinned = null;

  document.querySelectorAll(".term").forEach((t) => {
    if (!t.querySelector(".ti")) t.insertAdjacentHTML("beforeend", '<span class="ti" aria-hidden="true">i</span>');
    t.tabIndex = 0; t.setAttribute("role", "button");
  });

  function show(el) {
    const g = GLOSSARY[el.dataset.term] || { name: el.textContent, def: el.dataset.def || "" };
    tip.innerHTML = `<b>${g.name}</b>` + (g.fig ? `<div class="tip-fig">${g.fig}</div>` : "") + `<span>${g.def}</span>` + (g.link ? `<a href="${g.link}" target="_blank" rel="noopener">Learn more →</a>` : "");
    tip.classList.add("show");
    const r = el.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    let top = r.bottom + 8; if (top + h > innerHeight - 8 && r.top - h - 8 > 8) top = r.top - h - 8;
    tip.style.left = `${Math.max(8, Math.min(innerWidth - w - 8, r.left)) + scrollX}px`;
    tip.style.top = `${top + scrollY}px`;
  }
  const hide = () => { if (!pinned) tip.classList.remove("show"); };

  document.addEventListener("mouseover", (e) => { const t = e.target.closest(".term"); if (t && !pinned) show(t); });
  document.addEventListener("mouseout", (e) => { const t = e.target.closest(".term"); if (t && !t.contains(e.relatedTarget) && !tip.contains(e.relatedTarget)) hide(); });
  tip.addEventListener("mouseleave", hide);
  document.addEventListener("click", (e) => {
    const t = e.target.closest(".term");
    if (t) { if (pinned === t) { pinned = null; tip.classList.remove("show"); } else { pinned = t; show(t); } return; }
    if (!tip.contains(e.target)) { pinned = null; tip.classList.remove("show"); }
  });
  document.addEventListener("keydown", (e) => {
    const t = e.target.closest?.(".term");
    if (t && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); t.click(); }
    if (e.key === "Escape") { pinned = null; tip.classList.remove("show"); }
  });
  addEventListener("scroll", () => { if (pinned) show(pinned); }, { passive: true });
})();
