"""PDF -> clean markdown + JSON.

Pipeline per page:
  1. render page to an image (PyMuPDF), keep the PDF text layer if it has one
  2. detect layout regions with a YOLO model (our fine-tune or the baseline)
  3. read each region: text layer for text-like regions, Granite-Docling VLM for
     tables (OTSL -> markdown), formulas (LaTeX) and text on scanned pages
  4. order regions (column-aware) and assemble markdown + JSON

Usage:
  python parse_pdf.py file.pdf [--weights runs/.../best.pt] [--out out/] [--pages 0,1] [--no-vlm]
"""
import argparse, json, re
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

CLASSES = ["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header",
           "Picture", "Section-header", "Table", "Text", "Title"]
COLORS = {"Caption": "#e6194b", "Footnote": "#911eb4", "Formula": "#f58231", "List-item": "#3cb44b",
          "Page-footer": "#808080", "Page-header": "#808080", "Picture": "#4363d8", "Section-header": "#f032e6",
          "Table": "#ffe119", "Text": "#42d4f4", "Title": "#000075"}
DPI = 144
BULLET = re.compile(r"^[•·▪◦\-*]\s*")
VLM_DIR = Path(__file__).parent / "models/granite-docling-258M"


# ---------------------------------------------------------------- VLM reader
class Reader:
    """Granite-Docling (258M) used per region crop."""

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(VLM_DIR)
        self.model = AutoModelForImageTextToText.from_pretrained(VLM_DIR, dtype=torch.bfloat16).to(self.dev).eval()

    def ask(self, image, instruction, max_new_tokens=1024):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
        prompt = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=prompt, images=[image], return_tensors="pt").to(self.dev)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=False)[0].strip()

    def table(self, crop):
        return otsl_to_markdown(self.ask(pad(crop), "Convert table to OTSL.", 2048))

    def formula(self, crop):
        return strip_tags(self.ask(pad(crop), "Convert formula to LaTeX.", 512))

    def text(self, crop):
        return strip_tags(self.ask(pad(crop), "Convert this page to docling.", 512))


def pad(img, m=12):
    out = Image.new("RGB", (img.width + 2 * m, img.height + 2 * m), "white")
    out.paste(img, (m, m))
    return out


def strip_tags(s):
    s = re.sub(r"<loc_\d+>", "", s)
    s = re.sub(r"</?[a-z_]+>", " ", s)
    s = re.sub(r"<\|[^|>]*\|>|<end_of_utterance>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def otsl_to_markdown(s):
    """Convert Granite-Docling OTSL output into a markdown table."""
    s = re.sub(r"<loc_\d+>|<\|[^|>]*\|>", "", s)
    m = re.search(r"<otsl>(.*?)(</otsl>|$)", s, re.S)
    body = m.group(1) if m else s
    rows, row = [], []
    for tok, txt in re.findall(r"<(fcel|ecel|lcel|ucel|xcel|ched|rhed|srow|nl)>([^<]*)", body):
        if tok == "nl":
            if row:
                rows.append(row)
            row = []
        elif tok in ("fcel", "ched", "rhed", "srow"):
            row.append(txt.strip())
        elif tok == "ecel":
            row.append("")
        else:  # merged cells: lcel/ucel/xcel
            row.append("")
    if row:
        rows.append(row)
    if not rows:
        return strip_tags(s)
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    esc = lambda c: c.replace("|", "\\|")
    lines = ["| " + " | ".join(map(esc, rows[0])) + " |", "|" + "---|" * w]
    lines += ["| " + " | ".join(map(esc, r)) + " |" for r in rows[1:]]
    return "\n".join(lines)


# ---------------------------------------------------------------- layout
def detect(model, img, conf=0.25):
    r = model.predict(img, imgsz=1024, conf=conf, verbose=False)[0]
    regs = []
    for (x0, y0, x1, y1), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        regs.append({"type": CLASSES[int(c)], "box": [round(x0), round(y0), round(x1), round(y1)], "conf": round(s, 3)})
    return dedupe(regs)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0


def dedupe(regs, thr=0.6):
    """Drop overlapping boxes of different classes, keeping the most confident."""
    keep = []
    for r in sorted(regs, key=lambda r: -r["conf"]):
        if all(iou(r["box"], k["box"]) < thr for k in keep):
            keep.append(r)
    return keep


def reading_order(regs, page_w):
    """Column-aware order: full-width blocks split the page into bands; within a
    band, the left column is read before the right column."""
    mid = page_w / 2
    def is_full(r):
        x0, _, x1, _ = r["box"]
        return x0 < mid - page_w * 0.05 and x1 > mid + page_w * 0.05
    regs = sorted(regs, key=lambda r: r["box"][1])
    out, band = [], []
    def flush():
        left = [r for r in band if (r["box"][0] + r["box"][2]) / 2 < mid]
        right = [r for r in band if r not in left]
        out.extend(sorted(left, key=lambda r: r["box"][1]) + sorted(right, key=lambda r: r["box"][1]))
        band.clear()
    for r in regs:
        if is_full(r):
            flush(); out.append(r)
        else:
            band.append(r)
    flush()
    return out


# ---------------------------------------------------------------- reading
def text_from_layer(page, box, scale):
    rect = fitz.Rect(*(v / scale for v in box))
    words = page.get_text("words", clip=rect)
    words = [w for w in words if fitz.Rect(w[:4]).intersect(rect).get_area() > 0.5 * fitz.Rect(w[:4]).get_area()]
    words.sort(key=lambda w: (w[5], w[6], w[7]))  # block, line, word
    out = ""
    for i, w in enumerate(words):
        out += w[4]
        nxt = words[i + 1] if i + 1 < len(words) else None
        # a word ending in "-" at the end of a line was hyphenated by the line break: glue it to the next word
        broke = nxt and w[4].endswith("-") and len(w[4]) > 1 and (nxt[5], nxt[6]) != (w[5], w[6])
        if broke and nxt[4][:1].islower():
            out = out[:-1]  # "conver-" + "gence" -> "convergence"
        elif broke:
            pass            # "ResNet-" + "34" -> "ResNet-34" (real hyphen, just no space)
        elif nxt:
            out += " "
    return re.sub(r"\s+", " ", out).strip()


def to_markdown(r, fig_dir, page_idx, i, img):
    t, c = r["type"], r.get("content", "")
    if t in ("Page-header", "Page-footer"):
        return None
    if t == "Title":
        return f"# {c}"
    if t == "Section-header":
        return f"## {c}"
    if t == "List-item":
        return "- " + BULLET.sub("", c)
    if t == "Formula":
        return f"$$\n{c}\n$$"
    if t == "Picture":
        fig_dir.mkdir(parents=True, exist_ok=True)
        p = fig_dir / f"p{page_idx}_fig{i}.png"
        img.crop(r["box"]).save(p)
        r["image"] = str(p)
        return f"![figure]({p.parent.name}/{p.name})"
    if t == "Caption":
        return f"*{c}*"
    if t == "Footnote":
        return f"<sub>{c}</sub>"
    return c


def draw(img, regs):
    im = img.copy(); d = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for n, r in enumerate(regs, 1):
        col = COLORS[r["type"]]
        rgb = tuple(int(col[i:i + 2], 16) for i in (1, 3, 5))
        d.rectangle(r["box"], outline=col, width=3, fill=rgb + (40,))
        label = f"{n}. {r['type']}"
        tw = d.textlength(label, font=font)
        d.rectangle([r["box"][0], r["box"][1] - 18, r["box"][0] + tw + 8, r["box"][1]], fill=col)
        d.text((r["box"][0] + 4, r["box"][1] - 17), label, fill="white", font=font)
    return im


def parse(pdf, weights, out, pages=None, use_vlm=True, reader=None):
    from ultralytics import YOLO
    model = YOLO(weights)
    reader = reader or (Reader() if use_vlm else None)
    doc = fitz.open(pdf)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stem = Path(pdf).stem
    md_parts, js = [], {"file": str(pdf), "pages": []}
    for pi in (pages if pages is not None else range(len(doc))):
        page = doc[pi]
        scale = DPI / 72
        pix = page.get_pixmap(dpi=DPI)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        has_text = len(page.get_text("words")) > 5
        regs = reading_order(detect(model, img), img.width)
        for r in regs:
            crop = img.crop(r["box"])
            t = r["type"]
            if t == "Table" and reader:
                r["content"], r["source"] = reader.table(crop), "vlm"
            elif t == "Formula" and reader:
                r["content"], r["source"] = reader.formula(crop), "vlm"
            elif t == "Picture":
                r["content"], r["source"] = "", "crop"
            elif has_text:
                r["content"], r["source"] = text_from_layer(page, r["box"], scale), "text-layer"
            elif reader:
                r["content"], r["source"] = reader.text(crop), "vlm"
            else:
                r["content"], r["source"] = "", "none"
        md_parts += [m for i, r in enumerate(regs) if (m := to_markdown(r, out / f"{stem}_figures", pi, i, img))]
        draw(img, regs).save(out / f"{stem}_p{pi}_layout.png")
        js["pages"].append({"page": pi, "size": [img.width, img.height], "has_text_layer": has_text, "regions": regs})
    (out / f"{stem}.md").write_text("\n\n".join(md_parts) + "\n")
    (out / f"{stem}.json").write_text(json.dumps(js, indent=2, ensure_ascii=False))
    return out / f"{stem}.md"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--weights", default="models/baseline/yolov11s-doclaynet.pt")
    ap.add_argument("--out", default="out")
    ap.add_argument("--pages", default=None, help="comma-separated 0-based page numbers")
    ap.add_argument("--no-vlm", action="store_true")
    a = ap.parse_args()
    pages = [int(p) for p in a.pages.split(",")] if a.pages else None
    print(parse(a.pdf, a.weights, a.out, pages, not a.no_vlm))
