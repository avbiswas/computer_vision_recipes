"""Build before/after images for the video.

For each (pdf, page): [original page | detected layout | naive text dump | our markdown]
python make_demo.py --weights runs/y11s_doclaynet/weights/best.pt
"""
import argparse, json, textwrap
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

import parse_pdf

PAGES = [  # (pdf, page, why it's hard)
    ("samples/attention.pdf", 8, "complex table with merged cells"),
    ("samples/resnet_2col.pdf", 0, "two-column paper: naive dump interleaves columns"),
    ("samples/resnet_2col.pdf", 5, "two columns + tables"),
    ("samples/resnet_scanned.pdf", 0, "scanned page, no text layer at all"),
]
H = 1100
FONT = "/System/Library/Fonts/Menlo.ttc"


def text_panel(text, title, w=700):
    im = Image.new("RGB", (w, H), "white"); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, 13); ft = ImageFont.truetype(FONT, 20)
    d.rectangle([0, 0, w, 44], fill="#14201c"); d.text((14, 10), title, fill="white", font=ft)
    y = 56
    for para in text.splitlines():
        for line in (textwrap.wrap(para, 86) or [""]):
            if y > H - 20:
                d.text((14, y), "…", fill="#888", font=f); return im
            d.text((14, y), line, fill="#111", font=f); y += 17
    return im


def page_panel(img, title):
    s = (H - 44) / img.height
    im = img.resize((int(img.width * s), H - 44))
    out = Image.new("RGB", (im.width, H), "white"); d = ImageDraw.Draw(out)
    d.rectangle([0, 0, im.width, 44], fill="#14201c")
    d.text((14, 10), title, fill="white", font=ImageFont.truetype(FONT, 20))
    out.paste(im, (0, 44)); return out


def main(weights, out, use_vlm):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    reader = parse_pdf.Reader() if use_vlm else None
    notes = []
    for pdf, pg, why in PAGES:
        stem = f"{Path(pdf).stem}_p{pg}"
        tmp = out / "parsed"
        md_path = parse_pdf.parse(pdf, weights, tmp, [pg], use_vlm, reader)
        ours = md_path.read_text()
        page = fitz.open(pdf)[pg]
        naive = page.get_text() or "(empty: this PDF has no text layer)"
        pix = page.get_pixmap(dpi=parse_pdf.DPI)
        orig = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        layout = Image.open(tmp / f"{Path(pdf).stem}_p{pg}_layout.png")
        panels = [page_panel(orig, "1. Original page"), page_panel(layout, "2. Detected layout"),
                  text_panel(naive, "3. Naive text dump"), text_panel(ours, "4. Our markdown")]
        W = sum(p.width for p in panels) + 20 * 3
        canvas = Image.new("RGB", (W, H), "#d5ddd9"); x = 0
        for p in panels:
            canvas.paste(p, (x, 0)); x += p.width + 20
        canvas.save(out / f"{stem}_before_after.png")
        (out / f"{stem}_naive.txt").write_text(naive)
        (out / f"{stem}_ours.md").write_text(ours)
        notes.append({"page": stem, "why_hard": why, "image": f"{stem}_before_after.png"})
        print("saved", stem, flush=True)
    json.dump(notes, open(out / "index.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/baseline/yolov11s-doclaynet.pt")
    ap.add_argument("--out", default="demo")
    ap.add_argument("--no-vlm", action="store_true")
    a = ap.parse_args()
    main(a.weights, a.out, not a.no_vlm)
