"""Query -> best meme templates with pre-filled, editable text slots.

  python search.py "my code finally compiled after 6 hours" --ckpt runs/clip/best.pt -k 3

1. CLIP ranks text-free templates (out/clean) against the query.
2. Text lines found by remove_text.py (out/layout) are grouped into slots.
3. Leave-one-out CLIP scoring finds the query words that drive the match; they pre-fill the slots.
4. Writes out/search/<rank>_<name>.png (preview) and .json (slots, for an editor).
"""
import argparse, json, os, re
import numpy as np, torch, torch.nn.functional as F, open_clip
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
STOP = set("a an the and or but of to in on at for with my me i im i'm is are was were be been it its this that so "
           "when you your we our they their he she his her just about after before from by as do does did have has had".split())


class Index:
    def __init__(self, model="ViT-B-16:openai", ckpt=None, clean_dir=f"{ROOT}/out/clean", cache=f"{ROOT}/out/index.pt"):
        name, pt = model.split(":")
        self.model, _, self.pre = open_clip.create_model_and_transforms(name, pretrained=pt)
        if ckpt:
            self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model = self.model.to(dev).eval()
        self.tok = open_clip.get_tokenizer(name)
        key = f"{model}|{ckpt}"
        if os.path.exists(cache) and torch.load(cache)["key"] == key:
            c = torch.load(cache); self.files, self.emb = c["files"], c["emb"].to(dev)
            return
        self.files = sorted(os.listdir(clean_dir))
        embs = []
        with torch.no_grad():
            for i in range(0, len(self.files), 64):
                b = torch.stack([self.pre(Image.open(f"{clean_dir}/{f}").convert("RGB")) for f in self.files[i:i + 64]])
                embs.append(self.model.encode_image(b.to(dev)).float())
        self.emb = F.normalize(torch.cat(embs), dim=-1)
        torch.save({"key": key, "files": self.files, "emb": self.emb.cpu()}, cache)

    @torch.no_grad()
    def text(self, texts):
        return F.normalize(self.model.encode_text(self.tok(texts).to(dev)).float(), dim=-1)

    def search(self, query, k=5):
        s = (self.text([query]) @ self.emb.T)[0]
        top = s.topk(k)
        return [(self.files[i], float(v)) for v, i in zip(top.values.tolist(), top.indices.tolist())]

    def word_importance(self, query, fname):
        """Similarity drop when each word is removed (leave-one-out)."""
        words = query.split()
        img = self.emb[self.files.index(fname)]
        variants = [query] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        s = (self.text(variants) @ img).tolist()
        return [(w, s[0] - s[i + 1]) for i, w in enumerate(words)]


def group_slots(lines, img_h):
    """Merge OCR lines stacked closely with similar height into one slot (a caption block)."""
    lines = sorted(lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))
    slots = []
    for l in lines:
        x0, y0, x1, y1 = l["bbox"]; h = max(1, y1 - y0)
        for s in slots:
            sx0, sy0, sx1, sy1 = s["bbox"]
            gap = y0 - sy1
            overlap = min(x1, sx1) - max(x0, sx0)
            if -0.3 * h < gap < 0.8 * h and overlap > 0 and 0.6 < h / s["line_h"] < 1.6:
                s["bbox"] = [min(x0, sx0), sy0, max(x1, sx1), max(y1, sy1)]
                s["lines"] += 1; s["text"] += " " + l["text"]
                break
        else:
            slots.append({"bbox": [x0, y0, x1, y1], "line_h": h, "lines": 1, "text": l["text"],
                          "color": l["color"], "stroke": l["stroke"]})
    # merge slots whose boxes overlap (fragments of one caption), keep the 4 largest
    merged = True
    while merged:
        merged = False
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                a, b = slots[i]["bbox"], slots[j]["bbox"]
                if min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]):
                    slots[i]["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    slots[i]["text"] += " " + slots[j]["text"]; slots[i]["lines"] += slots[j]["lines"]
                    slots[i]["line_h"] = max(slots[i]["line_h"], slots[j]["line_h"])
                    del slots[j]; merged = True; break
            if merged:
                break
    # drop watermarks / usernames ("made with mematic", "imgflip.com", "u/name") and tiny text
    wm = re.compile(r"mematic|imgflip|\.com|^u/|^@|reddit", re.I)
    slots = [s for s in slots if s["line_h"] > 0.02 * img_h and not wm.search(s["text"].strip())]
    area = lambda s: (s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1])
    keep = sorted(sorted(slots, key=area, reverse=True)[:4], key=lambda s: (s["bbox"][1], s["bbox"][0]))
    return keep


def fill_slots(query, slots, importance):
    """Largest slot gets the full query; other slots get the most important words (leave-one-out), one each."""
    if not slots:
        return []
    area = lambda s: (s["bbox"][2] - s["bbox"][0]) * (s["bbox"][3] - s["bbox"][1])
    main = max(range(len(slots)), key=lambda i: area(slots[i]))
    ranked = [w.strip(".,!?") for w, d in sorted(importance, key=lambda x: -x[1]) if re.sub(r"\W", "", w.lower()) not in STOP]
    out, k = [], 0
    for i in range(len(slots)):
        if i == main:
            out.append(query)
        else:
            out.append(ranked[k] if k < len(ranked) else ""); k += 1
    return out


def wrap(draw, text, font, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    return lines + ([cur] if cur else [])


def render(img, slots, texts):
    """Classic meme style: white + black outline on images, black on light caption bars. Shrink to fit."""
    img = img.copy(); d = ImageDraw.Draw(img); g = np.asarray(img.convert("L"))
    for s, t in zip(slots, texts):
        if not t:
            continue
        x0, y0, x1, y1 = [int(v) for v in s["bbox"]]
        x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(img.width, x1), min(img.height, y1)
        w, h = max(1, x1 - x0), max(1, y1 - y0)
        bg = g[y0:y1, x0:x1]
        light_bar = bg.size and bg.mean() > 200 and bg.std() < 30
        fill, stroke = ((0, 0, 0), None) if light_bar else ((255, 255, 255), (0, 0, 0))
        size = max(10, int(min(s["line_h"] * 1.2, h)))
        while True:
            font = ImageFont.load_default(size)
            lines = wrap(d, t.upper(), font, w)
            fits = all(d.textlength(l, font=font) <= w for l in lines) and len(lines) * size * 1.15 <= max(h, s["line_h"] * 1.3)
            if fits or size <= 10:
                break
            size -= 2
        yy = y0 + (h - len(lines) * size * 1.15) / 2
        for ln in lines:
            tw = d.textlength(ln, font=font)
            d.text((x0 + (w - tw) / 2, yy), ln, font=font, fill=fill,
                   stroke_width=max(1, size // 12) if stroke else 0, stroke_fill=stroke)
            yy += size * 1.15
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--model", default="ViT-B-16:openai")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=f"{ROOT}/out/search")
    args = ap.parse_args()
    idx = Index(args.model, args.ckpt)
    os.makedirs(args.out, exist_ok=True)
    for r, (f, score) in enumerate(idx.search(args.query, args.k)):
        stem = os.path.splitext(f)[0]
        img = Image.open(f"{ROOT}/out/clean/{f}").convert("RGB")
        lay = json.load(open(f"{ROOT}/out/layout/{stem}.json"))
        slots = group_slots(lay["text"], img.height)
        imp = idx.word_importance(args.query, f)
        texts = fill_slots(args.query, slots, imp)
        render(img, slots, texts).save(f"{args.out}/{r + 1}_{stem}.png")
        json.dump({"query": args.query, "template": f, "score": round(score, 4),
                   "word_importance": [(w, round(v, 4)) for w, v in imp],
                   "slots": [{"bbox": s["bbox"], "font_px": s["line_h"], "color": s["color"], "stroke": s["stroke"],
                              "lines": s["lines"], "prefill": t} for s, t in zip(slots, texts)]},
                  open(f"{args.out}/{r + 1}_{stem}.json", "w"), indent=1)
        print(f"{r + 1}. {f} score={score:.3f} slots={len(slots)} prefill={texts}")


if __name__ == "__main__":
    main()
