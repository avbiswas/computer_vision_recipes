"""Remove overlaid text from memes: Florence-2 OCR (boxes + text) -> mask -> LaMa inpaint.

Writes per image:
  out/clean/<name>.png        text-free image
  out/layout/<name>.json      text, quad box, font size, colors (to re-render text later)
  out/compare/<name>.jpg      original | mask | inpainted   (with --compare)
Needs models/big-lama.pt (TorchScript LaMa).
"""
import argparse, json, os, random, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Florence2ForConditionalGeneration

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = f"{ROOT}/raw/memes"
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
TASK = "<OCR_WITH_REGION>"
if dev == "cuda" and os.environ.get("GPU_MEM_FRAC"):  # share a GPU safely: OOM ourselves, never the other job
    torch.cuda.set_per_process_memory_fraction(float(os.environ["GPU_MEM_FRAC"]))


class OCR:
    def __init__(self, name="florence-community/Florence-2-large"):
        self.proc = AutoProcessor.from_pretrained(name)
        self.model = Florence2ForConditionalGeneration.from_pretrained(name, dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()

    @torch.no_grad()
    def __call__(self, ims):
        """Batched OCR: list of PIL images -> list of [(text, quad), ...]."""
        x = self.proc(text=[TASK] * len(ims), images=ims, return_tensors="pt").to(dev)
        x["pixel_values"] = x["pixel_values"].to(self.model.dtype)
        ids = self.model.generate(**x, max_new_tokens=1024, num_beams=3, do_sample=False)
        res = []
        for raw, im in zip(self.proc.batch_decode(ids, skip_special_tokens=False), ims):
            raw = raw.replace("<pad>", "")
            out = self.proc.post_process_generation(raw, task=TASK, image_size=im.size)[TASK]
            res.append([(t.replace("</s>", "").replace("<s>", "").strip(), q) for t, q in zip(out["labels"], out["quad_boxes"])])
        return res


class LaMa:
    def __init__(self, path=f"{ROOT}/models/big-lama.pt"):
        self.model = torch.jit.load(path, map_location="cpu").eval()
        self.dev = "cpu"
        try:  # TorchScript LaMa sometimes has ops MPS lacks; fall back to CPU if so
            m = self.model.to(dev); m(torch.zeros(1, 3, 512, 512, device=dev), torch.zeros(1, 1, 512, 512, device=dev))
            self.model, self.dev = m, dev
        except Exception:
            self.model = self.model.to("cpu")

    def __call__(self, im, mask, side=512):
        """Fit into a fixed side x side canvas (one input shape -> TorchScript compiles once),
        inpaint, then paste only the masked pixels back into the full-res image."""
        s = side / max(im.size)
        w, h = max(1, round(im.width * s)), max(1, round(im.height * s))
        img = torch.from_numpy(np.asarray(im.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255).permute(2, 0, 1)[None]
        m = torch.from_numpy((np.asarray(mask.resize((w, h), Image.NEAREST)) > 0).astype(np.float32))[None, None]
        img = F.pad(img, (0, side - w, 0, side - h), mode="replicate")
        m = F.pad(m, (0, side - w, 0, side - h))
        with torch.no_grad():
            out = self.model(img.to(self.dev), m.to(self.dev))[0, :, :h, :w].clamp(0, 1)
        out = Image.fromarray((out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).resize(im.size, Image.LANCZOS)
        return Image.composite(out, im, mask)


def make_mask(size, quads, grow=0.3):
    """Fill each text quad, enlarged by grow * its height on every side (covers the black outline)."""
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    for q in quads:
        xs, ys = q[0::2], q[1::2]
        r = max(3, grow * (max(ys) - min(ys)))
        d.rectangle([min(xs) - r, min(ys) - r, max(xs) + r, max(ys) + r], fill=255)
    return mask


def layout_entry(im_np, text, q):
    xs, ys = q[0::2], q[1::2]
    x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    crop = im_np[max(y0, 0):y1 + 1, max(x0, 0):x1 + 1].reshape(-1, 3)
    fill = stroke = [255, 255, 255]
    if len(crop) > 10:  # brightest / darkest 15% of pixels ~ fill / outline color
        lum = crop.mean(1); o = np.argsort(lum); k = max(1, len(o) * 15 // 100)
        fill, stroke = crop[o[-k:]].mean(0).astype(int).tolist(), crop[o[:k]].mean(0).astype(int).tolist()
    return {"text": text, "quad": [round(v, 1) for v in q], "bbox": [x0, y0, x1, y1],
            "font_px": y1 - y0, "color": fill, "stroke": stroke}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="random sample size (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/out")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--shard", default="0/1", help="i/k: process every k-th image starting at i (run k copies in parallel)")
    ap.add_argument("--bs", type=int, default=1, help="OCR batch size (16-32 on a free A100)")
    args = ap.parse_args()
    files = sorted(os.listdir(IMG_DIR))
    if args.n:
        files = random.Random(args.seed).sample(files, args.n)
    si, sk = map(int, args.shard.split("/"))
    files = files[si::sk]
    for sub in ("clean", "layout", "compare"):
        os.makedirs(f"{args.out}/{sub}", exist_ok=True)
    ocr, lama = OCR(), LaMa()
    print(f"device={dev} lama_device={lama.dev}")
    todo = [f for f in files if not os.path.exists(f"{args.out}/layout/{os.path.splitext(f)[0]}.json")]  # resume
    print(f"{len(files) - len(todo)} done already, {len(todo)} to go")
    for i in range(0, len(todo), args.bs):
        chunk = todo[i:i + args.bs]
        ims = [Image.open(f"{IMG_DIR}/{f}").convert("RGB") for f in chunk]
        t0 = time.time()
        all_regions = ocr(ims)
        t1 = time.time()
        for f, im, regions in zip(chunk, ims, all_regions):
            stem = os.path.splitext(f)[0]
            mask = make_mask(im.size, [q for _, q in regions])
            clean = lama(im, mask) if regions else im
            clean.save(f"{args.out}/clean/{stem}.png")
            im_np = np.asarray(im)
            json.dump({"filename": f, "size": list(im.size),
                       "mask_frac": round(float((np.asarray(mask) > 0).mean()), 4),
                       "text": [layout_entry(im_np, t, q) for t, q in regions]},
                      open(f"{args.out}/layout/{stem}.json", "w"), indent=1)
            if args.compare:
                w, h = im.size
                red = Image.composite(Image.new("RGB", im.size, (255, 0, 0)), im, mask.point(lambda v: v // 2))
                grid = Image.new("RGB", (3 * w, h)); grid.paste(im, (0, 0)); grid.paste(red, (w, 0)); grid.paste(clean, (2 * w, 0))
                grid.thumbnail((2400, 2400)); grid.save(f"{args.out}/compare/{stem}.jpg", quality=88)
        print(f"[{i + len(chunk)}/{len(todo)}] ocr={(t1 - t0) / len(chunk):.2f}s/img total={(time.time() - t0) / len(chunk):.2f}s/img")

if __name__ == "__main__":
    main()
