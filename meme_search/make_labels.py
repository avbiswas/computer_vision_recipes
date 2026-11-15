"""Zero-shot topic labels for every meme image + MemeCap metadata, exported for import into Voxel51.

Uses CLIP (open_clip) with prompt ensembling. Images are keyed by filename everywhere.
Outputs (in labels/):
  labels.csv       one row per image: filename, split, topic, confidence, runner-up, captions
  labels.json      FiftyOne image-classification format: {"classes": [...], "labels": {stem: {...}}}
  memes_full.json  everything per image (captions, metaphors, all topic scores)
"""
import argparse, csv, json, os
import torch, open_clip
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = f"{ROOT}/raw/memes"
OUT = f"{ROOT}/labels"
dev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

TOPICS = {
    "work": "work, jobs, bosses and office life",
    "school": "school, college, exams and teachers",
    "gaming": "video games and gamers",
    "relationships": "dating, relationships and crushes",
    "politics": "politics, governments and world news",
    "animals": "animals and pets",
    "sports": "sports and athletes",
    "programming": "programming, computers and technology",
    "family": "parents, kids and family life",
    "food": "food and eating",
    "movies_tv": "movies, TV shows and celebrities",
    "history": "history and historical events",
    "money": "money, prices and being broke",
    "mental_health": "sadness, anxiety and depression",
}
TEMPLATES = ["a meme about {}.", "a funny meme about {}.", "an internet meme about {}.", "a reaction image about {}."]


def load_meta():
    meta = {}
    for split, fn in [("trainval", "memes-trainval.json"), ("test", "memes-test.json")]:
        for m in json.load(open(f"{ROOT}/raw/meme-cap/data/{fn}")):
            meta[m["img_fname"]] = {**m, "split": split}
    return meta


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-L-14:openai")
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()
    name, pt = args.model.split(":")
    model, _, pre = open_clip.create_model_and_transforms(name, pretrained=pt)
    tok = open_clip.get_tokenizer(name)
    model = model.to(dev).eval()

    keys = list(TOPICS)
    cls = []
    for k in keys:
        e = model.encode_text(tok([t.format(TOPICS[k]) for t in TEMPLATES]).to(dev)).float()
        cls.append(torch.nn.functional.normalize(e, dim=-1).mean(0))
    cls = torch.nn.functional.normalize(torch.stack(cls), dim=-1)

    files = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    probs = []
    for i in range(0, len(files), args.bs):
        ims = torch.stack([pre(Image.open(f"{IMG_DIR}/{f}").convert("RGB")) for f in files[i:i + args.bs]])
        e = torch.nn.functional.normalize(model.encode_image(ims.to(dev)).float(), dim=-1)
        probs.append((100 * e @ cls.T).softmax(-1).cpu())
        print(f"\r{min(i + args.bs, len(files))}/{len(files)}", end="", flush=True)
    probs = torch.cat(probs)
    print()

    meta = load_meta()
    os.makedirs(OUT, exist_ok=True)
    rows, fo_labels, full = [], {}, {}
    for f, p in zip(files, probs):
        top = p.argsort(descending=True)[:2].tolist()
        m = meta.get(f, {})
        split = m.get("split", "unlabeled")
        rec = {
            "filename": f, "split": split,
            "topic": keys[top[0]], "topic_conf": round(p[top[0]].item(), 4),
            "topic2": keys[top[1]], "topic2_conf": round(p[top[1]].item(), 4),
            "title": m.get("title", ""),
            "img_captions": " | ".join(m.get("img_captions", [])),
            "meme_captions": " | ".join(m.get("meme_captions", [])),
        }
        rows.append(rec)
        fo_labels[os.path.splitext(f)[0]] = {"label": rec["topic"], "confidence": rec["topic_conf"]}
        full[f] = {**rec, "img_captions": m.get("img_captions", []), "meme_captions": m.get("meme_captions", []),
                   "metaphors": m.get("metaphors", []), "url": m.get("url", ""),
                   "topic_scores": {k: round(v, 4) for k, v in zip(keys, p.tolist())}}

    with open(f"{OUT}/labels.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump({"classes": keys, "labels": fo_labels}, open(f"{OUT}/labels.json", "w"), indent=1)
    json.dump(full, open(f"{OUT}/memes_full.json", "w"), indent=1)

    from collections import Counter
    print("topic counts:", Counter(r["topic"] for r in rows).most_common())
    print("split counts:", Counter(r["split"] for r in rows).most_common())


if __name__ == "__main__":
    main()
