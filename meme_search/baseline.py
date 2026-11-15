"""Zero-shot text->meme retrieval on the MemeCap test set (559 memes).

Query types per meme:
  title    - the Reddit post title
  literal  - img_captions (what the image shows)
  meaning  - meme_captions (what the meme means)
Gallery = all 559 test images. Reports Recall@1/5/10 and median rank.
"""
import argparse, json, os
import numpy as np, torch, open_clip
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = json.load(open(f"{ROOT}/raw/meme-cap/data/memes-test.json"))
IMG_DIR = f"{ROOT}/raw/memes"
dev = "mps" if torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def embed(model_name, pretrained):
    model, _, pre = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_name)
    model = model.to(dev).eval()
    ims = []
    for i in range(0, len(DATA), 32):
        batch = torch.stack([pre(Image.open(f"{IMG_DIR}/{m['img_fname']}").convert("RGB")) for m in DATA[i:i + 32]])
        ims.append(model.encode_image(batch.to(dev)).float().cpu())
    ims = torch.nn.functional.normalize(torch.cat(ims), dim=-1)

    def enc_text(texts):
        out = []
        for i in range(0, len(texts), 128):
            out.append(model.encode_text(tok(texts[i:i + 128]).to(dev)).float().cpu())
        return torch.nn.functional.normalize(torch.cat(out), dim=-1)

    queries = {}
    for kind, field in [("title", "title"), ("literal", "img_captions"), ("meaning", "meme_captions")]:
        texts, owners = [], []
        for j, m in enumerate(DATA):
            vals = m[field] if isinstance(m[field], list) else [m[field]]
            for t in vals:
                if t and t.strip():
                    texts.append(t.strip()); owners.append(j)
        queries[kind] = (texts, np.array(owners), enc_text(texts))
    return ims, queries


def metrics(ims, q):
    texts, owners, txt = q
    sims = (txt @ ims.T).numpy()
    order = np.argsort(-sims, axis=1)
    ranks = np.array([np.where(order[i] == owners[i])[0][0] + 1 for i in range(len(owners))])
    r = {f"R@{k}": float((ranks <= k).mean() * 100) for k in (1, 5, 10)}
    r["MedR"] = float(np.median(ranks)); r["n"] = len(ranks)
    return r, order, ranks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="ViT-B-16:openai,ViT-B-16-SigLIP2:webli,ViT-L-14:openai")
    args = ap.parse_args()
    os.makedirs(f"{ROOT}/results", exist_ok=True)
    allres, examples = {}, {}
    for spec in args.models.split(","):
        name, pt = spec.split(":")
        ims, queries = embed(name, pt)
        allres[spec] = {}
        for kind, q in queries.items():
            r, order, ranks = metrics(ims, q)
            allres[spec][kind] = r
            print(f"{spec:28s} {kind:8s} " + "  ".join(f"{k}={v:.1f}" if k != "n" else f"n={v}" for k, v in r.items()))
            # store top-5 for the first 40 queries for inspection
            examples.setdefault(spec, {})[kind] = [
                {"query": q[0][i], "target": DATA[q[1][i]]["img_fname"], "rank": int(ranks[i]),
                 "top5": [DATA[j]["img_fname"] for j in order[i][:5]]} for i in range(0, len(ranks), max(1, len(ranks) // 40))]
    json.dump(allres, open(f"{ROOT}/results/baseline.json", "w"), indent=1)
    json.dump(examples, open(f"{ROOT}/results/baseline_examples.json", "w"), indent=1)
