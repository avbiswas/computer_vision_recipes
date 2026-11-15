"""Fine-tune CLIP on text-free meme templates <-> MemeCap captions, and evaluate text->meme retrieval.

  python train_clip.py --eval-only                       # zero-shot on original + clean test images
  python train_clip.py --epochs 10 --out runs/clip_b16   # fine-tune on clean trainval, eval each epoch

Images: raw/memes (original, with text) and out/clean (text removed by remove_text.py).
Queries per test meme: meaning (meme_captions), title, literal (img_captions). Gallery = 559 test images. 500 trainval memes are held out as val for checkpoint selection.
"""
import argparse, json, math, os, random, time
import numpy as np, torch, torch.nn.functional as F, open_clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
FIELDS = {"meaning": "meme_captions", "title": "title", "literal": "img_captions"}


def load(split):
    return json.load(open(f"{ROOT}/raw/meme-cap/data/memes-{split}.json"))


def texts_of(m, kinds):
    out = []
    for k in kinds:
        v = m[FIELDS[k]]
        out += [t.strip() for t in (v if isinstance(v, list) else [v]) if isinstance(t, str) and t.strip()]
    return out


class Pairs(Dataset):
    """One item per image; a random caption is drawn each time."""
    def __init__(self, memes, img_dir, pre, kinds):
        self.items = [(f"{img_dir}/{m['img_fname']}", texts_of(m, kinds)) for m in memes]
        self.items = [(p, t) for p, t in self.items if t and os.path.exists(p)]
        self.pre = pre

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, ts = self.items[i]
        return self.pre(Image.open(p).convert("RGB")), random.choice(ts)


@torch.no_grad()
def evaluate(model, pre, tok, memes, img_dir):
    model.eval()
    ims = []
    for i in range(0, len(memes), 64):
        b = torch.stack([pre(Image.open(f"{img_dir}/{m['img_fname']}").convert("RGB")) for m in memes[i:i + 64]])
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            ims.append(model.encode_image(b.to(dev)).float())
    ims = F.normalize(torch.cat(ims), dim=-1)
    res = {}
    for kind in FIELDS:
        texts, owners = [], []
        for j, m in enumerate(memes):
            for t in texts_of(m, [kind]):
                texts.append(t); owners.append(j)
        txt = []
        for i in range(0, len(texts), 256):
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                txt.append(model.encode_text(tok(texts[i:i + 256]).to(dev)).float())
        sims = F.normalize(torch.cat(txt), dim=-1) @ ims.T
        target = sims[torch.arange(len(owners)), torch.tensor(owners)]
        ranks = (sims > target[:, None]).sum(1).cpu().numpy() + 1
        res[kind] = {**{f"R@{k}": round(float((ranks <= k).mean() * 100), 2) for k in (1, 5, 10)},
                     "MedR": float(np.median(ranks)), "n": len(ranks)}
    model.train()
    return res


def fmt(r):
    return "  ".join(f"{k}: R@1={v['R@1']:.1f} R@5={v['R@5']:.1f} R@10={v['R@10']:.1f} MedR={v['MedR']:.0f}" for k, v in r.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-16:openai")
    ap.add_argument("--train-fields", default="meaning,title,literal")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clean-dir", default=f"{ROOT}/out/clean")
    ap.add_argument("--out", default=f"{ROOT}/runs/clip")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)

    name, pt = args.model.split(":")
    model, pre_train, pre = open_clip.create_model_and_transforms(name, pretrained=pt)
    tok = open_clip.get_tokenizer(name)
    model = model.to(dev)
    test = load("test")
    os.makedirs(args.out, exist_ok=True)
    log = {"args": vars(args)}

    log["zeroshot_original"] = evaluate(model, pre, tok, test, f"{ROOT}/raw/memes")
    log["zeroshot_clean"] = evaluate(model, pre, tok, test, args.clean_dir)
    print("zero-shot original |", fmt(log["zeroshot_original"]))
    print("zero-shot clean    |", fmt(log["zeroshot_clean"]))
    if args.eval_only:
        json.dump(log, open(f"{args.out}/results.json", "w"), indent=1)
        return

    tv = load("trainval"); random.Random(0).shuffle(tv)
    val, train = tv[:500], tv[500:]   # val picks the checkpoint; test is only reported
    ds = Pairs(train, args.clean_dir, pre_train, args.train_fields.split(","))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(dl); warm = max(1, total // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    print(f"train pairs={len(ds)} steps={total}")
    best, log["epochs"] = -1, []
    model.train()
    for ep in range(args.epochs):
        t0, losses = time.time(), []
        for ims, txts in dl:
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                fi = F.normalize(model.encode_image(ims.to(dev, non_blocking=True)), dim=-1)
                ft = F.normalize(model.encode_text(tok(list(txts)).to(dev)), dim=-1)
                logits = model.logit_scale.exp() * fi @ ft.T
                y = torch.arange(len(ims), device=dev)
                loss = (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0, math.log(100))
            losses.append(loss.item())
        rv = evaluate(model, pre, tok, val, args.clean_dir)
        r = evaluate(model, pre, tok, test, args.clean_dir)
        log["epochs"].append({"epoch": ep + 1, "loss": float(np.mean(losses)), "val_clean": rv, "test_clean": r})
        print(f"ep {ep + 1} loss={np.mean(losses):.3f} {time.time() - t0:.0f}s | val meaning R@10={rv['meaning']['R@10']:.1f} | test", fmt(r))
        score = rv["meaning"]["R@10"]
        if score > best:
            best = score
            torch.save(model.state_dict(), f"{args.out}/best.pt")
            log["best"] = {"epoch": ep + 1, "val_clean": rv, "test_clean": r}
        json.dump(log, open(f"{args.out}/results.json", "w"), indent=1)
    print("best:", log["best"]["epoch"], fmt(log["best"]["test_clean"]))


if __name__ == "__main__":
    main()
