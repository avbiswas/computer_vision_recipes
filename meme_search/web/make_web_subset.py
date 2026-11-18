"""Pick a smaller, diverse set of templates for the website (fewer files to upload and serve).

  python web/make_web_subset.py --n 2000      # web/data -> web/data_web (templates.json, embeddings.bin, templates/)

1. Drop near-duplicates (reposts): cosine > 0.95 to a template already kept (same rule the demo uses).
2. k-means with n clusters on the CLIP image embeddings; keep the template nearest each cluster centre.
   This spreads the subset over every kind of meme, instead of a random slice.
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--src", default=str(HERE / "data"))
ap.add_argument("--out", default=str(HERE / "data_web"))
a = ap.parse_args()
src, out = Path(a.src), Path(a.out)

meta = json.loads((src / "templates.json").read_text())
emb = np.fromfile(src / "embeddings.bin", np.float32).reshape(len(meta), -1)

keep = []
for i in range(len(meta)):
    if not keep or (emb[keep] @ emb[i]).max() <= 0.95:
        keep.append(i)
print(f"{len(meta)} templates, {len(keep)} after removing near-duplicates")

x = emb[keep]
km = KMeans(a.n, n_init=1, random_state=0).fit(x)
pick = set()
for c in range(a.n):
    members = np.flatnonzero(km.labels_ == c)
    pick.add(keep[members[np.argmax(x[members] @ km.cluster_centers_[c])]])
pick = sorted(pick)
print(f"picked {len(pick)}")

if out.exists():
    shutil.rmtree(out)
(out / "templates").mkdir(parents=True)
(out / "templates.json").write_text(json.dumps([meta[i] for i in pick]))
emb[pick].astype(np.float32).tofile(out / "embeddings.bin")
for i in pick:
    shutil.copy2(src / "templates" / meta[i]["file"], out / "templates" / meta[i]["file"])
mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
print(f"wrote {out} ({mb:.0f} MB)")
