"""Build the cv.neuralavb.com site into dist/, and upload large files to R2.

Two kinds of content:
  * text (HTML / JS / small JSON) is in git. CI (Cloudflare Workers Builds) runs this script and `wrangler deploy`
    on every push, so page edits go live by themselves.
  * binaries (models, images, videos, data) are git-ignored. They live in the R2 bucket and pages load them
    from /r2/<slug>/<key>. Upload them by hand after you change them: `python site/build.py --upload-r2`.

Every project that wants a page has a `web/site.json` next to its `web/index.html`:

  {
    "slug": "meme-search",                       # URL: cv.neuralavb.com/meme-search/
    "title": "Meme search", "blurb": "...", "tags": ["CLIP"], "order": 1,
    "card": "../../images/meme.png",              # landing-page picture -> R2 <slug>/card.png
    "files": ["index.html", "app.js"],            # in git; copied to dist/<slug>/
    "r2": {"data/embeddings.bin": "data/embeddings.bin",     # R2 key under <slug>/  <-  path relative to web/
           "data/templates/": "data/templates/"}             # a key ending in / uploads a whole folder
  }

  python site/build.py                       # -> dist/   (this is what CI runs)
  python site/build.py --upload-r2           # upload new/changed R2 files (needs `npx wrangler login`)
  python site/build.py --upload-r2 --local   # same, into wrangler's local R2, for `npx wrangler dev`
  npx wrangler deploy                        # what CI runs after the build
"""
import argparse, hashlib, html, json, mimetypes, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE, DIST = ROOT / "site", ROOT / "dist"
BUCKET = "cv-speedrun-assets"
MAX_ASSET = 25 * 1024 * 1024
SKIP = {".venv", "node_modules", ".uvcache", "third_party", "dist", ".git"}
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("text/javascript", ".mjs")


def find_projects():
    out = []
    for p in ROOT.rglob("web/site.json"):
        if SKIP & set(p.relative_to(ROOT).parts):
            continue
        cfg = json.loads(p.read_text())
        cfg["_web"] = p.parent
        out.append(cfg)
    return sorted(out, key=lambda c: (c.get("order", 99), c["slug"]))


def copy(src: Path, dst: Path):
    if not src.exists():
        sys.exit(f"missing: {src.relative_to(ROOT)}  (files in \"files\" must be committed to git)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def git_ignored(paths):
    """Paths git would not commit (so CI would not see them). Empty outside a git checkout."""
    try:
        r = subprocess.run(["git", "check-ignore", "--stdin"], input="\n".join(map(str, paths)),
                           capture_output=True, text=True, cwd=ROOT)
    except FileNotFoundError:
        return []
    return [l for l in r.stdout.splitlines() if l]


def card_html(c):
    tags = "".join(f"<span>{html.escape(t)}</span>" for t in c.get("tags", []))
    img = f'<img src="/r2/{c["slug"]}/card.png" alt="" loading="lazy">' if c.get("card") else ""
    return (f'<a class="proj" href="/{c["slug"]}/">{img}<div class="proj-body">'
            f'<h2>{html.escape(c["title"])}</h2><p>{html.escape(c["blurb"])}</p>'
            f'<div class="tags">{tags}</div></div></a>')


def build():
    projects = find_projects()
    srcs = [SITE / n for n in ("index.html", "site.css", "404.html", "favicon.svg")]
    srcs += [c["_web"] / f for c in projects for f in c.get("files", [])]
    ignored = git_ignored(srcs)
    if ignored:
        sys.exit("git-ignored, so CI can't see them (move to \"r2\"): " + ", ".join(ignored))

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for name in ("site.css", "404.html", "favicon.svg"):
        copy(SITE / name, DIST / name)
    for c in projects:
        for f in c.get("files", []):
            copy(c["_web"] / f, DIST / c["slug"] / f)
    page = (SITE / "index.html").read_text().replace("<!--PROJECTS-->", "\n".join(card_html(c) for c in projects))
    (DIST / "index.html").write_text(page)

    files = [p for p in DIST.rglob("*") if p.is_file()]
    big = [str(p.relative_to(DIST)) for p in files if p.stat().st_size > MAX_ASSET]
    if big:
        sys.exit("over the 25 MiB asset limit (move to \"r2\"): " + ", ".join(big))
    print(f"dist/: {len(projects)} projects, {len(files)} files, {sum(p.stat().st_size for p in files) / 1e3:.0f} kB")
    for c in projects:
        print(f"  /{c['slug']}/  {c['title']}")


# ------------------------------------------------------------------ R2
def r2_files(c):
    """(key, local path) for every R2 file of a project, folders expanded."""
    web, out = c["_web"], []
    if c.get("card"):
        out.append((f"{c['slug']}/card.png", (web / c["card"]).resolve()))
    for key, src in c.get("r2", {}).items():
        path = (web / src).resolve()
        if not path.exists():
            sys.exit(f"missing R2 source: {path}")
        if key.endswith("/"):
            out += [(f"{c['slug']}/{key}{p.relative_to(path).as_posix()}", p) for p in sorted(path.rglob("*")) if p.is_file()]
        else:
            out.append((f"{c['slug']}/{key}", path))
    return out


def sha(p: Path):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def upload_r2(local: bool, workers: int = 12):
    state_f = SITE / (".r2_local.json" if local else ".r2_uploaded.json")
    state = json.loads(state_f.read_text()) if state_f.exists() else {}
    todo = []
    for c in find_projects():
        for key, path in r2_files(c):
            d = sha(path)
            if state.get(key) != d:
                todo.append((key, path, d))
    print(f"{len(todo)} files to upload ({sum(p.stat().st_size for _, p, _ in todo) / 1e6:.0f} MB)"
          + (" to local R2" if local else f" to {BUCKET}"))

    def put(item):
        key, path, d = item
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        r = subprocess.run(["npx", "wrangler", "r2", "object", "put", f"{BUCKET}/{key}", "--file", str(path),
                            "--local" if local else "--remote", "--content-type", ctype],
                           capture_output=True, text=True, cwd=ROOT)
        return key, d, r.returncode, r.stderr[-400:]

    failed = 0
    with ThreadPoolExecutor(workers) as ex:
        for i, (key, d, rc, err) in enumerate(ex.map(put, todo), 1):
            if rc:
                failed += 1
                print(f"  FAILED {key}: {err.strip()}")
                continue
            state[key] = d
            if i % 200 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
                state_f.write_text(json.dumps(state, indent=0, sort_keys=True))
    state_f.write_text(json.dumps(state, indent=0, sort_keys=True))
    if failed:
        sys.exit(f"{failed} uploads failed; run again to retry them")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload-r2", action="store_true", help="upload new/changed R2 files to the bucket")
    ap.add_argument("--local", action="store_true", help="with --upload-r2: wrangler's local R2 (for `wrangler dev`)")
    a = ap.parse_args()
    upload_r2(a.local) if a.upload_r2 else build()
