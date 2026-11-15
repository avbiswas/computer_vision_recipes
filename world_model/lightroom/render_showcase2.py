"""Showcase video v2 (1920x1080): scene on the RIGHT, the model's internals small on the LEFT,
no text labels.

  left:  16x16 codebook-id grid (changed ids glow faintly), memory strip (anchor + strided past),
         controller keys
  right: the dreamed room, large

  python render_showcase2.py --script still --seed 7 --out runs/tweet/still_seed7.mp4
"""
import argparse
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from env import FireflyRoom
from dream import load, Dream
from vqvae import VQVAE
from videoutil import draw_keys, write_mp4

BG = (8, 9, 13)
SCRIPTS = {
    "still": [(0, 140)],                                                   # the flashlight never moves
    "drive": [(0, 6), (4, 7), (1, 8), (3, 10), (1, 6), (0, 6), (4, 14), (1, 10), (2, 5), (3, 8),
              (1, 12), (0, 8), (4, 9), (1, 8), (3, 12), (0, 6)],
}


def font(size):
    for f in ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/System/Library/Fonts/Menlo.ttc"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def up(img, size):
    return Image.fromarray(img).resize((size, size), Image.NEAREST)


def keys_no_label(action, size):
    k = draw_keys(action, size=size)
    return k.crop((0, 0, k.width, int(k.height - size * 0.5)))           # cut off the text line


def user_icon(size=110, active=False):
    """A little person-at-a-keyboard glyph: makes 'this is the user's input' readable at a glance."""
    col = (242, 184, 75) if active else (150, 156, 168)
    img = Image.new("RGB", (size, size), BG)
    d = ImageDraw.Draw(img)
    cx = size // 2
    d.ellipse([cx - 13, 6, cx + 13, 32], outline=col, width=3)                 # head
    d.arc([cx - 30, 34, cx + 30, 86], 200, 340, fill=col, width=3)             # shoulders
    kx0, ky0, kx1, ky1 = 8, 66, size - 8, size - 10                            # keyboard
    d.rounded_rectangle([kx0, ky0, kx1, ky1], radius=5, outline=col, width=2)
    for r in range(3):
        for c in range(7):
            x = kx0 + 7 + c * ((kx1 - kx0 - 14) / 7)
            y = ky0 + 6 + r * 9
            d.rectangle([x, y, x + 7, y + 4], fill=col)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
    ap.add_argument("--wm", default="runs/rope1/ckpts/wm_010000.pt")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--script", default="still", choices=list(SCRIPTS))
    ap.add_argument("--out", default="runs/tweet/still.mp4")
    ap.add_argument("--fps", type=int, default=12)
    a = ap.parse_args()

    vq, wm = load(a.vqvae, a.wm)
    env = FireflyRoom(res=64, seed=a.seed, n_boxes=3, n_flies=2)
    frames, acts = [env.reset()], []
    for _ in range(7):
        acts.append(0); frames.append(env.step(0))
    dream = Dream(vq, wm, frames, acts, passes=1, temperature=0)
    with torch.no_grad():
        ids = torch.arange(512, device="cuda")[:, None, None].expand(512, 16, 16)
        code_rgb = torch.cat([VQVAE.to_image(vq.decode(ids[i:i + 64])) for i in range(0, 512, 64)])
        code_rgb = code_rgb.float().mean((1, 2)).cpu().numpy()

    W, H = 1920, 1080
    S, sx, sy = 960, 1920 - 960 - 60, 60                 # scene: right, large
    CELL, gx, gy = 42, 60 + (780 - 16 * 42) // 2, 60     # id grid: left
    TH, mgap = 70, 8                                      # memory thumbnails
    mx, my = 60 + (780 - 9 * (TH + mgap) + mgap) // 2, gy + 16 * CELL + 30
    f_id = font(15)
    prev = dream.hist[-1].cpu().numpy()
    actions = [x for act, n in SCRIPTS[a.script] for x in [act] * n]
    video = []
    for action in actions:
        img = dream.step(action)
        tok = dream.hist[-1].cpu().numpy()
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)

        # scene (right)
        canvas.paste(up(img, S), (sx, sy))

        # codebook-id grid (left, muted)
        for i in range(256):
            r, c = divmod(i, 16)
            x, y = gx + c * CELL, gy + r * CELL
            col = code_rgb[tok[i]]
            changed = tok[i] != prev[i]
            fill = tuple(int(12 + 0.22 * v) for v in col)
            d.rectangle([x, y, x + CELL - 2, y + CELL - 2], fill=fill,
                        outline=(120, 92, 40) if changed else None)
            tc = (200, 150, 70) if changed else tuple(int(60 + 0.35 * v) for v in col)
            d.text((x + (CELL - 2) / 2, y + (CELL - 2) / 2), str(tok[i]), fill=tc, font=f_id, anchor="mm")
        prev = tok

        # memory strip (left, under the grid): anchor + strided past frames
        t = len(dream.hist) - 1
        mem = [dream.anchor] + [dream.hist[max(t - o, 0)] for o in wm.offsets[:-1]]
        with torch.no_grad():
            thumbs = VQVAE.to_image(vq.decode(torch.stack(mem).view(-1, 16, 16))).cpu().numpy()
        for j, timg in enumerate(thumbs):
            x = mx + j * (TH + mgap)
            canvas.paste(up(timg, TH), (x, my))
            if j == 0:
                d.rectangle([x - 2, my - 2, x + TH + 1, my + TH + 1], outline=(90, 70, 35), width=1)

        # controller (left, bottom): user icon + arrow keys inside a subtle panel
        keys = keys_no_label(action, size=38)
        icon = user_icon(110, active=action != 0)
        pw = icon.width + 40 + keys.width
        px = 60 + (780 - pw) // 2
        py = my + TH + 40
        d.rounded_rectangle([px - 24, py - 20, px + pw + 24, py + max(icon.height, keys.height) + 20],
                            radius=14, outline=(60, 50, 32) if action == 0 else (150, 112, 45), width=2)
        canvas.paste(icon, (px, py))
        canvas.paste(keys, (px + icon.width + 40, py + (icon.height - keys.height) // 2))
        video.append(np.asarray(canvas))
    write_mp4(video, a.out, fps=a.fps)
    print("wrote", a.out, len(video), "frames")


if __name__ == "__main__":
    main()
