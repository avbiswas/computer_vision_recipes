"""Preview the real environment at training resolution (64x64), upscaled for viewing.

  python preview_env.py   -> preview_sheet.png (8 rooms) + preview_episode.mp4 (with keys)
"""
import numpy as np
from PIL import Image, ImageDraw
from env import FireflyRoom
from record import lamp_policy
from videoutil import draw_keys, upscale, write_mp4, font, BG

S = 8  # 64 -> 512 px
env, rng = FireflyRoom(res=64, seed=7), np.random.default_rng(7)
tiles = []
for i in range(8):
    env.reset()
    pol = lamp_policy(env, rng)
    for _ in range(int(rng.integers(0, 15))):
        env.step(next(pol))
    tiles.append(upscale(env.render(), S // 2))           # 256 px tiles
sheet = Image.new("RGB", (4 * 256 + 5 * 8, 2 * 256 + 3 * 8), BG)
for i, t in enumerate(tiles):
    sheet.paste(t, (8 + (i % 4) * 264, 8 + (i // 4) * 264))
sheet.save("preview_sheet.png")

env, rng = FireflyRoom(res=64, seed=3, n_flies=2), np.random.default_rng(3)
frames, pol, a = [], lamp_policy(env, rng), 0
img = env.reset()
for t in range(200):
    canvas = Image.new("RGB", (64 * S + 32, 64 * S + 170), BG)
    canvas.paste(upscale(img, S), (16, 16))
    keys = draw_keys(a, size=44)
    canvas.paste(keys, ((canvas.width - keys.width) // 2, 64 * S + 26))
    frames.append(np.asarray(canvas))
    a = next(pol)
    img = env.step(a)
write_mp4(frames, "preview_episode.mp4", fps=12)
print("wrote preview_sheet.png, preview_episode.mp4")
