"""Live view: drive the flashlight inside the world model's dream, with its internals on the left.

  python play_live.py                         # best model, 1440x810 window
  python play_live.py --scale 1.0             # full 1920x1080
  python play_live.py --fullscreen

Layout (same as the showcase videos): dream on the right; codebook-id grid, memory strip and
your controller on the left. No real room is shown: everything you see is generated.

Keys: arrows drive · space re-sync from a real frame · n new room · c record MP4 · f fullscreen · q quit
"""
import argparse
import os
import time
import numpy as np
import pygame
import torch
from env import FireflyRoom
from dream import load, Dream
from vqvae import VQVAE, DEVICE
from videoutil import draw_keys, write_mp4
from render_showcase2 import user_icon

KEYMAP = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
BG = (8, 9, 13)


def pil_to_surface(img):
    return pygame.image.frombytes(img.tobytes(), img.size, img.mode)


def np_to_surface(arr, size):
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(arr.swapaxes(0, 1)))
    return pygame.transform.scale(surf, (size, size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
    ap.add_argument("--wm", default="runs/best/world_model_rope.pt")
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--frames", type=int, default=0, help="quit after N frames (testing)")
    a = ap.parse_args()

    k = a.scale
    W, H = int(1920 * k), int(1080 * k)
    S, sx, sy = int(960 * k), int((1920 - 960 - 60) * k), int(60 * k)          # dream (right)
    CELL = max(int(42 * k), 16)
    gx, gy = int(60 * k + (780 * k - 16 * CELL) / 2), int(60 * k)               # id grid (left, big)
    TH, mg = int(70 * k), int(8 * k)
    mx, my = int(60 * k + (780 * k - 9 * (TH + mg) + mg) / 2), gy + 16 * CELL + int(30 * k)

    vq, wm = load(a.vqvae, a.wm)
    with torch.no_grad():
        ids = torch.arange(512, device=DEVICE)[:, None, None].expand(512, 16, 16)
        code_rgb = torch.cat([VQVAE.to_image(vq.decode(ids[i:i + 64])) for i in range(0, 512, 64)])
        code_rgb = code_rgb.float().mean((1, 2)).cpu().numpy()

    pygame.init()
    pygame.display.set_caption("world model - live")
    flags = pygame.FULLSCREEN if a.fullscreen else 0
    screen = pygame.display.set_mode((W, H), flags)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,dejavusansmono,monospace", max(int(15 * k), 9))

    # pre-render every codebook id twice (normal / changed) and the controller states
    fills = [tuple(int(12 + 0.22 * v) for v in code_rgb[c]) for c in range(512)]
    id_norm = [font.render(str(c), True, tuple(int(60 + 0.35 * v) for v in code_rgb[c])) for c in range(512)]
    id_hot = [font.render(str(c), True, (200, 150, 70)) for c in range(512)]
    key_surf = []
    for act in range(5):
        kimg = draw_keys(act, size=int(38 * k))
        key_surf.append(pil_to_surface(kimg.crop((0, 0, kimg.width, int(kimg.height - 19 * k)))))
    icons = [pil_to_surface(user_icon(int(110 * k), active)) for active in (False, True)]

    env = FireflyRoom(res=64, seed=a.seed)
    history, past = [], []

    def new_room():
        history.clear(), past.clear()
        history.append(env.reset())
        for _ in range(7):
            past.append(0); history.append(env.step(0))

    def sync():
        return Dream(vq, wm, history[-8:], past[-7:], passes=1, temperature=0)

    new_room()
    dream = sync()
    prev = dream.hist[-1].cpu().numpy()
    recording, rec, n = False, [], 0
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_q, pygame.K_ESCAPE)):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                dream = sync()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_n:
                new_room(); dream = sync()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_f:
                pygame.display.toggle_fullscreen()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                recording = not recording
                if not recording and rec:
                    os.makedirs("runs/play", exist_ok=True)
                    path = f"runs/play/live_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    write_mp4(rec, path, fps=a.fps); print("saved", path); rec = []
        held = pygame.key.get_pressed()
        action = next((v for key, v in KEYMAP.items() if held[key]), 0)

        img = dream.step(action)                      # the dream
        past.append(action); history.append(env.step(action))   # hidden real room (only for re-sync)
        history[:] = history[-40:]; past[:] = past[-40:]
        tok = dream.hist[-1].cpu().numpy()

        screen.fill(BG)
        screen.blit(np_to_surface(img, S), (sx, sy))
        for i in range(256):
            r, c = divmod(i, 16)
            x, y = gx + c * CELL, gy + r * CELL
            hot = tok[i] != prev[i]
            pygame.draw.rect(screen, fills[tok[i]], (x, y, CELL - 2, CELL - 2))
            if hot:
                pygame.draw.rect(screen, (120, 92, 40), (x, y, CELL - 2, CELL - 2), 1)
            s = (id_hot if hot else id_norm)[tok[i]]
            screen.blit(s, (x + (CELL - 2 - s.get_width()) // 2, y + (CELL - 2 - s.get_height()) // 2))
        prev = tok
        t = len(dream.hist) - 1
        mem = [dream.anchor] + [dream.hist[max(t - o, 0)] for o in wm.offsets[:-1]]
        with torch.no_grad():
            thumbs = VQVAE.to_image(vq.decode(torch.stack(mem).view(-1, 16, 16))).cpu().numpy()
        for j, timg in enumerate(thumbs):
            x = mx + j * (TH + mg)
            screen.blit(np_to_surface(timg, TH), (x, my))
            if j == 0:
                pygame.draw.rect(screen, (90, 70, 35), (x - 2, my - 2, TH + 4, TH + 4), 1)
        icon, keys = icons[action != 0], key_surf[action]
        pw = icon.get_width() + int(40 * k) + keys.get_width()
        px, py = int(60 * k + (780 * k - pw) / 2), my + TH + int(40 * k)
        ph = max(icon.get_height(), keys.get_height())
        pygame.draw.rect(screen, (150, 112, 45) if action else (60, 50, 32),
                         (px - int(24 * k), py - int(20 * k), pw + int(48 * k), ph + int(40 * k)), 2,
                         border_radius=int(14 * k))
        screen.blit(icon, (px, py))
        screen.blit(keys, (px + icon.get_width() + int(40 * k), py + (ph - keys.get_height()) // 2))
        if recording:
            pygame.draw.circle(screen, (220, 60, 60), (W - int(30 * k), int(30 * k)), int(8 * k))
            rec.append(pygame.surfarray.array3d(screen).swapaxes(0, 1).copy())
        pygame.display.flip()
        clock.tick(a.fps)
        n += 1
        if a.frames and n >= a.frames:
            print(f"ran {n} frames at {clock.get_fps():.1f} fps")
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()
