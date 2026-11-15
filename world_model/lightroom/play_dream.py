"""Play inside the world model's dream, next to the real room.

  python play_dream.py                       # best checkpoint, both rooms side by side
  python play_dream.py --dream-only          # just the dream (the "neural game engine")

Controls
  arrow up / down      move the flashlight forward / backward
  arrow left / right   turn the flashlight
  space                re-sync: restart the dream from the real room's last frames
  n                    new random room
  d                    toggle dream-only view
  c                    start / stop recording an MP4 (saved to runs/play/)
  esc / q              quit

Both rooms get exactly the same key presses. The dream only ever sees its own
predictions after the start (or after a re-sync), so watch how long it stays true.
"""
import argparse
import os
import time
from collections import deque
import numpy as np
import pygame
from env import FireflyRoom
from dream import load, Dream
from videoutil import draw_keys, write_mp4

KEYMAP = {pygame.K_UP: 1, pygame.K_DOWN: 2, pygame.K_LEFT: 3, pygame.K_RIGHT: 4}
BG, TEXT, ACCENT = (12, 13, 17), (200, 200, 200), (242, 184, 75)


def to_surface(img_uint8, size):
    surf = pygame.surfarray.make_surface(np.ascontiguousarray(img_uint8.swapaxes(0, 1)))
    return pygame.transform.scale(surf, (size, size))  # nearest-neighbour: crisp pixels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqvae", default="runs/vqvae_ft/vqvae.pt")
    ap.add_argument("--wm", default="runs/dyn_k4/best.pt")
    ap.add_argument("--scale", type=int, default=8, help="64 px * scale per panel")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--dream-only", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    vq, wm = load(a.vqvae, a.wm)
    K, S, pad = wm.K, 64 * a.scale, 20
    env = FireflyRoom(res=64, seed=a.seed)
    history = deque(maxlen=K)          # last K real frames (for starting / re-syncing the dream)
    past_actions = deque(maxlen=K - 1)

    def new_room():
        history.clear(), past_actions.clear()
        history.append(env.reset())
        for _ in range(K - 1):         # K-1 "no input" steps to fill the dream's memory
            past_actions.append(0)
            history.append(env.step(0))

    def sync():
        return Dream(vq, wm, list(history), list(past_actions), passes=1, temperature=0)

    new_room()
    dream = sync()
    real_img = fake_img = history[-1]
    since_sync, dream_only, recording, rec_frames = 0, a.dream_only, False, []

    pygame.init()
    pygame.display.set_caption("FireflyRoom - real vs. dream")
    font = pygame.font.SysFont("menlo", 18)
    big = pygame.font.SysFont("menlo", 22)

    def layout():
        panels = 1 if dream_only else 2
        return pygame.display.set_mode((panels * S + (panels + 1) * pad, S + 210))

    screen, clock = layout(), pygame.time.Clock()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q)):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                dream, since_sync = sync(), 0
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_n:
                new_room()
                dream, since_sync = sync(), 0
                real_img = fake_img = history[-1]
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_d:
                dream_only = not dream_only
                screen = layout()
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_c:
                recording = not recording
                if not recording and rec_frames:
                    os.makedirs("runs/play", exist_ok=True)
                    path = f"runs/play/session_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    write_mp4(rec_frames, path, fps=a.fps)
                    print("saved", path)
                    rec_frames = []

        held = pygame.key.get_pressed()
        action = next((v for k, v in KEYMAP.items() if held[k]), 0)

        # one tick of both worlds with the same action
        real_img = env.step(action)
        fake_img = dream.step(action)
        past_actions.append(action)
        history.append(real_img)
        since_sync += 1

        # draw
        screen.fill(BG)
        x = pad
        if not dream_only:
            screen.blit(big.render("real room", True, TEXT), (x, 14))
            screen.blit(to_surface(real_img, S), (x, 48))
            x += S + pad
        screen.blit(big.render("dream (world model)", True, ACCENT), (x, 14))
        screen.blit(to_surface(fake_img, S), (x, 48))
        keys = draw_keys(action, size=44)
        ksurf = pygame.image.fromstring(keys.tobytes(), keys.size, "RGB")
        screen.blit(ksurf, ((screen.get_width() - keys.width) // 2, S + 60))
        info = f"{since_sync} frames since sync   {clock.get_fps():4.1f} fps" + ("   ● REC" if recording else "")
        screen.blit(font.render(info, True, TEXT), (pad, S + 156))
        help_ = "arrows: move/turn  space: re-sync  n: new room  d: dream-only  c: record  q: quit"
        screen.blit(font.render(help_, True, (120, 126, 138)), (pad, S + 182))
        pygame.display.flip()
        if recording:
            rec_frames.append(pygame.surfarray.array3d(screen).swapaxes(0, 1).copy())
        clock.tick(a.fps)
    pygame.quit()


if __name__ == "__main__":
    main()
