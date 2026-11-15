"""Shared helpers for video-production renders: arrow-key indicator + MP4 writer."""
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

AMBER, IDLE, EDGE, BG = (242, 184, 75), (40, 44, 56), (90, 96, 110), (12, 13, 17)


def font(size):
    for f in ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
              "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_keys(action, size=48):
    """Arrow-key cluster image; the pressed key (1 up, 2 down, 3 left, 4 right) is lit.
    action 0 ("stay") lights nothing and shows a small 'no input' label."""
    g = size // 8
    W, H = 3 * size + 4 * g, 2 * size + 3 * g + size // 2
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    pos = {1: (1, 0), 3: (0, 1), 2: (1, 1), 4: (2, 1)}       # up, left, down, right
    arrows = {1: [(.5, .25), (.75, .7), (.25, .7)], 2: [(.5, .75), (.75, .3), (.25, .3)],
              3: [(.25, .5), (.7, .25), (.7, .75)], 4: [(.75, .5), (.3, .25), (.3, .75)]}
    for a, (cx, cy) in pos.items():
        x0, y0 = g + cx * (size + g), g + cy * (size + g)
        on = action == a
        d.rounded_rectangle([x0, y0, x0 + size, y0 + size], radius=size // 6,
                            fill=AMBER if on else IDLE, outline=EDGE, width=max(1, size // 24))
        d.polygon([(x0 + u * size, y0 + v * size) for u, v in arrows[a]],
                  fill=(20, 20, 24) if on else (150, 156, 168))
    label = ["no input", "forward", "backward", "turn left", "turn right"][action]
    d.text((W // 2, 2 * size + 2 * g + size // 4), label, fill=AMBER if action else (120, 126, 138),
           font=font(size // 3), anchor="mm")
    return img


def upscale(frame, scale):
    return Image.fromarray(frame).resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST)


def write_mp4(frames, path, fps=20):
    """frames: list of HxWx3 uint8 arrays (same size). High quality H.264."""
    h, w = frames[0].shape[:2]
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "12",
                           "-preset", "slow", "-pix_fmt", "yuv420p",
                           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", path], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(np.ascontiguousarray(f).tobytes())
    ff.stdin.close()
    ff.wait()
