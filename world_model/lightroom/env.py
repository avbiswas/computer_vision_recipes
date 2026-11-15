"""FireflyRoom: a dark top-down room with fixed boxes, a user flashlight and fireflies.

- Boxes: 2-3, random position/size/angle per episode, never move.
- Flashlight: warm white cone of light with a soft edge, plus a faint halo around it.
  Tank controls: forward / backward along its heading, turn left / right.
  It sits on the floor, is blocked by boxes and walls, and boxes cast shadows from it.
- Fireflies: 1-2 coloured lights that wander on their own (smoothly turning heading,
  constant speed) and bounce cleanly off the walls. They fly *above* the boxes:
  while a firefly is over a box, that box casts no shadow from it.
- Light is additive, so coloured shadows appear where one light is blocked and another isn't.

Exact 2D ray casting in numpy. Coordinates are room units in [0, 1].
"""
from __future__ import annotations
import numpy as np

# Action ids. Names match the arrow keys: up = forward, down = backward,
# left / right = turn the flashlight (counter-)clockwise.
DIR_NAMES = ["stay", "up", "down", "left", "right"]
ACTION_LABELS = ["no input", "forward", "backward", "turn left", "turn right"]
N_ACTIONS = len(DIR_NAMES)

PALETTE = {  # firefly colours
    "amber": (1.00, 0.62, 0.15), "cyan": (0.20, 0.85, 1.00), "magenta": (1.00, 0.25, 0.80),
    "lime": (0.55, 1.00, 0.25), "violet": (0.60, 0.40, 1.00),
}


class FireflyRoom:
    MAX_BOXES = 3
    MAX_FLIES = 2
    LAMP_STEP = 0.03                 # forward/backward distance per step
    TURN_STEP = np.deg2rad(10)       # turn per step
    LAMP_R = 0.045                   # flashlight body radius (~3 px at 64x64)
    BEAM_HALF = np.deg2rad(25)       # half-angle of the cone (50 degrees wide)
    BEAM_SOFT = np.deg2rad(7)        # soft edge width
    BEAM_FALLOFF = 0.8               # beam reaches far
    HALO_FALLOFF = 0.10              # faint glow right around the flashlight
    LAMP_RGB = np.array([1.0, 0.88, 0.62])
    FLY_R = 0.03                     # firefly body radius (~2 px at 64x64)
    FLY_FALLOFF = 0.22
    FLY_GAIN = 0.85
    FLY_SPEED = 0.018      # room units per step
    FLY_TURN = 0.18        # std of heading change per step (radians)
    AMBIENT = np.array([0.02, 0.025, 0.035])
    BOX_RGB = np.array([0.29, 0.31, 0.36])

    def __init__(self, res: int = 64, n_boxes: int | None = None, n_flies: int | None = None,
                 seed: int | None = None):
        self.res = res
        self.fixed_n_boxes, self.fixed_n_flies = n_boxes, n_flies
        self.rng = np.random.default_rng(seed)
        c = (np.arange(res) + 0.5) / res
        self.px, self.py = np.meshgrid(c, c)
        self.last_blocked = False
        self.reset()

    # ---- state ---------------------------------------------------------------
    def reset(self):
        rng = self.rng
        self.n_boxes = self.fixed_n_boxes or int(rng.integers(2, self.MAX_BOXES + 1))
        self.n_flies = self.fixed_n_flies or int(rng.integers(1, self.MAX_FLIES + 1))
        for _ in range(1000):
            self.boxes = [dict(c=rng.uniform(0.15, 0.85, 2), w=rng.uniform(0.08, 0.22),
                               h=rng.uniform(0.06, 0.14), ang=rng.uniform(0, np.pi))
                          for _ in range(self.n_boxes)]
            self.polys = [self._poly(b) for b in self.boxes]
            self.lamp = self._sample_lamp_start()
            self.heading = rng.uniform(-np.pi, np.pi)
            if self._boxes_ok() and self._lamp_ok(self.lamp):
                break
        else:
            raise RuntimeError("could not place objects")
        names = rng.choice(list(PALETTE), self.n_flies, replace=False)
        self.fly_names = [str(n) for n in names]
        self.fly_rgb = np.array([PALETTE[n] for n in names])
        self.fly_pos = rng.uniform(0.1, 0.9, (self.n_flies, 2))
        self.fly_head = rng.uniform(0, 2 * np.pi, self.n_flies)
        self.t = 0
        return self.render()

    def _sample_lamp_start(self) -> np.ndarray:
        """40% of episodes start the lamp near an edge or corner, the rest uniform."""
        rng, lo, hi = self.rng, 0.06, 0.94
        p = rng.uniform(lo, hi, 2)
        if rng.random() < 0.4:
            band = lambda: rng.uniform(lo, 0.18) if rng.random() < .5 else 1 - rng.uniform(lo, 0.18)
            if rng.random() < 0.5:
                p = np.array([band(), band()])
            else:
                p[int(rng.integers(2))] = band()
        return p

    def _poly(self, b) -> np.ndarray:
        ca, sa = np.cos(b["ang"]), np.sin(b["ang"])
        hw, hh = b["w"] / 2, b["h"] / 2
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        return local @ np.array([[ca, -sa], [sa, ca]]).T + b["c"]

    @staticmethod
    def _inside(poly, x, y):
        inside = np.ones(np.shape(x), dtype=bool)
        for i in range(4):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % 4]
            inside &= (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1) >= 0
        return inside

    def _boxes_ok(self) -> bool:
        for p in self.polys:
            if (p < 0.02).any() or (p > 0.98).any():
                return False
        g = (np.arange(128) + 0.5) / 128
        gx, gy = np.meshgrid(g, g)
        occ = sum(self._inside(p, gx, gy).astype(int) for p in self.polys)
        return occ.max() <= 1

    def _lamp_ok(self, pos) -> bool:
        r = self.LAMP_R * 1.5
        if (pos < r).any() or (pos > 1 - r).any():
            return False
        # lamp disc must not touch any box: test centre + ring of points
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        xs = np.r_[pos[0], pos[0] + r * np.cos(ang)]
        ys = np.r_[pos[1], pos[1] + r * np.sin(ang)]
        return not any(self._inside(p, xs, ys).any() for p in self.polys)

    # ---- dynamics ------------------------------------------------------------
    def step(self, action: int):
        # 1) flashlight: tank controls
        a = int(action)
        self.last_blocked = False
        if a in (1, 2):
            sign = 1 if a == 1 else -1
            new = self.lamp + sign * self.LAMP_STEP * np.array([np.cos(self.heading), np.sin(self.heading)])
            self.last_blocked = not self._lamp_ok(new)
            if not self.last_blocked:
                self.lamp = new
        elif a in (3, 4):  # left = counter-clockwise on screen (y points down)
            self.heading = (self.heading + (-1 if a == 3 else 1) * self.TURN_STEP + np.pi) % (2 * np.pi) - np.pi
        # 2) fireflies: wander + clean mirror bounce off walls
        self.fly_head += self.rng.normal(0, self.FLY_TURN, self.n_flies)
        v = self.FLY_SPEED * np.c_[np.cos(self.fly_head), np.sin(self.fly_head)]
        pos = self.fly_pos + v
        lo, hi = self.FLY_R, 1 - self.FLY_R
        for k in range(2):
            low, high = pos[:, k] < lo, pos[:, k] > hi
            pos[low, k] = 2 * lo - pos[low, k]
            pos[high, k] = 2 * hi - pos[high, k]
            v[low | high, k] *= -1
        self.fly_pos = pos
        self.fly_head = np.arctan2(v[:, 1], v[:, 0])
        self.t += 1
        return self.render()

    def state(self) -> dict:
        return dict(lamp=self.lamp.copy(), heading=self.heading, fly_pos=self.fly_pos.copy(), fly_head=self.fly_head.copy())

    # ---- rendering -----------------------------------------------------------
    def _visibility(self, lx, ly, polys):
        px, py = self.px, self.py
        lit = np.ones_like(px, dtype=bool)
        dx, dy = lx - px, ly - py
        for p in polys:
            for i in range(4):
                (ax, ay), (bx, by) = p[i], p[(i + 1) % 4]
                ex, ey = bx - ax, by - ay
                den = dx * ey - dy * ex
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = ((ax - px) * ey - (ay - py) * ex) / den
                    u = ((ax - px) * dy - (ay - py) * dx) / den
                lit &= ~((np.abs(den) > 1e-12) & (t > 1e-6) & (t < 1) & (u >= 0) & (u <= 1))
        return lit

    def render(self) -> np.ndarray:
        px, py = self.px, self.py
        img = np.broadcast_to(self.AMBIENT, px.shape + (3,)).copy()
        box_mask = np.zeros_like(px, dtype=bool)
        for p in self.polys:
            box_mask |= self._inside(p, px, py)

        # flashlight (floor level, blocked by every box): soft-edged cone + small halo
        lx, ly = self.lamp
        hx, hy = np.cos(self.heading), np.sin(self.heading)
        d_lamp = np.hypot(px - lx, py - ly)
        cos_to_pixel = ((px - lx) * hx + (py - ly) * hy) / np.maximum(d_lamp, 1e-6)
        inner, outer = np.cos(self.BEAM_HALF), np.cos(self.BEAM_HALF + self.BEAM_SOFT)
        edge = np.clip((cos_to_pixel - outer) / (inner - outer), 0, 1)
        cone = edge * edge * (3 - 2 * edge)                          # smoothstep -> soft edge
        light = (cone * np.exp(-(d_lamp / self.BEAM_FALLOFF) ** 2)
                 + 0.35 * np.exp(-(d_lamp / self.HALO_FALLOFF) ** 2))
        img += (light * self._visibility(lx, ly, self.polys))[..., None] * self.LAMP_RGB * 0.95

        # fireflies (fly above boxes: the box underneath casts no shadow from it)
        fly_d = []
        for (fx, fy), rgb in zip(self.fly_pos, self.fly_rgb):
            under = [self._inside(p, np.array([fx]), np.array([fy]))[0] for p in self.polys]
            occluders = [p for p, u in zip(self.polys, under) if not u]
            d = np.hypot(px - fx, py - fy)
            fly_d.append(d)
            img += (np.exp(-(d / self.FLY_FALLOFF) ** 2) * self._visibility(fx, fy, occluders)
                    )[..., None] * rgb * self.FLY_GAIN

        # boxes: flat grey tops, lightly tinted by any firefly hovering close above
        tint = sum(np.exp(-(d / (self.FLY_FALLOFF * 0.6)) ** 2)[..., None] * rgb * 0.35
                   for d, rgb in zip(fly_d, self.fly_rgb))
        img[box_mask] = (self.BOX_RGB + tint)[box_mask] if self.n_flies else self.BOX_RGB

        # light bodies drawn last (fireflies on top of everything)
        # flashlight: dark outline ring, white body, and a bright "lens" towards the heading
        img[d_lamp < self.LAMP_R * 1.35] = (0.08, 0.08, 0.10)
        img[d_lamp < self.LAMP_R] = (0.92, 0.92, 0.88)
        lens = np.hypot(px - (lx + hx * self.LAMP_R * 0.75), py - (ly + hy * self.LAMP_R * 0.75))
        img[lens < self.LAMP_R * 0.55] = (1.0, 0.85, 0.35)
        # fireflies: saturated coloured body with a white-hot centre
        for d, rgb in zip(fly_d, self.fly_rgb):
            img[d < self.FLY_R] = rgb
            img[d < self.FLY_R * 0.45] = 0.5 + 0.5 * rgb
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)
