"""Stage A - the tokenizer: a VQ-VAE that turns one 64x64 frame into a 16x16 grid
of integers (codebook ids), and back.

  encoder: 64x64x3 -> 16x16xD   (two stride-2 convs)
  quantize: each D-dim vector -> nearest of K codebook vectors -> its id
  decoder: 16x16xD -> 64x64x3

The codebook is learned with an exponential moving average (EMA) of the encoder
outputs assigned to each code, and unused ("dead") codes are restarted.

Losses (v2 - v1 used only the first line and came out blurry):
  pixel:  L1 + MSE                         get the colours right
  edges:  L1 on image gradients (dx, dy)   box / shadow / beam edges count extra
  GAN:    a small "patch critic" judges patches as real or reconstructed;
          the decoder is pushed to fool it -> crisp instead of averaged-out blur.
          Switched on after --gan_start steps, once reconstructions are decent.

usage: python vqvae.py --steps 20000
"""
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
                                 nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class Quantizer(nn.Module):
    """Nearest-neighbour codebook lookup with EMA updates."""

    def __init__(self, K=512, D=64, decay=0.99):
        super().__init__()
        self.K, self.D, self.decay = K, D, decay
        self.register_buffer("codebook", torch.randn(K, D) * 0.1)
        self.register_buffer("ema_count", torch.ones(K))
        self.register_buffer("ema_sum", self.codebook.clone())

    def ids(self, z):  # z: (B, D, H, W) -> (B, H, W) code ids
        flat = z.permute(0, 2, 3, 1).reshape(-1, self.D)
        dist = (flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.T
                + self.codebook.pow(2).sum(1))
        return dist.argmin(1).view(z.shape[0], z.shape[2], z.shape[3])

    def lookup(self, ids):  # (B, H, W) -> (B, D, H, W)
        return self.codebook[ids].permute(0, 3, 1, 2)

    def forward(self, z):
        ids = self.ids(z)
        q = self.lookup(ids)
        if self.training:
            with torch.no_grad():
                flat = z.permute(0, 2, 3, 1).reshape(-1, self.D)
                onehot = F.one_hot(ids.view(-1), self.K).float()
                self.ema_count.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.ema_sum.mul_(self.decay).add_(onehot.T @ flat, alpha=1 - self.decay)
                self.codebook.copy_(self.ema_sum / self.ema_count.clamp(min=1e-5)[:, None])
                # restart dead codes on random encoder outputs from this batch
                dead = self.ema_count < 1e-3
                if dead.any():
                    pick = flat[torch.randint(len(flat), (int(dead.sum()),), device=flat.device)]
                    self.codebook[dead] = pick
                    self.ema_sum[dead] = pick
                    self.ema_count[dead] = 1.0
        commit = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()  # straight-through gradient
        return q, ids, commit


class VQVAE(nn.Module):
    """grid=16: 64x64 -> 16x16 tokens (4x4 px each).  grid=8: 64x64 -> 8x8 tokens (8x8 px each)."""

    def __init__(self, K=512, D=64, ch=128, grid=16):
        super().__init__()
        self.grid = grid
        enc = [nn.Conv2d(3, ch // 2, 3, padding=1), ResBlock(ch // 2),
               nn.Conv2d(ch // 2, ch, 4, stride=2, padding=1), ResBlock(ch),            # 32x32
               nn.Conv2d(ch, ch, 4, stride=2, padding=1), ResBlock(ch), ResBlock(ch)]   # 16x16
        dec_in = [nn.Conv2d(D, ch, 3, padding=1), ResBlock(ch), ResBlock(ch)]
        if grid == 8:  # one more stage, wider, so each token can hold an 8x8 patch
            enc += [nn.Conv2d(ch, 2 * ch, 4, stride=2, padding=1), ResBlock(2 * ch), ResBlock(2 * ch)]  # 8x8
            dec_in = [nn.Conv2d(D, 2 * ch, 3, padding=1), ResBlock(2 * ch), ResBlock(2 * ch),
                      nn.ConvTranspose2d(2 * ch, ch, 4, stride=2, padding=1), ResBlock(ch)]        # 16x16
        top = 2 * ch if grid == 8 else ch
        self.enc = nn.Sequential(*enc, nn.GroupNorm(8, top), nn.SiLU(), nn.Conv2d(top, D, 1))
        self.vq = Quantizer(K, D)
        self.dec = nn.Sequential(
            *dec_in,
            nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), ResBlock(ch),     # 32x32
            nn.ConvTranspose2d(ch, ch // 2, 4, stride=2, padding=1), ResBlock(ch // 2),  # 64x64
            nn.GroupNorm(8, ch // 2), nn.SiLU(), nn.Conv2d(ch // 2, 3, 3, padding=1))

    @staticmethod
    def to_input(frames_uint8):  # (B, 64, 64, 3) uint8 -> (B, 3, 64, 64) in [-1, 1]
        return frames_uint8.permute(0, 3, 1, 2).float() / 127.5 - 1

    @staticmethod
    def to_image(x):  # (B, 3, 64, 64) -> (B, 64, 64, 3) uint8
        return ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1)

    def forward(self, x):
        q, ids, commit = self.vq(self.enc(x))
        return self.dec(q), ids, commit

    @torch.no_grad()
    def encode(self, x):  # -> (B, 16, 16) ids
        return self.vq.ids(self.enc(x))

    @torch.no_grad()
    def decode(self, ids):  # (B, 16, 16) ids -> (B, 3, 64, 64)
        return self.dec(self.vq.lookup(ids))


class PatchCritic(nn.Module):
    """PatchGAN discriminator: one real/fake score per patch (64x64 image -> 16x16 scores)."""

    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(ch, 2 * ch, 4, 2, 1), nn.GroupNorm(8, 2 * ch), nn.LeakyReLU(0.2),
            nn.Conv2d(2 * ch, 4 * ch, 3, 1, 1), nn.GroupNorm(8, 4 * ch), nn.LeakyReLU(0.2),
            nn.Conv2d(4 * ch, 1, 3, 1, 1))

    def forward(self, x):
        return self.net(x)


def edge_loss(recon, x):
    """L1 between image gradients: rewards matching sharp edges, not just colours."""
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return F.l1_loss(dx(recon), dx(x)) + F.l1_loss(dy(recon), dy(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="runs/vqvae")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--edge_w", type=float, default=1.0)
    ap.add_argument("--gan_w", type=float, default=0.1)
    ap.add_argument("--gan_start", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--fly_w", type=float, default=0.0, help="extra loss weight on firefly pixels")
    ap.add_argument("--init", default=None, help="start from these weights (fine-tuning)")
    ap.add_argument("--grid", type=int, default=16, choices=[8, 16], help="tokens per side")
    ap.add_argument("--K", type=int, default=512, help="codebook size")
    ap.add_argument("--D", type=int, default=64, help="code vector size")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frames = np.load(f"{a.cache}/frames.npy", mmap_mode="r")
    n_train = int(len(frames) * 0.95)  # last 5% of episodes held out
    flat = lambda ep, t: frames[ep, t]
    rng = np.random.default_rng(0)
    fly_pos = np.load(f"{a.cache}/fly_pos.npy") if a.fly_w > 0 else None   # (N, T+1, 2 flies, xy)
    c = (torch.arange(64, device=DEVICE) + 0.5) / 64
    gy, gx = torch.meshgrid(c, c, indexing="ij")

    def batch(lo, hi):
        ep = np.sort(rng.integers(lo, hi, a.batch))
        t = rng.integers(0, frames.shape[1], a.batch)
        x = torch.from_numpy(np.stack([flat(e, s) for e, s in zip(ep, t)])).to(DEVICE)
        weight = torch.ones(a.batch, 1, 64, 64, device=DEVICE)
        if fly_pos is not None:  # pixels within ~3 px of a firefly count (1 + fly_w) times
            fp = torch.from_numpy(np.nan_to_num(fly_pos[ep, t], nan=-9.0)).to(DEVICE)  # (B, 2, 2)
            d = torch.hypot(gx - fp[:, :, 0, None, None], gy - fp[:, :, 1, None, None])  # (B, 2, 64, 64)
            weight += a.fly_w * (d.min(1, keepdim=True).values < 0.05).float()
        batch.weight = weight
        return VQVAE.to_input(x)

    model, critic = VQVAE(K=a.K, D=a.D, grid=a.grid).to(DEVICE), PatchCritic().to(DEVICE)
    import json
    json.dump({"K": a.K, "D": a.D, "grid": a.grid}, open(f"{a.out}/config.json", "w"))
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=DEVICE))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.5, 0.9), weight_decay=0.0)
    opt_c = torch.optim.AdamW(critic.parameters(), lr=a.lr, betas=(0.5, 0.9), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    first, state = 1, f"{a.out}/train_state.pt"
    if os.path.exists(state):  # resume after a crash / restart
        ck = torch.load(state, map_location=DEVICE)
        model.load_state_dict(ck["model"]), critic.load_state_dict(ck["critic"])
        opt.load_state_dict(ck["opt"]), opt_c.load_state_dict(ck["opt_c"])
        sched.load_state_dict(ck["sched"])
        first = ck["step"] + 1
        print(f"resumed from step {ck['step']}")
    print(f"VQ-VAE params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M on {DEVICE}")
    t0 = time.time()
    g_adv = torch.zeros(())
    for step in range(first, a.steps + 1):
        x = batch(0, n_train)
        recon, ids, commit = model(x)
        err = (recon - x).abs() + (recon - x) ** 2
        rec = (batch.weight * err).mean() / batch.weight.mean()
        loss = rec + a.edge_w * edge_loss(recon, x) + 0.25 * commit
        use_gan = step >= a.gan_start
        if use_gan:  # decoder tries to make the critic call its patches "real"
            ramp = min(1.0, (step - a.gan_start) / 1000)
            g_adv = -critic(recon).mean()
            loss = loss + a.gan_w * ramp * g_adv
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if use_gan:  # critic: hinge loss, real patches > +1, reconstructed patches < -1
            d_loss = F.relu(1 - critic(x)).mean() + F.relu(1 + critic(recon.detach())).mean()
            opt_c.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_c.step()
        if step % 250 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                xv = batch(n_train, len(frames))
                rv, idv, _ = model(xv)
                mse = F.mse_loss(rv, xv).item() / 4  # [-1,1] -> [0,1] scale
                psnr = 10 * np.log10(1 / max(mse, 1e-10))
                used = len(torch.unique(idv))
            model.train()
            print(f"step {step:6d}  rec {rec.item():.4f}  gan {g_adv.item():+.3f}  val PSNR {psnr:.2f} dB  "
                  f"codes used {used}/{a.K}  {time.time() - t0:.0f}s", flush=True)
        if step % 500 == 0 or step == a.steps:
            torch.save(model.state_dict(), f"{a.out}/vqvae.pt")
            os.makedirs(f"{a.out}/ckpts", exist_ok=True)       # keep every checkpoint, pick the best later
            torch.save(model.state_dict(), f"{a.out}/ckpts/vqvae_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "critic": critic.state_dict(), "opt": opt.state_dict(),
                        "opt_c": opt_c.state_dict(), "sched": sched.state_dict(), "step": step}, state)
            # save a comparison strip: top = real, bottom = reconstruction
            grid = torch.cat([VQVAE.to_image(xv[:8]), VQVAE.to_image(rv[:8])], 1)  # (8,128,64,3)
            from PIL import Image
            img = Image.fromarray(torch.cat(list(grid), 1).cpu().numpy()).resize((1024, 256), 0)
            img.save(f"{a.out}/recon.png")
            os.makedirs(f"{a.out}/progress", exist_ok=True)
            img.save(f"{a.out}/progress/recon_{step:06d}.png")  # for a "watch it learn" montage


if __name__ == "__main__":
    main()
