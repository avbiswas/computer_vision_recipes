"""Export the final world model + VQ-VAE decoder to ONNX for the browser demo, check parity,
and dump a few starting rooms (token history) as JSON.

  uv run --with onnx --with onnxruntime python web/export_onnx.py
"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import vqvae
vqvae.DEVICE = "cpu"
import dream
dream.DEVICE = "cpu"
from dream import load, Dream
from env import FireflyRoom
from dynamics import MASK

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "..", "video_assets", "checkpoints")


class Step(torch.nn.Module):
    """ctx (1,7,256) int64, fa (1,8) int64, times (1,8) f32, anchor (1,256) int64, anchor_time (1,) f32
    -> next frame ids (1,256) int64   (1-pass greedy, same as the dream)"""
    def __init__(self, wm):
        super().__init__(); self.wm = wm

    def forward(self, ctx, fa, times, anchor, anchor_time):
        tgt = torch.full_like(ctx[:, :1], MASK)
        logits = self.wm.forward_seq(torch.cat([ctx, tgt], 1), fa, times, anchor, anchor_time)
        return logits[:, -1].argmax(-1)


class Decode(torch.nn.Module):
    def __init__(self, vq):
        super().__init__(); self.vq = vq

    def forward(self, ids):  # (1,256) -> (1,64,64,4) uint8 RGBA, ready for ImageData
        x = self.vq.dec(self.vq.vq.lookup(ids.view(1, 16, 16)))
        rgb = ((x.clamp(-1, 1) + 1) * 127.5).round().permute(0, 2, 3, 1)
        return torch.cat([rgb, torch.full_like(rgb[..., :1], 255)], -1).to(torch.uint8)


def main():
    vq, wm = load(f"{CK}/vqvae_FINAL_16x16_K512.pt", f"{CK}/world_model_FINAL_rope1_step10000_9.8M.pt")
    vq.eval(); wm.eval()
    print("offsets", wm.offsets)

    # starting rooms: 8 real frames, encoded
    rooms = []
    for seed in range(8):
        env = FireflyRoom(res=64, seed=seed, n_boxes=3, n_flies=2)
        frames, acts = [env.reset()], []
        for _ in range(7):
            acts.append(0); frames.append(env.step(0))
        d = Dream(vq, wm, frames, acts, passes=1, temperature=0)
        rooms.append({"hist": [h.tolist() for h in d.hist], "acts": acts})
    json.dump(rooms, open(f"{HERE}/rooms.json", "w"))

    # reference rollout from room 0 in torch (to check the ONNX path end-to-end)
    env = FireflyRoom(res=64, seed=0, n_boxes=3, n_flies=2)
    frames, acts = [env.reset()], []
    for _ in range(7):
        acts.append(0); frames.append(env.step(0))
    d = Dream(vq, wm, frames, acts, passes=1, temperature=0)
    script = [1] * 10 + [3] * 8 + [1] * 10 + [0] * 12
    ref_imgs = [d.step(a) for a in script]

    step, dec = Step(wm).eval(), Decode(vq).eval()
    ex = (torch.zeros(1, 7, 256, dtype=torch.long), torch.zeros(1, 8, dtype=torch.long),
          torch.arange(-7, 1).float()[None], torch.zeros(1, 256, dtype=torch.long), torch.tensor([-7.0]))
    with torch.no_grad():
        torch.onnx.export(step, ex, f"{HERE}/world_model.onnx", opset_version=17, dynamo=False,
                          input_names=["ctx", "fa", "times", "anchor", "anchor_time"], output_names=["next"])
        torch.onnx.export(dec, (torch.zeros(1, 256, dtype=torch.long),), f"{HERE}/decoder.onnx",
                          opset_version=17, dynamo=False, input_names=["ids"], output_names=["rgba"])
    for f in ("world_model.onnx", "decoder.onnx"):
        print(f, f"{os.path.getsize(f'{HERE}/{f}') / 1e6:.1f} MB")

    # replay the same rollout through onnxruntime, mirroring DreamRoPE.step
    import onnxruntime as ort
    s_wm = ort.InferenceSession(f"{HERE}/world_model.onnx")
    s_dec = ort.InferenceSession(f"{HERE}/decoder.onnx")
    r = rooms[0]
    hist, acts = [np.array(h) for h in r["hist"]], list(r["acts"])
    anchor = hist[0]
    diffs, ms = [], []
    for a, ref in zip(script, ref_imgs):
        acts.append(a)
        t = len(hist)
        idx = [max(t - o, 0) for o in wm.offsets]
        fa = [acts[i - 1] if (j > 0 and i >= 1 and idx[j - 1] == i - 1) else 5 for j, i in enumerate(idx)]
        t0 = time.time()
        nxt = s_wm.run(None, {"ctx": np.stack([hist[i] for i in idx[:-1]])[None].astype(np.int64),
                              "fa": np.array([fa], np.int64),
                              "times": np.array([[i - t for i in idx]], np.float32),
                              "anchor": anchor[None].astype(np.int64),
                              "anchor_time": np.array([-min(t, 250)], np.float32)})[0][0]
        img = s_dec.run(None, {"ids": nxt[None]})[0][0, :, :, :3]
        ms.append((time.time() - t0) * 1000)
        hist.append(nxt)
        diffs.append(np.abs(img.astype(int) - ref.astype(int)).mean())
    print(f"ONNX vs torch rollout, mean |pixel diff| per step: max {max(diffs):.3f}, mean {np.mean(diffs):.3f}")
    print(f"onnxruntime CPU (python): {np.median(ms):.0f} ms per frame")


if __name__ == "__main__":
    main()
