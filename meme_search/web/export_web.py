"""Export everything the browser demo needs (run once after train_clip.py + remove_text.py).

  python web/export_web.py --ckpt runs/clip_b16/best.pt

Writes (all gitignored, regenerate with this script):
  web/models/meme-clip/          fine-tuned CLIP text encoder for transformers.js
      config.json, tokenizer files, onnx/text_model.onnx, onnx/text_model_quantized.onnx (int8)
  web/data/embeddings.bin        float32 [N, 512] L2-normalised template embeddings
  web/data/templates.json        [{file, w, h, slots: [{bbox, line_h, light}]}]  (coords in thumbnail pixels)
  web/data/templates/*.webp      text-free templates, max side 640
Export-only deps: onnx, onnxruntime (not needed for training or the demo itself).
"""
import argparse, json, os, shutil, sys
from huggingface_hub import hf_hub_download
import numpy as np, torch, torch.nn.functional as F, open_clip
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from search import group_slots  # noqa: E402

REF = "Xenova/clip-vit-base-patch16"  # config + tokenizer layout transformers.js expects
TOK_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]
MAX_SIDE = 640


def to_hf_text(oc):
    """Copy open_clip text tower weights into a transformers CLIPTextModelWithProjection."""
    from transformers import CLIPTextConfig, CLIPTextModelWithProjection
    d = oc.token_embedding.weight.shape[1]
    cfg = CLIPTextConfig(vocab_size=oc.token_embedding.weight.shape[0], hidden_size=d, intermediate_size=4 * d,
                         num_hidden_layers=len(oc.transformer.resblocks), num_attention_heads=oc.transformer.resblocks[0].attn.num_heads,
                         max_position_embeddings=oc.positional_embedding.shape[0], projection_dim=oc.text_projection.shape[1],
                         hidden_act="gelu",  # open_clip "ViT-B-16" uses exact GELU (we fine-tuned with it)
                         eos_token_id=2, pad_token_id=1, bos_token_id=0)  # eos_token_id=2 -> argmax(input_ids) pooling
    hf = CLIPTextModelWithProjection(cfg).eval()
    sd = {"text_model.embeddings.token_embedding.weight": oc.token_embedding.weight,
          "text_model.embeddings.position_embedding.weight": oc.positional_embedding,
          "text_model.final_layer_norm.weight": oc.ln_final.weight, "text_model.final_layer_norm.bias": oc.ln_final.bias,
          "text_projection.weight": oc.text_projection.T}
    for i, b in enumerate(oc.transformer.resblocks):
        p = f"text_model.encoder.layers.{i}."
        qw, kw, vw = b.attn.in_proj_weight.chunk(3); qb, kb, vb = b.attn.in_proj_bias.chunk(3)
        sd.update({p + "self_attn.q_proj.weight": qw, p + "self_attn.q_proj.bias": qb,
                   p + "self_attn.k_proj.weight": kw, p + "self_attn.k_proj.bias": kb,
                   p + "self_attn.v_proj.weight": vw, p + "self_attn.v_proj.bias": vb,
                   p + "self_attn.out_proj.weight": b.attn.out_proj.weight, p + "self_attn.out_proj.bias": b.attn.out_proj.bias,
                   p + "layer_norm1.weight": b.ln_1.weight, p + "layer_norm1.bias": b.ln_1.bias,
                   p + "layer_norm2.weight": b.ln_2.weight, p + "layer_norm2.bias": b.ln_2.bias,
                   p + "mlp.fc1.weight": b.mlp.c_fc.weight, p + "mlp.fc1.bias": b.mlp.c_fc.bias,
                   p + "mlp.fc2.weight": b.mlp.c_proj.weight, p + "mlp.fc2.bias": b.mlp.c_proj.bias})
    missing, unexpected = hf.load_state_dict({k: v.detach().float().clone() for k, v in sd.items()}, strict=False)
    missing = [m for m in missing if "position_ids" not in m]
    assert not missing and not unexpected, (missing, unexpected)
    return hf


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, attention_mask):
        o = self.m(input_ids=input_ids, attention_mask=attention_mask)
        return o.text_embeds, o.last_hidden_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ViT-B-16:openai")
    ap.add_argument("--ckpt", default=f"{ROOT}/runs/clip_b16/best.pt")
    args = ap.parse_args()
    mdir, ddir = f"{HERE}/models/meme-clip", f"{HERE}/data"
    os.makedirs(f"{mdir}/onnx", exist_ok=True); os.makedirs(f"{ddir}/templates", exist_ok=True)

    name, pt = args.model.split(":")
    oc, _, pre = open_clip.create_model_and_transforms(name, pretrained=pt)
    oc.load_state_dict(torch.load(args.ckpt, map_location="cpu")); oc.eval()
    oc_tok = open_clip.get_tokenizer(name)

    # 1. tokenizer + config from the reference repo
    for f in TOK_FILES:
        if not os.path.exists(f"{mdir}/{f}"):
            shutil.copy(hf_hub_download(REF, f), f"{mdir}/{f}")
    from transformers import AutoTokenizer
    hf_tok = AutoTokenizer.from_pretrained(mdir)

    # 2. convert + check against open_clip
    hf = to_hf_text(oc)
    probe = ["my code finally compiled after 6 hours of debugging", "cat", "when the teacher says the exam is open book!"]
    with torch.no_grad():
        ref = F.normalize(oc.encode_text(oc_tok(probe)), dim=-1)
        x = hf_tok(probe, padding=True, return_tensors="pt")
        got = F.normalize(hf(**x).text_embeds, dim=-1)
    cos = (ref * got).sum(-1)
    print("open_clip vs HF cosine:", cos.tolist()); assert cos.min() > 0.999

    # 3. ONNX export + int8 quantization, checked with onnxruntime
    onnx_path, q_path = f"{mdir}/onnx/text_model.onnx", f"{mdir}/onnx/text_model_quantized.onnx"
    torch.onnx.export(Wrap(hf), (x["input_ids"], x["attention_mask"]), onnx_path, dynamo=False, opset_version=17,
                      input_names=["input_ids", "attention_mask"], output_names=["text_embeds", "last_hidden_state"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "text_embeds": {0: "batch"}, "last_hidden_state": {0: "batch", 1: "seq"}})
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnxruntime as ort
    quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
    feeds = {"input_ids": x["input_ids"].numpy(), "attention_mask": x["attention_mask"].numpy()}
    for p in (onnx_path, q_path):
        e = ort.InferenceSession(p).run(["text_embeds"], feeds)[0]
        e = e / np.linalg.norm(e, axis=-1, keepdims=True)
        print(f"{os.path.basename(p)}: {os.path.getsize(p) / 1e6:.0f} MB, cosine vs torch {(e * ref.numpy()).sum(-1).round(4).tolist()}")

    # 4. template embeddings, slots, thumbnails
    clean, lay = f"{ROOT}/out/clean", f"{ROOT}/out/layout"
    files = sorted(os.listdir(clean))
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    oc = oc.to(dev); embs, meta = [], []
    with torch.no_grad():
        for i in range(0, len(files), 64):
            chunk = files[i:i + 64]
            ims = [Image.open(f"{clean}/{f}").convert("RGB") for f in chunk]
            embs.append(F.normalize(oc.encode_image(torch.stack([pre(im) for im in ims]).to(dev)).float(), dim=-1).cpu())
            for f, im in zip(chunk, ims):
                stem = os.path.splitext(f)[0]
                s = min(1.0, MAX_SIDE / max(im.size))
                th = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS) if s < 1 else im
                th.save(f"{ddir}/templates/{stem}.webp", quality=82)
                g = np.asarray(im.convert("L"))
                slots = []
                for sl in group_slots(json.load(open(f"{lay}/{stem}.json"))["text"], im.height):
                    x0, y0, x1, y1 = [int(v) for v in sl["bbox"]]
                    bg = g[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
                    slots.append({"bbox": [round(v * s) for v in (x0, y0, x1, y1)], "line_h": round(sl["line_h"] * s),
                                  "light": bool(bg.size and bg.mean() > 200 and bg.std() < 30)})
                meta.append({"file": f"{stem}.webp", "w": th.width, "h": th.height, "slots": slots})
            print(f"\rtemplates {min(i + 64, len(files))}/{len(files)}", end="", flush=True)
    print()
    np.concatenate([e.numpy() for e in embs]).astype(np.float32).tofile(f"{ddir}/embeddings.bin")
    json.dump(meta, open(f"{ddir}/templates.json", "w"))
    print(f"wrote {len(meta)} templates, logit_scale={oc.logit_scale.exp().item():.1f}")


if __name__ == "__main__":
    main()
