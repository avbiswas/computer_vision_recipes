"""Export the D-FINE layout detector to ONNX for the browser demo, make an fp16 copy (check both with web/check_onnx.py).

python web/export_onnx.py [--weights models/layout_dfine2_ep9]
-> web/models/layout_fp32.onnx, layout_fp16.onnx (+ classes.json)
Input  "pixel_values": float32 [1, 3, 1024, 1024], RGB in 0-1 (page resized to 1024x1024, no normalisation)
Output "logits": [1, 300, 11] (apply sigmoid), "pred_boxes": [1, 300, 4] (cx, cy, w, h in 0-1)
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, torch
from PIL import Image
from transformers import DFineForObjectDetection

ap = argparse.ArgumentParser(); ap.add_argument("--weights", default="models/layout_dfine2_ep9"); a = ap.parse_args()
OUT, S = Path("web/models"), 1024
OUT.mkdir(parents=True, exist_ok=True)


class Wrap(torch.nn.Module):  # plain tensors in/out
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values)
        return o.logits, o.pred_boxes


m = DFineForObjectDetection.from_pretrained(a.weights).eval()
json.dump([m.config.id2label[i] for i in range(len(m.config.id2label))], open(OUT / "classes.json", "w"))
page = lambda f: torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((S, S), Image.BILINEAR), dtype=np.float32) / 255).permute(2, 0, 1)[None]
x = page(sorted(glob.glob("data/doclaynet/images/test/*.jpg"))[0])
torch.onnx.export(Wrap(m), (x,), OUT / "layout_fp32.onnx", input_names=["pixel_values"], output_names=["logits", "pred_boxes"],
                  opset_version=17, dynamo=False)


# ---- fp16 for the browser (8 MB); verify with: python web/check_onnx.py web/models/layout_fp16.onnx
# fixed 1024x1024 input: fold everything input-independent (position embeddings, anchors) into constants.
# onnxruntime-web can't run some of the Cast nodes in that code, and the graph gets simpler.
import onnxslim
onnx.save(onnxslim.slim(onnx.load(OUT / "layout_fp32.onnx")), OUT / "layout_fp32.onnx")

from onnxruntime.transformers.float16 import convert_float_to_float16
fp = onnx.load(OUT / "layout_fp32.onnx")
# keep_io_types: page in / boxes out stay fp32; ops sensitive to fp16 (sampling, top-k, casts in the position code) stay fp32
fp16 = convert_float_to_float16(fp, keep_io_types=True, op_block_list=["Resize", "GridSample", "TopK", "Cast", "Range", "CumSum"])
# the converter can emit the same cast node twice (one tensor feeding several blocked ops): keep the first copy
produced, keep = set(), []
for node in fp16.graph.node:
    if node.output and all(o in produced for o in node.output): continue
    produced.update(node.output); keep.append(node)
del fp16.graph.node[:]; fp16.graph.node.extend(keep)
onnx.save(fp16, OUT / "layout_fp16.onnx")
for f in ["layout_fp32.onnx", "layout_fp16.onnx"]:
    print(f, round((OUT / f).stat().st_size / 1e6, 1), "MB")
# int8 (dynamic, MatMul only) was tried: 11.6 MB and it changed detections, so it isn't shipped.
