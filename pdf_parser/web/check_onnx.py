"""Check ONNX exports against PyTorch on 5 val pages: same detections (score>0.5) and box differences.
python web/check_onnx.py web/models/layout_fp16.onnx [more.onnx]"""
import glob, sys, numpy as np, onnxruntime as ort, torch
from PIL import Image
from transformers import DFineForObjectDetection
S=1024
page = lambda f: torch.from_numpy(np.asarray(Image.open(f).convert("RGB").resize((S, S), Image.BILINEAR), dtype=np.float32) / 255).permute(2, 0, 1)[None]
m = DFineForObjectDetection.from_pretrained("models/layout_dfine2_ep9").eval()
def dets(l, b, t=0.5):
    s = 1/(1+np.exp(-l)); sc, c = s.max(-1), s.argmax(-1); k = sc > t
    return sorted(zip(sc[k], c[k], map(tuple, b[k])), key=lambda r: -r[0])
for path in sys.argv[1:]:
    sess = ort.InferenceSession(path); worst=0; nd=[]
    for f in sorted(glob.glob("data/doclaynet/images/val/*.jpg"))[:5]:
        x = page(f)
        with torch.no_grad(): o = m(pixel_values=x)
        A = dets(o.logits[0].numpy(), o.pred_boxes[0].numpy())
        l, b = sess.run(None, {"pixel_values": x.numpy()}); B = dets(l[0], b[0])
        nd.append((len(A), len(B)))
        for sa, ca, ba in A:  # nearest same-class box in ONNX output
            d = min([np.abs(np.array(ba) - np.array(bb)).max() for sb, cb, bb in B if cb == ca] or [1.0]); worst = max(worst, d)
    print(path, "detections torch/onnx per page", nd, "worst box diff", round(float(worst), 5))
