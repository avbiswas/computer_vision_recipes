"""Fine-tune a COCO-pretrained YOLO11 on our DocLayNet subset (Apple Silicon / MPS).

python train.py --model yolo11s.pt --epochs 12 --imgsz 1024 --batch 8 --name y11s_doclaynet
"""
import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolo11s.pt")
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--imgsz", type=int, default=1024)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--fraction", type=float, default=1.0)
ap.add_argument("--name", default="y11s_doclaynet")
a = ap.parse_args()

YOLO(a.model).train(
    data="data/doclaynet/data.yaml",
    epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, fraction=a.fraction,
    device="mps", workers=4, project="runs", name=a.name, exist_ok=True,
    # documents are not natural images: no flips (text direction matters), no mosaic-heavy tricks at the end
    fliplr=0.0, flipud=0.0, mosaic=1.0, close_mosaic=3, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
    degrees=0.0, scale=0.25, translate=0.05,
    cos_lr=True, patience=50, plots=True, val=True,
)
