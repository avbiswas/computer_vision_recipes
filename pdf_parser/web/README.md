# PDF layout parser in the browser

pdf.js renders each page and reads its text layer; D-FINE nano (fine-tuned on DocLayNet, Apache-2.0) runs in
onnxruntime-web on WebGPU (WASM fallback) to find layout regions; the page becomes markdown with the same rules as
`parse_pdf.py` (duplicate-box removal, two-column reading order, headings/lists/captions by class).
No server-side compute: everything runs in the tab.

```
python web/export_onnx.py --weights models/layout_dfine2_ep9   # -> web/models/layout_fp16.onnx (8 MB) + layout_fp32.onnx
python web/check_onnx.py web/models/layout_fp16.onnx           # same detections as PyTorch on 5 val pages?
cp samples/resnet_2col.pdf web/samples/sample.pdf              # for the "Try a sample" button
cd web && python -m http.server 8000                           # open http://localhost:8000  (add ?wasm to force CPU)
```

Measured in Chrome on an M2 Max (ResNet paper): model load 4.6 s, then ~37 ms per page on WebGPU, ~560 ms on WASM.

Export notes: the fp32 graph is constant-folded with onnxslim first (fixed 1024x1024 input) because onnxruntime-web
can't run some Cast nodes in the position-embedding code; the fp16 converter keeps sampling/top-k/cast ops in fp32 and
can emit duplicate cast nodes, which the script removes. int8 was tried and changed detections, so it isn't shipped.

Limits: scanned PDFs (no text layer) get boxes but no text; tables and formulas come out as raw text-layer text
(no structure / LaTeX; that needs a reader model). Models and samples are gitignored.
