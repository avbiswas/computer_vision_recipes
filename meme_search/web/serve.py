"""Local server for the meme search demo: static files + /api/fill (LLM text for multi-box templates).

  OPENROUTER_API_KEY=... python web/serve.py            # then open http://127.0.0.1:8000

The API key stays on this server; the page only sends {query, file, slots} and gets back {texts, model}.
"""
import argparse, json, os, re, sys, time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from search import LLM_MODEL, llm_fill  # noqa: E402

SAFE_FILE = re.compile(r"^[\w.-]+\.webp$")


class Handler(SimpleHTTPRequestHandler):
    model = LLM_MODEL

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path.rstrip("/") != "/api/fill":
            return self._json(404, {"error": "not found"})
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            query, file, slots = str(req["query"])[:300], str(req["file"]), req["slots"]
            if not SAFE_FILE.match(file) or not (1 <= len(slots) <= 8):
                return self._json(400, {"error": "bad request"})
            img = Image.open(os.path.join(HERE, "data", "templates", file)).convert("RGB")
            # page slots are in thumbnail pixels; llm_fill needs bbox + an approximate line count
            sl = [{"bbox": [float(v) for v in s["bbox"]][:4],
                   "lines": max(1, round((s["bbox"][3] - s["bbox"][1]) / max(1.0, float(s["line_h"]))))} for s in slots]
            t0 = time.time()
            texts = llm_fill(query, sl, img, self.model)
            self.log_message("fill %s %d boxes %.1fs", file, len(sl), time.time() - t0)
            return self._json(200, {"texts": texts, "model": self.model})
        except Exception as e:
            self.log_message("fill failed: %s: %s", type(e).__name__, e)
            return self._json(502, {"error": f"{type(e).__name__}"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--llm", default=LLM_MODEL, help="OpenRouter model id")
    args = ap.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("warning: OPENROUTER_API_KEY not set; multi-box templates will fall back to keywords")
    Handler.model = args.llm
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=HERE))
    print(f"serving http://127.0.0.1:{args.port}  (LLM: {args.llm})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
