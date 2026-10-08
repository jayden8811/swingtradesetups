"""Tiny local web server: serves the page, runs scans in the background, places optional paper trades."""
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import alpaca, config as C, pipeline

WEB = C.ROOT / "web" / "index.html"
STATE = {"running": False, "log": [], "error": None}
_lock = threading.Lock()


def _progress(msg):
    STATE["log"] = (STATE["log"] + [str(msg)])[-30:]
    print(msg, flush=True)


def _run_scan():
    try:
        pipeline.scan(_progress)
    except Exception as e:
        STATE["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        STATE["running"] = False


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/latest":
            if pipeline.LATEST.exists():
                return self._send(200, pipeline.LATEST.read_bytes())
            return self._send(200, {"empty": True})
        if self.path in ("/api/heatmap", "/api/dips"):
            f = C.DATA_DIR / ("heatmap.json" if self.path == "/api/heatmap" else "dips_report.json")
            return self._send(200, f.read_bytes() if f.exists() else {"empty": True})
        if self.path == "/api/status":
            return self._send(200, STATE)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/scan":
            with _lock:
                if not STATE["running"]:
                    STATE.update(running=True, log=[], error=None)
                    threading.Thread(target=_run_scan, daemon=True).start()
            return self._send(202, STATE)
        if self.path == "/api/paper":
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                eq = float(alpaca.account()["equity"])
                risk = float(body.get("risk_pct", 1.0)) / 100
                qty = int(eq * risk / (body["entry"] - body["stop"]))
                if qty < 1:
                    return self._send(400, {"error": "position size < 1 share"})
                order = alpaca.bracket_order(body["symbol"], qty, body["t1"], body["stop"])
                return self._send(200, {"qty": qty, "order_id": order.get("id"), "status": order.get("status")})
            except Exception as e:
                return self._send(400, {"error": str(e)})
        self._send(404, {"error": "not found"})


def serve(port=8000):
    print(f"📈 Swing Setups running at http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
