"""Tiny live dashboard ([C]): stdlib HTTP server serving index.html + the run's state.json.

    python -m trader.ui.dashboard runs/<run_dir> [--port 8765]

Open http://127.0.0.1:8765 — the page polls state.json once a second. Works while
run_experiment.py is writing, or afterwards to inspect the final state.
"""
from __future__ import annotations

import argparse
import http.server
import os
import sys
from functools import partial

HERE = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, run_dir: str, **kw):
        self.run_dir = run_dir
        super().__init__(*a, directory=HERE, **kw)

    def translate_path(self, path):
        if path.startswith("/state.json") or path.startswith("/metrics.json") or path.startswith("/manifest.json"):
            return os.path.join(self.run_dir, path.lstrip("/").split("?")[0])
        if path in ("/", ""):
            path = "/index.html"
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args):  # quiet
        pass


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args(argv)
    if not os.path.isdir(a.run_dir):
        print(f"run dir not found: {a.run_dir}", file=sys.stderr)
        return 2
    srv = http.server.ThreadingHTTPServer((a.host, a.port), partial(Handler, run_dir=os.path.abspath(a.run_dir)))
    print(f"dashboard: http://{a.host}:{a.port}  (serving {a.run_dir}; Ctrl-C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
