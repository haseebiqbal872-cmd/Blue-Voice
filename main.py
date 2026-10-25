"""
Blue Voice - local app.

    python main.py [--port 8765] [--no-browser]

Starts the hand-tracking core and serves the Glass Chat UI on
http://127.0.0.1:<port> (this computer only).
"""

from __future__ import annotations

import argparse
import threading
import webbrowser

from aiohttp import web

from server import HOST, PORT, make_app


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{HOST}:{args.port}/")).start()
    web.run_app(make_app(), host=HOST, port=args.port, print=lambda *_: None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
