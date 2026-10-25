"""Overlay mode: only the three buttons and the message box float over whatever app you are using.

    python overlay.py

Sign a message, show your open palm, and the text is typed into the app that was in front (ChatGPT, Claude, Gemini,
any message box). You press that app's own Send button. Press "End conversation" to go back to Blue Voice.

Needs one extra package:  pip install pywebview
The transparent window is Windows-first. If it does not work on your computer, python main.py still works
(it then uses a normal window instead of an overlay).
"""
from __future__ import annotations

import ctypes
import sys
import threading
import time

from aiohttp import web

from server import HOST, PORT, make_app


def _no_activate(window) -> None:
    """Clicking the buttons must not steal keyboard focus from the app behind (Windows)."""
    if not sys.platform.startswith("win"):
        return
    try:
        hwnd = int(window.native.Handle.ToInt64())
        u = ctypes.windll.user32
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x00000080
        u.SetWindowLongW(hwnd, GWL_EXSTYLE, u.GetWindowLongW(hwnd, GWL_EXSTYLE) | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW)
    except Exception:
        pass


class Api:
    """The page tells us which rectangles hold buttons, so the rest of the window lets clicks through to the app behind."""
    window = None

    def regions(self, rects):
        if not sys.platform.startswith("win") or self.window is None:
            return
        try:
            u, g = ctypes.windll.user32, ctypes.windll.gdi32
            hwnd = int(self.window.native.Handle.ToInt64())
            rgn = g.CreateRectRgn(0, 0, 0, 0)
            for x, y, w, h in rects[:12]:
                part = g.CreateRectRgn(int(x), int(y), int(x + w), int(y + h))
                g.CombineRgn(rgn, rgn, part, 2)                      # RGN_OR
                g.DeleteObject(part)
            u.SetWindowRgn(hwnd, rgn, True)
        except Exception:
            pass


def main() -> int:
    try:
        import webview
    except ImportError:
        print("Overlay mode needs pywebview.  Run:  pip install pywebview")
        return 1
    runner = web.AppRunner(make_app())
    loop_ready = threading.Event()

    def serve():
        import asyncio
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, HOST, PORT).start())
        loop_ready.set(); loop.run_forever()

    threading.Thread(target=serve, daemon=True).start()
    loop_ready.wait(10)
    w, h = 1280, 800
    try:
        import tkinter
        r = tkinter.Tk(); w, h = r.winfo_screenwidth(), r.winfo_screenheight(); r.destroy()
    except Exception:
        pass
    api = Api()
    win = webview.create_window("Blue Voice", f"http://{HOST}:{PORT}/?overlay=1", x=0, y=0, width=w, height=h, js_api=api,
                                frameless=True, transparent=True, on_top=True, easy_drag=False, background_color="#000000")
    api.window = win
    webview.start(lambda: (time.sleep(1.0), _no_activate(win)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
