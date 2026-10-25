"""Sign-word page: live word recognition + record-your-own-words.

    python words_app.py        -> http://127.0.0.1:8766   (opens your browser)

Everything runs on this computer. Protocol (JSON over WebSocket at /ws):
  page -> app : {"cmd":"start","source":"camera"|"video","path":"..."} | {"cmd":"stop"} | {"cmd":"record","word":"..."}
                {"cmd":"cancel"} | {"cmd":"delete","word":"..."} | {"cmd":"list"}
  app -> page : status / frame / moving / sign / teach / words / error   (see core/word_session.py)
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import webbrowser

from aiohttp import web, WSMsgType

HOST, PORT = "127.0.0.1", 8766


def base_dir() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


class Hub:
    def __init__(self, session_factory=None):
        self.clients: set = set()
        self.loop = None
        self.session = None
        self.session_factory = session_factory

    async def _send_all(self, msg: dict):
        data = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    def emit(self, msg: dict):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._send_all(msg)))

    def get_session(self):
        if self.session is None:
            self.session = (self.session_factory or self._default)(self.emit)
        return self.session

    def _default(self, emit):
        from core.word_session import WordSession
        return WordSession(base_dir(), emit)


async def ws_handler(request: web.Request):
    hub: Hub = request.app["hub"]
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    hub.clients.add(ws)
    run = asyncio.get_running_loop().run_in_executor
    try:
        s = await run(None, hub.get_session)
        await ws.send_str(json.dumps({"type": "status", "state": s.state, "note": ""}))
        await ws.send_str(json.dumps({"type": "words", "words": s.words()}))
        async for m in ws:
            if m.type != WSMsgType.TEXT:
                continue
            try:
                c = json.loads(m.data)
            except ValueError:
                continue
            name = c.get("cmd")
            if name == "start":
                await run(None, s.start, str(c.get("source", "camera")), str(c.get("path", "")))
            elif name == "stop":
                await run(None, s.stop); await hub._send_all({"type": "status", "state": "idle", "note": ""})
            elif name == "record":
                err = s.record(str(c.get("word", "")))
                if err: await ws.send_str(json.dumps({"type": "error", "message": err}))
            elif name == "cancel":
                s.cancel_record()
            elif name == "delete":
                s.delete_word(str(c.get("word", "")))
            elif name == "list":
                s.send_words()
    except Exception as e:
        try: await ws.send_str(json.dumps({"type": "error", "message": str(e)}))
        except Exception: pass
    finally:
        hub.clients.discard(ws)
    return ws


async def index(request: web.Request):
    return web.FileResponse(os.path.join(base_dir(), "web", "words.html"))


def make_app(hub: Hub | None = None) -> web.Application:
    app = web.Application()
    app["hub"] = hub or Hub()
    async def on_start(a): a["hub"].loop = asyncio.get_running_loop()
    async def on_clean(a):
        if a["hub"].session is not None: a["hub"].session.stop()
    app.on_startup.append(on_start); app.on_cleanup.append(on_clean)
    app.router.add_get("/", index); app.router.add_get("/ws", ws_handler)
    return app


if __name__ == "__main__":
    print(f"Sign-word page: http://{HOST}:{PORT}   (close this window to stop)")
    try:
        webbrowser.open(f"http://{HOST}:{PORT}")
    except Exception:
        pass
    web.run_app(make_app(), host=HOST, port=PORT, print=None)
