"""
Local bridge: runs the hand-tracking core and serves the Glass Chat UI.

    python main.py          -> http://127.0.0.1:8765  (opens your browser)

Protocol (JSON over WebSocket at /ws)
  record a sign : {"cmd":"rec_start","word":"..."} | rec_again | rec_finish | rec_save | rec_cancel | {"cmd":"mysign_delete","word":"..."} | mysigns_list
                  server -> UI {"type":"rec","state":"loading|recording|review|saved|cancelled|error","word","takes","left"} and {"type":"mysigns","signs":[...]}
  AI : {"cmd":"llm_get"|"llm_set","config":{...}|"agent_send","text"|"agent_stop"|"agent_reset"|"chat_list"|"chat_open","id"|"chat_delete","id"|"agent_approve","id","ok"|"target","mode":"agent"|"overlay"|"none"}
       server -> UI {"type":"llm",...public settings} {"type":"agent","event":"thinking|text|tool|approval|error|done"} {"type":"injected","ok","message"}
  UI -> server : {"cmd":"start","source":"camera"|"simulated"|"mock","path":"..."} | {"cmd":"stop"} | {"cmd":"clear"}
  server -> UI : {"type":"status","state":"idle"|"tracking","source":str,"letters":bool}
                 {"type":"text","text":str}
                 {"type":"letter","label":str}           (a letter was committed)
                 {"type":"gesture","gesture":"SWIPE_LEFT"|"SWIPE_RIGHT"|"OPEN_PALM"}
                 {"type":"submit","text":str}            (palm held 1.5 s)
                 {"type":"error","message":str}
All processing stays on this computer; nothing is sent anywhere else.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import threading
from typing import Optional

from aiohttp import web, WSMsgType

from core.gesture_recognizer import Gesture
from core.hand_tracker import HandTrackerThread
from core.letter_classifier import LetterClassifier, LetterCommitter
from core.text_engine import TextEngine

HOST, PORT = "127.0.0.1", 8765


def base_dir() -> str:
    return getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))


class Controller:
    """Owns the tracker, text buffer and connected UI clients."""

    def __init__(self, classifier: Optional[LetterClassifier] = None) -> None:
        self.engine = TextEngine()
        self.classifier = classifier if classifier is not None else LetterClassifier()
        self.committer = LetterCommitter()
        self.clients: set = set()
        self.tracker: Optional[HandTrackerThread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.state = "idle"
        self.source = ""
        self.words = None                       # WordSession, created the first time someone records a sign
        self.llm_cfg = None
        self.agent = None
        self.target = "none"                    # where a palm-submitted message goes: "agent" (this app), "overlay" (the app in front), or "none"
        self.injector = None

    def word_session(self):
        if self.words is None:
            from core.word_session import WordSession
            self.words = WordSession(base_dir(), self.emit)
        return self.words

    # -- accounts, updates, sign packs --------------------------------- #
    def oauth(self):
        if getattr(self, "_oauth", None) is None:
            from core.oauth import OAuth
            self._oauth = OAuth(os.environ.get("GC_DATA", os.path.join(base_dir(), "data")))
        return self._oauth

    def app_cmd(self, name: str, cmd: dict) -> None:
        from core.oauth import SignInError
        o = self.oauth()
        if name == "oauth_state":
            self.emit({"type": "auth", "event": "state", **o.available()}); return
        if name == "oauth_cancel":
            o.gh_cancel.set(); return
        if name == "oauth_github":
            try:
                o.gh_cancel = threading.Event()
                st = o.github_start()
                self.emit({"type": "auth", "event": "code", "user_code": st["user_code"], "uri": st["uri"]})
                self.open_browser(st["uri"])
                self.emit({"type": "auth", "event": "ok", **o.github_wait(st)})
            except SignInError as e:
                self.emit({"type": "auth", "event": "error", "text": str(e)})
            return
        if name == "oauth_google":
            try:
                port = int(cmd.get("port") or PORT)
                redirect = f"http://{HOST}:{port}/oauth/google"
                self.emit({"type": "auth", "event": "waiting"})
                self.open_browser(o.google_url(redirect))
            except SignInError as e:
                self.emit({"type": "auth", "event": "error", "text": str(e)})
            return
        if name == "update_check":
            self.emit({"type": "update", **self.check_update()}); return
        ws = self.word_session()
        from core import packs
        meta = ws.model_info()
        if name == "pack_export":
            self.emit({"type": "pack", "pack": packs.export_pack(ws.store, meta["version"], meta["dim"])}); return
        if name == "pack_import":
            try:
                w, t = packs.import_pack(ws.store, cmd.get("pack"), meta["version"], meta["dim"])
                ws.send_signs()
                self.emit({"type": "packdone", "ok": True, "text": f"Added {w} sign{'s' if w != 1 else ''} ({t} recordings)."})
            except ValueError as e:
                self.emit({"type": "packdone", "ok": False, "text": str(e)})

    def open_browser(self, url: str) -> None:
        import webbrowser
        webbrowser.open(url)

    def check_update(self) -> dict:
        from core import version
        out = {"current": version.__version__, "available": False}
        if not version.REPO:
            return {**out, "error": "Updates are not switched on yet for this copy of Blue Voice."}
        import urllib.request, urllib.error
        try:
            req = urllib.request.Request(f"https://api.github.com/repos/{version.REPO}/releases/latest", headers={"User-Agent": "BlueVoice", "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                j = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return {**out, "error": "No release has been published yet." if e.code == 404 else f"GitHub answered with an error ({e.code})."}
        except Exception:
            return {**out, "error": "Could not reach GitHub. Check your internet connection."}
        latest = str(j.get("tag_name", "")).lstrip("vV")
        def num(v): return tuple(int(x) for x in re.findall(r"\d+", v)[:3])
        try:
            newer = num(latest) > num(version.__version__)
        except Exception:
            newer = False
        return {**out, "latest": latest, "available": newer, "url": j.get("html_url", ""), "notes": (j.get("body") or "")[:600]}

    # -- AI agent ------------------------------------------------------ #
    def llm(self):
        if self.llm_cfg is None:
            from core.agent.config import LLMConfig
            self.llm_cfg = LLMConfig(os.path.join(os.environ.get("GC_DATA", os.path.join(base_dir(), "data")), "llm.json"))
        return self.llm_cfg

    def get_agent(self):
        if self.agent is None:
            from core.agent.agent import Agent
            from core.agent.store import ChatStore
            self.agent = Agent(self.llm(), self.emit, store=ChatStore(os.path.join(os.path.dirname(self.llm().path), "chats")))
        return self.agent

    def agent_cmd(self, name: str, cmd: dict) -> Optional[str]:
        if name == "llm_get":
            self.emit({"type": "llm", **self.llm().public()}); return None
        if name == "llm_set":
            self.llm().update(cmd.get("config") or {})
            self.emit({"type": "llm", **self.llm().public()}); return None
        if name == "agent_send":
            return self.get_agent().send(str(cmd.get("text", "")))
        if name == "agent_stop":
            if self.agent: self.agent.stop()
            return None
        if name == "agent_reset":
            self.get_agent().reset()
            self.emit({"type": "agent", "event": "chats", "items": self.get_agent().chats()})
            return None
        if name == "chat_list":
            self.emit({"type": "agent", "event": "chats", "items": self.get_agent().chats()}); return None
        if name == "chat_open":
            msgs = self.get_agent().open_chat(str(cmd.get("id", "")))
            if msgs is None:
                return "That chat was not found."
            self.emit({"type": "agent", "event": "opened", "messages": msgs}); return None
        if name == "chat_delete":
            self.get_agent().delete_chat(str(cmd.get("id", "")))
            self.emit({"type": "agent", "event": "chats", "items": self.get_agent().chats()}); return None
        if name == "agent_approve":
            if self.agent: self.agent.approve(str(cmd.get("id", "")), bool(cmd.get("ok")))
            return None
        if name == "target":
            self.target = cmd.get("mode") if cmd.get("mode") in ("agent", "overlay") else "none"
            if self.target == "overlay":
                if self.injector is None:
                    from core.inject import Injector
                    self.injector = Injector()
                self.injector.start_watching()
            elif self.injector:
                self.injector.stop_watching()
            return None
        return None

    def route_submit(self) -> None:
        """The palm was shown: send the signed text where the person chose."""
        text = self.engine.text.strip()
        if not text:
            self.emit({"type": "submit", "text": ""}); return
        if self.target == "agent":
            err = self.get_agent().send(text)
            if err:
                self.emit({"type": "error", "message": err}); return
        elif self.target == "overlay":
            ok, msg = self.injector.paste(text) if self.injector else (False, "Overlay mode is not on.")
            self.emit({"type": "injected", "ok": ok, "message": msg})
            if not ok:
                return
        else:
            self.emit({"type": "submit", "text": text}); return
        self.engine.clear(); self.committer.reset()
        self.emit({"type": "submit", "text": text, "sent": True})
        self.emit({"type": "text", "text": ""})

    # -- messaging ---------------------------------------------------- #
    def status_msg(self) -> dict:
        return {"type": "status", "state": self.state, "source": self.source,
                "letters": self.classifier.available}

    async def _send_all(self, msg: dict) -> None:
        data = json.dumps(msg)
        for ws in list(self.clients):
            try:
                await ws.send_str(data)
            except Exception:
                self.clients.discard(ws)

    def emit(self, msg: dict) -> None:
        """Thread-safe: may be called from the tracker thread."""
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._send_all(msg)))

    # -- tracker callbacks (tracker thread) ----------------------------- #
    def _on_landmarks(self, ts: float, pts) -> None:
        label, conf = (None, 0.0) if pts is None else self.classifier.predict(pts)
        ch = self.committer.update(label, conf)
        if ch:
            self.engine.append_character(ch)
            self.emit({"type": "letter", "label": ch})
            self.emit({"type": "text", "text": self.engine.text})

    def _on_gesture(self, event) -> None:
        g = event.gesture
        if g is Gesture.SWIPE_LEFT:
            self.engine.backspace()
        elif g is Gesture.SWIPE_RIGHT:
            self.engine.add_space()
        elif g is Gesture.OPEN_PALM:
            self.emit({"type": "gesture", "gesture": g.name})
            self.route_submit()
            return
        self.emit({"type": "gesture", "gesture": g.name})
        self.emit({"type": "text", "text": self.engine.text})

    def _on_state(self, state: str, source: str) -> None:
        self.state = "tracking" if state == "started" else "idle"
        self.source = source if state == "started" else ""
        self.emit(self.status_msg())

    def _on_error(self, message: str) -> None:
        self.emit({"type": "error", "message": message})

    # -- commands ------------------------------------------------------- #
    def _on_word(self, word: str, score: float) -> None:
        """A word sign was read: add it to the message."""
        for ch in word:
            self.engine.append_character(ch)
        self.engine.add_space()
        self.emit({"type": "letter", "label": word})
        self.emit({"type": "text", "text": self.engine.text})

    def start(self, source: str, path: str = "") -> Optional[str]:
        """Start tracking. Returns an error message, or None on success."""
        self.stop()
        self.committer.reset()
        if source == "words":                       # signed words (250 built in + the person's own recorded signs)
            ws = self.word_session()
            ws.on_word, ws.on_palm, ws.preview = self._on_word, self.route_submit, False
            ws.start("camera")
            self.state, self.source = "tracking", "words"
            self.emit(self.status_msg())
            return None
        t = HandTrackerThread(on_landmarks=self._on_landmarks, on_gesture=self._on_gesture,
                              on_state=self._on_state, on_error=self._on_error)
        if source == "simulated":
            ok = t.start_simulated_test()
        elif source == "mock":
            ok = t.start_mock_video(path)
            if not ok:
                return "Could not open that video file."
        else:
            ok = t.start_live_camera()
            if not ok:
                return "No camera found, or the hand model could not be loaded."
        self.tracker = t if ok else None
        return None

    def stop(self) -> None:
        if self.words is not None and self.source == "words":
            self.words.on_word = self.words.on_palm = None
            self.words.stop()
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None
        self.state, self.source = "idle", ""

    def sign_cmd(self, name: str, cmd: dict) -> Optional[str]:
        """Record-your-own-sign commands (run off the event loop). Returns an error text or None."""
        ws = self.word_session()
        if name in ("rec_arm", "rec_start"):
            if self.tracker is not None:
                self.stop()                      # the letter tracker and the recorder must not share the camera
                self.emit(self.status_msg())
            if name == "rec_arm":
                return ws.rec_arm(str(cmd.get("word", "")))
            return ws.rec_start(str(cmd.get("word", "")))
        if name == "rec_word":
            ws.rec_word_set(str(cmd.get("word", ""))); return None
        if name == "rec_again":
            return ws.rec_again()
        if name == "rec_finish":
            ws.rec_finish(); return None
        if name == "rec_save":
            return ws.rec_save()
        if name == "rec_cancel":
            ws.rec_cancel(); return None
        if name == "mysign_delete":
            ws.delete_sign(str(cmd.get("word", ""))); return None
        if name == "mysigns_list":
            ws.send_signs(); return None
        return None

    def clear(self) -> None:
        self.engine.clear()
        self.committer.reset()


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    ctl: Controller = request.app["ctl"]
    ws = web.WebSocketResponse(heartbeat=20, max_msg_size=16 * 1024 * 1024)
    await ws.prepare(request)
    ctl.clients.add(ws)
    await ws.send_str(json.dumps(ctl.status_msg()))
    await ws.send_str(json.dumps({"type": "text", "text": ctl.engine.text}))
    try:
        async for m in ws:
            if m.type != WSMsgType.TEXT:
                continue
            try:
                cmd = json.loads(m.data)
            except ValueError:
                continue
            name = cmd.get("cmd")
            if name == "start":
                err = await asyncio.get_running_loop().run_in_executor(
                    None, ctl.start, str(cmd.get("source", "camera")), str(cmd.get("path", "")))
                if err:
                    await ws.send_str(json.dumps({"type": "error", "message": err}))
                await ws.send_str(json.dumps(ctl.status_msg()))
            elif name == "stop":
                await asyncio.get_running_loop().run_in_executor(None, ctl.stop)
                await ws.send_str(json.dumps(ctl.status_msg()))
            elif name in ("llm_get", "llm_set", "agent_send", "agent_stop", "agent_reset", "agent_approve", "chat_list", "chat_open", "chat_delete", "target"):
                err = await asyncio.get_running_loop().run_in_executor(None, ctl.agent_cmd, name, cmd)
                if err:
                    await ws.send_str(json.dumps({"type": "agent", "event": "error", "text": err}))
            elif name in ("rec_arm", "rec_word", "rec_start", "rec_again", "rec_finish", "rec_save", "rec_cancel", "mysign_delete", "mysigns_list"):
                err = await asyncio.get_running_loop().run_in_executor(None, ctl.sign_cmd, name, cmd)
                if err:
                    await ws.send_str(json.dumps({"type": "rec", "state": "error", "message": err}))
            elif name in ("oauth_state", "oauth_cancel", "oauth_github", "oauth_google", "update_check", "pack_export", "pack_import"):
                await asyncio.get_running_loop().run_in_executor(None, ctl.app_cmd, name, cmd)
            elif name == "clear":
                ctl.clear()
                await ctl._send_all({"type": "text", "text": ""})
    finally:
        ctl.clients.discard(ws)
    return ws


async def index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(os.path.join(base_dir(), "web", "index.html"))


PAGE = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Blue Voice</title>"
        "<body style='font-family:system-ui;background:#cfe6fd;color:#0d3a73;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'>"
        "<div style='text-align:center'><h1>%s</h1><p>%s</p></div>")


async def oauth_google(request: web.Request) -> web.StreamResponse:
    ctl: Controller = request.app["ctl"]
    if request.host.split(":")[0] not in (HOST, "localhost"):
        return web.Response(status=403, text="Not allowed")
    from core.oauth import SignInError
    q = request.query
    if q.get("error") or not q.get("code"):
        ctl.emit({"type": "auth", "event": "error", "text": "Google sign-in was cancelled."})
        return web.Response(text=PAGE % ("Sign-in cancelled", "You can close this tab and go back to Blue Voice."), content_type="text/html")
    try:
        user = await asyncio.get_running_loop().run_in_executor(None, ctl.oauth().google_finish, q.get("state", ""), q["code"])
    except SignInError as e:
        ctl.emit({"type": "auth", "event": "error", "text": str(e)})
        return web.Response(text=PAGE % ("Could not sign in", str(e)), content_type="text/html")
    ctl.emit({"type": "auth", "event": "ok", **user})
    return web.Response(text=PAGE % ("You are signed in", "You can close this tab and go back to Blue Voice."), content_type="text/html")


def make_app(controller: Optional[Controller] = None) -> web.Application:
    app = web.Application()
    ctl = controller or Controller()
    app["ctl"] = ctl

    async def on_start(a):
        ctl.loop = asyncio.get_running_loop()

    async def on_cleanup(a):
        ctl.stop()

    app.on_startup.append(on_start)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/oauth/google", oauth_google)
    return app
