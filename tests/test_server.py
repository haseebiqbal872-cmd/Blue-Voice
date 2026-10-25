import asyncio, json
import pytest
from aiohttp.test_utils import TestClient, TestServer
from core.gesture_recognizer import Gesture, GestureEvent
from server import make_app, Controller


class FakeClf:
    available = True
    def predict(self, pts): return "A", 0.99


async def _client(ctl=None):
    app = make_app(ctl or Controller(classifier=FakeClf()))
    c = TestClient(TestServer(app)); await c.start_server(); return c, app["ctl"]


async def _recv(ws, typ, n=20):
    for _ in range(n):
        m = json.loads((await asyncio.wait_for(ws.receive(), 3)).data)
        if m["type"] == typ: return m
    raise AssertionError("no " + typ)


def run(coro): return asyncio.run(coro)


def test_index_served_and_has_bridge():
    async def go():
        c, _ = await _client(); r = await c.get("/"); t = await r.text(); await c.close()
        assert r.status == 200 and "ws://" in t and "SIGNLIVE" in t
    run(go())


def test_ws_status_and_simulated_start_stop():
    async def go():
        c, ctl = await _client(); ws = await c.ws_connect("/ws")
        st = await _recv(ws, "status"); assert st["state"] == "idle" and st["letters"] is True
        await ws.send_str(json.dumps({"cmd": "start", "source": "simulated"}))
        st = await _recv(ws, "status")
        # tracking begins asynchronously; wait for tracking state
        for _ in range(20):
            if st["state"] == "tracking": break
            st = await _recv(ws, "status")
        assert st["state"] == "tracking"
        await ws.send_str(json.dumps({"cmd": "stop"})); await ws.close(); await c.close()
    run(go())


def test_letters_gestures_and_text_flow():
    async def go():
        c, ctl = await _client(); ws = await c.ws_connect("/ws"); await _recv(ws, "text")
        for _ in range(12): ctl._on_landmarks(0, [(0.5, 0.5, 0)] * 21)
        m = await _recv(ws, "letter"); assert m["label"] == "A"
        t = await _recv(ws, "text"); assert t["text"] == "A"
        ctl._on_gesture(GestureEvent(gesture=Gesture.SWIPE_RIGHT))
        t = await _recv(ws, "text"); assert t["text"] == "A "
        ctl._on_gesture(GestureEvent(gesture=Gesture.SWIPE_LEFT))
        t = await _recv(ws, "text"); assert t["text"] == "A"
        ctl._on_gesture(GestureEvent(gesture=Gesture.OPEN_PALM))
        s = await _recv(ws, "submit"); assert s["text"] == "A"
        await ws.send_str(json.dumps({"cmd": "clear"})); t = await _recv(ws, "text"); assert t["text"] == ""
        await ws.close(); await c.close()
    run(go())


def test_camera_missing_reports_error():
    async def go():
        c, ctl = await _client(); ws = await c.ws_connect("/ws"); await _recv(ws, "status")
        await ws.send_str(json.dumps({"cmd": "start", "source": "mock", "path": "/no/such.mp4"}))
        e = await _recv(ws, "error"); assert e["message"]
        await ws.close(); await c.close()
    run(go())
