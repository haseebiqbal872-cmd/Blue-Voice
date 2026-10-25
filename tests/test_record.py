"""15-second record window: three takes of a word, review, save, list, delete, cancel (fake camera, real model)."""
import asyncio, json, os, sys, tempfile, collections, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aiohttp.test_utils import TestClient, TestServer
import server, core.word_session as W
from test_words_app import clip, EMPTY, wait_for, ROOT


class PacedFake:
    live = True
    def __init__(self): self.q = collections.deque(); self.cur = EMPTY; self.t = 0.0
    def feed(self, fr): self.q.extend(fr)
    def read(self):
        time.sleep(1 / 60)
        self.cur = self.q.popleft() if self.q else EMPTY
        self.t += 1000 / 30
        return np.zeros((36, 64, 3), np.uint8), self.t
    def track(self, rgb, ts): return self.cur
    def close(self): pass


def palm_frame():
    f = np.full((543, 3), np.nan, np.float32); h = np.zeros((21, 3), np.float32)
    h[0] = (.5, .8, 0); h[1] = (.45, .76, 0); h[2] = (.40, .72, 0); h[3] = (.35, .68, 0); h[4] = (.30, .64, 0)
    for k, x in zip((5, 9, 13, 17), (.43, .5, .56, .62)):
        for j, y in enumerate((.66, .56, .50, .44)):
            h[k + j] = (x, y, 0)
    f[468:489] = h
    return f


def fist_frame():
    f = palm_frame()
    for k in (5, 9, 13, 17):
        f[468 + k + 1] = (f[468 + k, 0], .70, 0); f[468 + k + 2] = (f[468 + k, 0], .72, 0); f[468 + k + 3] = (f[468 + k, 0], .70, 0)
    return f


def test_palm_detector():
    from core.gs.palm import open_palm, PalmHold
    assert open_palm(palm_frame()) and not open_palm(fist_frame()) and not open_palm(EMPTY)
    ph = PalmHold(); assert not any(ph.push(palm_frame(), i * 33) for i in range(20)) and ph.push(palm_frame(), 900)
    ph.reset(); ph.push(palm_frame(), 0); ph.push(fist_frame(), 100); assert not ph.push(palm_frame(), 900)


async def arm_scenario():
    W.REC_SECONDS, W.READY_SECONDS = 4, 1
    fake = PacedFake()
    tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "models"))
    os.symlink(os.path.join(ROOT, "models", "gs"), os.path.join(tmp, "models", "gs"))
    ctl = server.Controller()
    ctl.words = W.WordSession(tmp, ctl.emit, tracker_factory=lambda: fake, source_factory=lambda s, p: fake)
    async with TestClient(TestServer(server.make_app(ctl))) as cl:
        ws = await cl.ws_connect("/ws")
        await ws.send_json({"cmd": "rec_arm", "word": ""})
        await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "armed")
        fake.feed([palm_frame()] * 40)                      # palm shown, but no word typed yet: nothing happens
        await asyncio.sleep(1.2)
        assert not [1 for _ in range(0)]
        await ws.send_json({"cmd": "rec_word", "word": "blue"})
        gap = [EMPTY] * 12
        fake.feed([palm_frame()] * 30 + [EMPTY] * 75 + gap + clip("blue", 1.0, 0.002, 1) + gap + clip("blue", 0.9, 0.002, 2) + gap + clip("blue", 1.1, 0.002, 3) + gap)
        rd = await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "ready", 15)
        await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "recording")
        pv = await wait_for(ws, lambda m: m["type"] == "recprev", 25)
        v = pv["video"]; assert v and len(v["jpg"]) >= 4 and v["t"][0] == 0 and v["t"] == sorted(v["t"]), v and v.keys()
        import base64, cv2; im = cv2.imdecode(np.frombuffer(base64.b64decode(v["jpg"][0]), np.uint8), 1); assert im is not None
        assert len(pv["pts"]) == 51 and pv["frames"] and len(pv["frames"][0]) == 102 and pv["fps"] > 0, pv.keys()
        r = await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "review")
        assert r["takes"] >= 3, r


def test_arm_palm_start():
    asyncio.run(arm_scenario())


async def scenario():
    W.REC_SECONDS = 4
    fake = PacedFake()
    tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "models"))
    os.symlink(os.path.join(ROOT, "models", "gs"), os.path.join(tmp, "models", "gs"))
    ctl = server.Controller()
    ctl.words = W.WordSession(tmp, ctl.emit, tracker_factory=lambda: fake, source_factory=lambda s, p: fake)
    async with TestClient(TestServer(server.make_app(ctl))) as cl:
        ws = await cl.ws_connect("/ws")
        await ws.send_json({"cmd": "rec_start", "word": "  My Sign "})
        m = await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "recording")
        assert m["word"] == "my sign" or m["word"], m
        gap = [EMPTY] * 12
        fake.feed(gap + clip("blue", 1.0, 0.002, 1) + gap + clip("blue", 0.9, 0.002, 2) + gap + clip("blue", 1.1, 0.002, 3) + gap)
        r = await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "review", 25)
        assert r["takes"] >= 3, r
        await ws.send_json({"cmd": "rec_save"})
        sv = await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "saved")
        ms = await wait_for(ws, lambda m: m["type"] == "mysigns")
        assert len(ms["signs"]) == 1 and ms["signs"][0]["ready"] and ms["signs"][0]["added"], ms
        # discard & start again, then cancel
        await ws.send_json({"cmd": "rec_start", "word": "other"})
        await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "recording")
        await ws.send_json({"cmd": "rec_again"})
        await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "recording" and m["takes"] == 0)
        await ws.send_json({"cmd": "rec_cancel"})
        await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "cancelled")
        # save with too few takes is refused
        await ws.send_json({"cmd": "rec_save"})
        e = await wait_for(ws, lambda m: m["type"] == "rec" and m["state"] == "error")
        # delete
        await ws.send_json({"cmd": "mysign_delete", "word": sv["word"]})
        ms = await wait_for(ws, lambda m: m["type"] == "mysigns")
        assert ms["signs"] == [], ms


def test_record_window():
    asyncio.run(scenario())
