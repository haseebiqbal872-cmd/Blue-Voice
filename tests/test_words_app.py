"""End-to-end test of the word page back end with a fake camera and a fake tracker (real model, real clips)."""
import asyncio, json, os, sys, tempfile, threading, collections, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aiohttp.test_utils import TestClient, TestServer
import words_app
from core.word_session import WordSession

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
d = json.load(open(os.path.join(ROOT, "models", "gs", "replay_clips.json"))); pts = d["points"]
EMPTY = np.full((543, 3), np.nan, np.float32)


def clip(title, speed=1.0, noise=0.0, seed=0):
    c = [c for c in d["clips"] if c["title"] == title][0]
    fr = np.full((len(c["frames"]), 543, 3), np.nan, np.float32)
    for t, row in enumerate(c["frames"]):
        fr[t, pts, :2] = np.array([np.nan if v is None else v for v in row], np.float32).reshape(-1, 2)
    idx = np.clip(np.round(np.arange(0, len(fr) - 1, speed)).astype(int), 0, len(fr) - 1)
    fr = fr[idx]; fr[..., :2] += np.random.default_rng(seed).normal(0, noise, fr[..., :2].shape).astype(np.float32)
    return list(fr)


class Fake:
    live = False
    def __init__(self): self.q = collections.deque(); self.cur = None; self.t = 0.0; self.stop = False
    def feed(self, frames): self.q.extend(frames)
    def read(self):
        while not self.q:
            if self.stop: return None
            time.sleep(0.005)
        self.cur = self.q.popleft(); self.t += 1000 / 30
        return np.zeros((36, 64, 3), np.uint8), self.t
    def track(self, rgb, ts): return self.cur
    def close(self): self.stop = True


async def wait_for(ws, pred, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        try: m = await asyncio.wait_for(ws.receive_json(), 1.0)
        except asyncio.TimeoutError: continue
        if pred(m): return m
    raise AssertionError("timed out")


async def scenario():
    fake = Fake()
    tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "models"))
    os.symlink(os.path.join(ROOT, "models", "gs"), os.path.join(tmp, "models", "gs"))
    hub = words_app.Hub(lambda emit: WordSession(tmp, emit, tracker_factory=lambda: fake, source_factory=lambda s, p: fake))
    async with TestClient(TestServer(words_app.make_app(hub))) as cl:
        ws = await cl.ws_connect("/ws")
        await ws.send_json({"cmd": "start", "source": "video", "path": "x"})
        await wait_for(ws, lambda m: m["type"] == "status" and m["state"] == "tracking")
        await ws.send_json({"cmd": "record", "word": "mysign"})
        await wait_for(ws, lambda m: m["type"] == "teach" and m["state"] == "waiting")
        gap = [EMPTY] * 15
        for i, sp in enumerate((0.9, 1.0, 1.1)):
            fake.feed(gap + clip("blue", sp, 0.002, i) + gap)
            m = await wait_for(ws, lambda m: m["type"] == "teach" and m["state"] in ("saved", "done"))
            print("teach:", m["state"], m["take"], "/", m["of"])
        assert m["state"] == "done"
        fake.feed(gap + clip("blue", 1.05, 0.004, 9) + gap)
        m = await wait_for(ws, lambda m: m["type"] == "sign"); print("blue again ->", m["source"], m["top"][0])
        assert m["source"] == "yours" and m["top"][0][0] == "mysign"
        fake.feed(gap + clip("police") + gap)
        m = await wait_for(ws, lambda m: m["type"] == "sign"); print("police ->", m["source"], m["top"][0])
        assert m["source"] == "built-in" and m["top"][0][0] == "police"
        await ws.send_json({"cmd": "delete", "word": "mysign"})
        m = await wait_for(ws, lambda m: m["type"] == "words" and not m["words"]); print("deleted, words:", m["words"])
        await ws.send_json({"cmd": "stop"})
        await wait_for(ws, lambda m: m["type"] == "status" and m["state"] == "idle")


def test_app():
    asyncio.run(scenario())

if __name__ == "__main__":
    test_app(); print("OK")
