"""Server wiring for the AI agent and the palm-to-send routing (fake injector, fake model)."""
import asyncio, json, os, sys, tempfile, types
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from aiohttp.test_utils import TestClient, TestServer
import server
from core.gesture_recognizer import Gesture
from core.inject import Injector
from core.agent.agent import Agent


class FakeBackend:
    def __init__(self): self.pasted = []; self.watching = False
    def watch(self): self.watching = True
    def unwatch(self): self.watching = False
    def paste(self, t): self.pasted.append(t); return True, "Typed into “Fake app”."


async def until(ws, pred, t=10):
    end = asyncio.get_event_loop().time() + t
    while asyncio.get_event_loop().time() < end:
        try: m = await asyncio.wait_for(ws.receive_json(), 1)
        except asyncio.TimeoutError: continue
        if pred(m): return m
    raise AssertionError("timeout")


async def scenario():
    os.environ["GC_DATA"] = tempfile.mkdtemp()
    ctl = server.Controller(); fb = FakeBackend(); ctl.injector = Injector(fb)
    async with TestClient(TestServer(server.make_app(ctl))) as cl:
        ws = await cl.ws_connect("/ws")
        await ws.send_json({"cmd": "llm_set", "config": {"provider": "ollama", "workspace": tempfile.mkdtemp(), "api_key": "sk-secret-9999"}})
        m = await until(ws, lambda m: m["type"] == "llm")
        assert m["ready"] and m["has_key"] and "secret" not in json.dumps(m) and "ollama" in m["presets"]
        # agent mode: palm sends the signed text to the model
        ctl.get_agent().chat_fn = lambda cfg, sys_, msgs, tools: {"text": "Hello! You said: " + msgs[-1]["content"], "tool_calls": []}
        await ws.send_json({"cmd": "target", "mode": "agent"})
        await asyncio.sleep(0.1)
        ctl.engine.clear(); [ctl.engine.append_character(c) for c in "hi there"]
        ctl._on_gesture(types.SimpleNamespace(gesture=Gesture.OPEN_PALM))
        a = await until(ws, lambda m: m["type"] == "agent" and m["event"] == "text")
        assert a["text"] == "Hello! You said: hi there" and ctl.engine.text == ""
        await until(ws, lambda m: m["type"] == "agent" and m["event"] == "done")
        # overlay mode: palm pastes into the app in front, never presses Send
        await ws.send_json({"cmd": "target", "mode": "overlay"})
        await asyncio.sleep(0.1)
        assert fb.watching
        [ctl.engine.append_character(c) for c in "help me"]
        ctl._on_gesture(types.SimpleNamespace(gesture=Gesture.OPEN_PALM))
        i = await until(ws, lambda m: m["type"] == "injected")
        assert i["ok"] and fb.pasted == ["help me"] and ctl.engine.text == ""
        # nothing signed: nothing pasted
        ctl._on_gesture(types.SimpleNamespace(gesture=Gesture.OPEN_PALM)); await asyncio.sleep(0.2); assert fb.pasted == ["help me"]
        await ws.send_json({"cmd": "target", "mode": "none"}); await asyncio.sleep(0.1); assert not fb.watching
        # no key and no model chosen: a plain message, not a crash
        await ws.send_json({"cmd": "llm_set", "config": {"provider": "anthropic", "clear_key": True}})
        await until(ws, lambda m: m["type"] == "llm" and not m["ready"])
        await ws.send_json({"cmd": "agent_send", "text": "hello"})
        e = await until(ws, lambda m: m["type"] == "agent" and m["event"] == "error"); assert "API key" in e["text"]


def test_ai_wiring():
    asyncio.run(scenario())


async def words_scenario():
    import numpy as np, collections, time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import core.word_session as W
    from test_record import PacedFake, palm_frame
    from test_words_app import clip, EMPTY, ROOT
    os.environ["GC_DATA"] = tempfile.mkdtemp()
    fake = PacedFake()
    tmp = tempfile.mkdtemp(); os.makedirs(tmp + "/models"); os.symlink(ROOT + "/models/gs", tmp + "/models/gs")
    ctl = server.Controller(); fb = FakeBackend(); ctl.injector = Injector(fb)
    ctl.words = W.WordSession(tmp, ctl.emit, tracker_factory=lambda: fake, source_factory=lambda s, p: fake)
    async with TestClient(TestServer(server.make_app(ctl))) as cl:
        ws = await cl.ws_connect("/ws")
        await ws.send_json({"cmd": "target", "mode": "overlay"}); await asyncio.sleep(0.1)
        await ws.send_json({"cmd": "start", "source": "words"})
        gap = [EMPTY] * 15
        fake.feed(gap + clip("blue", 1.0, 0.002, 1) + gap * 2 + [palm_frame()] * 60 + gap * 3)
        l = await until(ws, lambda m: m["type"] == "letter", 20)
        assert l["label"] == "blue"
        i = await until(ws, lambda m: m["type"] == "injected", 20)
        assert i["ok"] and fb.pasted == ["blue"], fb.pasted
        await ws.send_json({"cmd": "stop"}); await asyncio.sleep(0.3)


def test_signed_word_then_palm_sends():
    asyncio.run(words_scenario())
