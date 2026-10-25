"""Agent engine: wire formats against fake servers, file confinement, approvals, web safety, the loop."""
import json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest
from core.agent import providers
from core.agent.agent import Agent
from core.agent.config import LLMConfig
from core.agent.tools import Tools, ToolError

TOOLS = [{"name": "read_file", "description": "r", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}]
SEEN = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path.endswith("/chat/completions"):
            if self.headers.get("Authorization") != "Bearer sk-good":
                return self._j(401, {"error": {"message": "bad key"}})
            n = len(body["messages"])
            out = {"choices": [{"message": {"content": "hi from openai" if n < 4 else "done", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\":\"a.txt\"}"}}] if n == 2 else None}}]}
        elif self.path.endswith("/v1/messages"):
            out = {"content": [{"type": "text", "text": "hi from claude"}, {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a.txt"}}]}
        else:
            out = {"candidates": [{"content": {"parts": [{"text": "hi from gemini"}, {"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}}]}}]}
        self._j(200, out)
    def _j(self, code, o):
        d = json.dumps(o).encode(); self.send_response(code); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)


@pytest.fixture(scope="module")
def srv():
    s = HTTPServer(("127.0.0.1", 0), H); threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"; s.shutdown()


def test_openai_format_and_errors(srv):
    cfg = {"kind": "openai", "base_url": srv, "model": "m", "api_key": "sk-good"}
    msgs = [{"role": "user", "content": "x"}]
    r = providers.chat(cfg, "sys", msgs, TOOLS)
    assert r["text"] == "hi from openai" and r["tool_calls"][0]["args"] == {"path": "a.txt"}
    msgs += [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "name": "read_file", "args": {"path": "a.txt"}}]}]
    r = providers.chat(cfg, "sys", msgs, TOOLS)
    assert r["tool_calls"] == []
    body = SEEN[-1][2]; assert body["messages"][0]["role"] == "system" and body["tools"][0]["function"]["name"] == "read_file"
    with pytest.raises(providers.LLMError, match="key was not accepted"):
        providers.chat({**cfg, "api_key": "nope"}, "s", msgs[:1], [])


def test_anthropic_and_gemini_format(srv):
    c = {"kind": "anthropic", "base_url": srv, "model": "m", "api_key": "k"}
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"id": "t1", "name": "read_file", "args": {"path": "a"}}]},
            {"role": "tool", "tool_call_id": "t1", "name": "read_file", "content": "RES"}]
    r = providers.chat(c, "sys", msgs, TOOLS)
    assert r["text"] == "hi from claude" and r["tool_calls"][0]["id"] == "t1"
    path, hdr, body = SEEN[-1]
    assert hdr["x-api-key"] == "k" and body["messages"][-1]["content"][0]["type"] == "tool_result" and body["tools"][0]["input_schema"]
    g = {"kind": "gemini", "base_url": srv, "model": "gm", "api_key": "k"}
    r = providers.chat(g, "sys", msgs, TOOLS)
    assert r["text"] == "hi from gemini" and r["tool_calls"][0]["name"] == "read_file"
    path, hdr, body = SEEN[-1]
    assert path.endswith("/v1beta/models/gm:generateContent") and hdr["x-goog-api-key"] == "k" and "functionResponse" in body["contents"][-1]["parts"][0]


def test_files_confined_and_safe():
    d = tempfile.mkdtemp(); t = Tools(d)
    assert "Wrote" in t.run("write_file", {"path": "sub/a.txt", "content": "hello world"})
    assert t.run("read_file", {"path": "sub/a.txt"}) == "hello world"
    assert "sub/" in t.run("list_files", {"path": "."})
    t.run("edit_file", {"path": "sub/a.txt", "old": "world", "new": "there"}); assert open(d + "/sub/a.txt").read() == "hello there"
    for bad in ("../x.txt", "/etc/passwd", "sub/../../x", "..\\x"):
        with pytest.raises(ToolError):
            t.run("read_file", {"path": bad})
    os.symlink("/etc", d + "/link")
    with pytest.raises(ToolError):
        t.run("list_files", {"path": "link"})
    with pytest.raises(ToolError):
        t.run("edit_file", {"path": "sub/a.txt", "old": "zzz", "new": "q"})
    t.run("delete_file", {"path": "sub/a.txt"}); assert not os.path.exists(d + "/sub/a.txt")
    assert Tools("").specs() != [] and all(s["name"].startswith("web") for s in Tools("").specs())
    assert not any(s["name"].startswith("web") for s in Tools(d, web=False).specs())


def test_web_blocks_private():
    t = Tools("")
    for u in ("http://127.0.0.1:8765/ws", "http://localhost/", "http://192.168.1.1/", "http://169.254.169.254/latest", "file:///etc/passwd", "ftp://x.y/"):
        with pytest.raises(ToolError):
            t.run("web_fetch", {"url": u})
    assert "hello" in Tools._text("<html><script>x()</script><p>hello &amp; bye</p></html>")


def make_agent(script, auto=False, ws=None):
    d = tempfile.mkdtemp(); cfg = LLMConfig(d + "/llm.json")
    cfg.update({"provider": "ollama", "workspace": ws or d, "auto_approve": auto})
    ev = []
    calls = iter(script)
    return Agent(cfg, ev.append, chat_fn=lambda *a: next(calls)), ev, d


def wait(a):
    a.thread.join(10); assert not a.thread.is_alive()


def test_loop_reads_file_then_answers():
    a, ev, d = make_agent([{"text": "", "tool_calls": [{"id": "1", "name": "list_files", "args": {"path": "."}}]}, {"text": "Empty folder.", "tool_calls": []}])
    assert a.send("what is in my folder?") is None; wait(a)
    kinds = [e["event"] for e in ev]; assert kinds[-1] == "done" and "tool" in kinds and ev[-2]["text"] == "Empty folder."
    assert [m["role"] for m in a.history] == ["user", "assistant", "tool", "assistant"]


def test_write_needs_approval_yes_and_no():
    script = lambda: [{"text": "", "tool_calls": [{"id": "1", "name": "write_file", "args": {"path": "n.txt", "content": "hi"}}]}, {"text": "ok", "tool_calls": []}]
    a, ev, d = make_agent(script()); a.send("make a file")
    for _ in range(100):
        ap = [e for e in ev if e.get("event") == "approval"]
        if ap: break
        time.sleep(0.05)
    assert ap and not os.path.exists(d + "/n.txt"); a.approve(ap[0]["id"], True); wait(a); assert open(d + "/n.txt").read() == "hi"
    a, ev, d = make_agent(script()); a.send("make a file")
    for _ in range(100):
        ap = [e for e in ev if e.get("event") == "approval"]
        if ap: break
        time.sleep(0.05)
    a.approve(ap[0]["id"], False); wait(a); assert not os.path.exists(d + "/n.txt") and "said no" in a.history[2]["content"]


def test_errors_and_not_ready_and_stop():
    d = tempfile.mkdtemp(); cfg = LLMConfig(d + "/llm.json")
    ev = []; a = Agent(cfg, ev.append)
    assert "API key" in a.send("hi")                              # anthropic with no key
    assert "first" in (Agent(LLMConfig(d + "/x.json"), ev.append).send("") or "first")
    def boom(*x): raise providers.LLMError("The API key was not accepted.")
    cfg.update({"provider": "ollama"}); a = Agent(cfg, ev.append, chat_fn=boom); a.send("hi"); wait(a)
    assert ev[-2]["event"] == "error" and a.history == []
    cfg2 = LLMConfig(d + "/y.json"); cfg2.update({"provider": "ollama", "api_key": "secret-key-1234"})
    pub = cfg2.public(); assert "secret" not in json.dumps(pub) and pub["has_key"] and pub["key_hint"] == "…1234"
    cfg2.update({"model": "z"}); assert cfg2.data["api_key"] == "secret-key-1234"      # empty key field keeps the saved key


# ---------------------------------------------------------------- streaming, commands, saved chats
class SSE(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body.get("stream") or "streamGenerateContent" in self.path
        if self.path.endswith("/chat/completions"):
            ev = [{"choices": [{"delta": {"content": "Hel"}}]}, {"choices": [{"delta": {"content": "lo"}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c9", "function": {"name": "read_file", "arguments": "{\"pa"}}]}}]},
                  {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "th\":\"a.txt\"}"}}]}}]}]
            tail = "data: [DONE]\n\n"
        elif self.path.endswith("/v1/messages"):
            ev = [{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
                  {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}},
                  {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t9", "name": "read_file"}},
                  {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"path\":"}},
                  {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "\"a.txt\"}"}}]
            tail = ""
        else:
            ev = [{"candidates": [{"content": {"parts": [{"text": "Hel"}]}}]}, {"candidates": [{"content": {"parts": [{"text": "lo"}, {"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}}]}}]}]
            tail = ""
        data = ("".join("data: " + json.dumps(e) + "\n\n" for e in ev) + tail).encode()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)


@pytest.fixture(scope="module")
def sse():
    s = HTTPServer(("127.0.0.1", 0), SSE); threading.Thread(target=s.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{s.server_port}"; s.shutdown()


@pytest.mark.parametrize("kind", ["openai", "anthropic", "gemini"])
def test_streaming_all_formats(sse, kind):
    got = []
    r = providers.chat({"kind": kind, "base_url": sse, "model": "m", "api_key": "k"}, "s", [{"role": "user", "content": "x"}], TOOLS, got.append)
    assert got == ["Hel", "lo"] and r["text"] == "Hello" and r["tool_calls"][0]["args"] == {"path": "a.txt"}


def test_run_command_gated(tmp_path):
    off = Tools(str(tmp_path), True, False)
    assert "run_command" not in [t["name"] for t in off.specs()]
    with pytest.raises(ToolError): off.run("run_command", {"command": "echo hi"})
    on = Tools(str(tmp_path), True, True)
    assert "run_command" in [t["name"] for t in on.specs()] and on.needs_approval("run_command")
    out = on.run("run_command", {"command": "echo hi && cd"})
    assert "hi" in out and "Exit code" in out
    (tmp_path / "x.txt").write_text("here")
    assert str(tmp_path) in on.run("run_command", {"command": "pwd" if os.name != "nt" else "cd"})


def test_agent_streams_asks_for_commands_and_saves(tmp_path):
    from core.agent.store import ChatStore
    cfg = LLMConfig(str(tmp_path / "llm.json")); cfg.update({"provider": "custom", "base_url": "http://x", "model": "m", "api_key": "k", "workspace": str(tmp_path),
                                                          "commands": True, "auto_approve": True})
    ev, step = [], []
    def fake(r, system, msgs, tools, on_text=None):
        step.append(1)
        if len(step) == 1:
            on_text("run"); return {"text": "run", "tool_calls": [{"id": "1", "name": "run_command", "args": {"command": "echo ok"}}]}
        on_text("done"); return {"text": "done", "tool_calls": []}
    a = Agent(cfg, ev.append, fake, ChatStore(str(tmp_path / "chats")))
    assert a.send("go") is None
    for _ in range(100):                                    # auto_approve is on, but a command must still ask
        ap = [e for e in ev if e.get("event") == "approval"]
        if ap: break
        time.sleep(0.05)
    assert ap and "echo ok" in ap[0]["text"]
    a.approve(ap[0]["id"], True); a.thread.join(5)
    assert [e["event"] for e in ev if e["event"] in ("delta", "done")] == ["delta", "delta", "done"]
    assert any(e["event"] == "text" and e["streamed"] for e in ev)
    lst = a.chats(); assert len(lst) == 1 and lst[0]["title"] == "go"
    cid = a.cid; a.reset(); assert a.history == [] and a.cid != cid
    shown = a.open_chat(cid); assert shown == [{"role": "user", "text": "go"}, {"role": "assistant", "text": "run"}, {"role": "assistant", "text": "done"}]
    a.delete_chat(cid); assert a.chats() == []
    assert a.store.load("../../etc/passwd") is None
