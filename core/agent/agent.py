"""The agent loop: ask the model, run the tools it asks for (asking the user first for changes), repeat."""
from __future__ import annotations

import threading
from typing import Callable, Optional

from core.agent import providers
from core.agent.config import LLMConfig
from core.agent.store import ChatStore
from core.agent.tools import ToolError, Tools

SYSTEM = (
    "You are the assistant inside Blue Voice, an accessibility app. The person talking to you usually writes by "
    "signing, so messages can be short, spelled letter by letter, or have small mistakes. Work out what they mean, "
    "answer plainly and briefly, and ask one short question if you truly cannot tell. "
    "You can use tools to work with files in the person's workspace folder and to search the web. Use them when they help, "
    "never pretend to have done something you did not do, and say which web pages you used. "
    "Commands, when allowed, run in the workspace folder and the person approves each one."
)
MAX_STEPS = 12


class Agent:
    def __init__(self, cfg: LLMConfig, emit: Callable[[dict], None], chat_fn=None, store: Optional[ChatStore] = None):
        self.cfg, self.emit, self.store = cfg, emit, store
        self.cid = store.new_id() if store else ""
        self.chat_fn = chat_fn or providers.chat
        self.history: list[dict] = []
        self.thread: Optional[threading.Thread] = None
        self.stop_evt = threading.Event()
        self.pending: dict[str, tuple[threading.Event, list]] = {}
        self._n = 0
        self._mark = 0

    # ------------------------------------------------------------ commands
    def reset(self):
        self.stop()
        self.history = []
        self.cid = self.store.new_id() if self.store else ""

    def chats(self) -> list:
        return self.store.list() if self.store else []

    def open_chat(self, cid: str) -> Optional[list]:
        """Load a saved chat; returns what to show (user/assistant text only), or None."""
        d = self.store.load(cid) if self.store else None
        if not d:
            return None
        self.stop()
        self.history, self.cid = d["history"], cid
        return self.shown()

    def shown(self) -> list:
        return [{"role": m["role"], "text": m["content"]} for m in self.history if m["role"] in ("user", "assistant") and m.get("content")]

    def delete_chat(self, cid: str):
        if self.store:
            self.store.delete(cid)
        if cid == self.cid:
            self.reset()

    def _save(self):
        if self.store:
            try:
                self.store.save(self.cid, self.history)
            except Exception:
                pass

    def stop(self):
        self.stop_evt.set()
        for ev, res in list(self.pending.values()):
            res.append(False); ev.set()
        t = self.thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=3)

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def approve(self, aid: str, ok: bool):
        p = self.pending.get(aid)
        if p:
            p[1].append(bool(ok)); p[0].set()

    def send(self, text: str) -> Optional[str]:
        text = (text or "").strip()
        if not text:
            return "Write a message first."
        if self.busy():
            return "Wait for the answer first."
        problem = self.cfg.ready()
        if problem:
            return problem
        self.stop_evt = threading.Event()
        self._mark = len(self.history)
        self.history.append({"role": "user", "content": text})
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return None

    # ------------------------------------------------------------ loop
    def _ask(self, tool: str, a: dict, tools: Tools) -> bool:
        if self.cfg.data.get("auto_approve") and tool != "run_command":      # commands always ask
            return True
        self._n += 1; aid = f"a{self._n}"
        ev, res = threading.Event(), []
        self.pending[aid] = (ev, res)
        self.emit({"type": "agent", "event": "approval", "id": aid, "tool": tool, "text": tools.describe(tool, a)})
        while not ev.wait(0.25):
            if self.stop_evt.is_set():
                break
        self.pending.pop(aid, None)
        return bool(res and res[0]) and not self.stop_evt.is_set()

    def _run(self):
        try:
            r = self.cfg.resolved()
            tools = Tools(r["workspace"], r["web"], r.get("commands", False), r.get("search_provider", ""), r.get("search_key", ""))
            for _ in range(MAX_STEPS):
                if self.stop_evt.is_set():
                    break
                self.emit({"type": "agent", "event": "thinking"})
                streamed = []
                def piece(t):
                    streamed.append(1)
                    if not self.stop_evt.is_set():
                        self.emit({"type": "agent", "event": "delta", "text": t})
                try:
                    out = self.chat_fn(r, SYSTEM, self.history, tools.specs(), piece)
                except TypeError:                               # a chat function without streaming support
                    out = self.chat_fn(r, SYSTEM, self.history, tools.specs())
                if self.stop_evt.is_set():
                    break
                self.history.append({"role": "assistant", "content": out["text"], "tool_calls": out["tool_calls"]})
                if out["text"]:
                    self.emit({"type": "agent", "event": "text", "text": out["text"], "streamed": bool(streamed)})
                if not out["tool_calls"]:
                    break
                for c in out["tool_calls"]:
                    name, args = c["name"], c["args"]
                    self.emit({"type": "agent", "event": "tool", "tool": name, "text": tools.describe(name, args) if tools.needs_approval(name) else f"{name} {args.get('path') or args.get('url') or args.get('query') or ''}".strip()})
                    try:
                        if tools.needs_approval(name) and not self._ask(name, args, tools):
                            res = "The person said no. Do not do this."
                        else:
                            res = tools.run(name, args)
                    except ToolError as e:
                        res = "Error: " + str(e)
                    except Exception as e:                      # a bad tool call must not end the chat
                        res = f"Error: {e}"
                    self.history.append({"role": "tool", "tool_call_id": c["id"], "name": name, "content": res})
            else:
                self.emit({"type": "agent", "event": "text", "text": "I stopped because this needed too many steps. Tell me what to do next."})
        except providers.LLMError as e:
            self.history = self.history[:self._mark]          # forget the failed turn so the next try starts clean
            self.emit({"type": "agent", "event": "error", "text": str(e)})
        except Exception as e:
            self.history = self.history[:self._mark]
            self.emit({"type": "agent", "event": "error", "text": f"Something went wrong: {e}"})
        finally:
            self._save()
            self.emit({"type": "agent", "event": "done"})
