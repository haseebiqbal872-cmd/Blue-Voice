"""Saved chats: one small JSON file per conversation in data/chats/. Stays on this computer."""
from __future__ import annotations

import json
import os
import re
import time
import uuid


class ChatStore:
    def __init__(self, folder: str):
        self.dir = folder

    def _p(self, cid: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{12}", cid or ""):
            raise ValueError("bad chat id")
        return os.path.join(self.dir, cid + ".json")

    def new_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def save(self, cid: str, history: list) -> None:
        if not history:
            return
        first = next((m["content"] for m in history if m["role"] == "user"), "Chat")
        old = self.load(cid) or {}
        os.makedirs(self.dir, exist_ok=True)
        doc = {"id": cid, "title": (first[:60] or "Chat"), "created": old.get("created", time.time()), "updated": time.time(), "history": history}
        tmp = self._p(cid) + ".tmp"
        json.dump(doc, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, self._p(cid))

    def load(self, cid: str):
        try:
            return json.load(open(self._p(cid), encoding="utf-8"))
        except Exception:
            return None

    def list(self) -> list:
        out = []
        if os.path.isdir(self.dir):
            for n in os.listdir(self.dir):
                if n.endswith(".json"):
                    d = self.load(n[:-5])
                    if d:
                        out.append({"id": d["id"], "title": d.get("title", "Chat"), "updated": d.get("updated", 0)})
        return sorted(out, key=lambda x: -x["updated"])[:100]

    def delete(self, cid: str) -> None:
        try:
            os.remove(self._p(cid))
        except OSError:
            pass
