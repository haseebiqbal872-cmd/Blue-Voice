"""LLM settings, stored on this computer only (data/llm.json). The API key is never sent back to the page."""
from __future__ import annotations

import json
import os

PRESETS = {
    "anthropic": {"label": "Claude (Anthropic)", "kind": "anthropic", "base_url": "https://api.anthropic.com", "model": "claude-sonnet-4-5"},
    "openai": {"label": "ChatGPT (OpenAI)", "kind": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "gemini": {"label": "Gemini (Google)", "kind": "gemini", "base_url": "https://generativelanguage.googleapis.com", "model": "gemini-2.0-flash"},
    "ollama": {"label": "Ollama (on this computer)", "kind": "openai", "base_url": "http://localhost:11434/v1", "model": "llama3.1", "no_key": True},
    "openrouter": {"label": "OpenRouter", "kind": "openai", "base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini"},
    "groq": {"label": "Groq", "kind": "openai", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "custom": {"label": "Other (OpenAI-compatible)", "kind": "openai", "base_url": "", "model": ""},
}

DEFAULT = {"provider": "anthropic", "model": "", "base_url": "", "api_key": "", "workspace": "", "auto_approve": False, "web": True, "commands": False, "search_provider": "duckduckgo", "search_key": ""}


class LLMConfig:
    def __init__(self, path: str):
        self.path, self.data = path, dict(DEFAULT)
        if os.path.isfile(path):
            try:
                self.data.update(json.load(open(path, encoding="utf-8")))
            except Exception:
                pass

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        json.dump(self.data, open(tmp, "w", encoding="utf-8"), indent=1)
        os.replace(tmp, self.path)

    def update(self, new: dict):
        for k in ("provider", "model", "base_url", "workspace"):
            if k in new:
                self.data[k] = str(new[k]).strip()
        if new.get("api_key"):                       # empty means "keep the saved key"
            self.data["api_key"] = str(new["api_key"]).strip()
        if new.get("clear_key"):
            self.data["api_key"] = ""
        if new.get("search_provider") in ("duckduckgo", "brave", "tavily"):
            self.data["search_provider"] = new["search_provider"]
        if new.get("search_key"):
            self.data["search_key"] = str(new["search_key"]).strip()
        if new.get("clear_search_key"):
            self.data["search_key"] = ""
        for k in ("auto_approve", "web", "commands"):
            if k in new:
                self.data[k] = bool(new[k])
        self.save()

    def resolved(self) -> dict:
        d = dict(self.data); p = PRESETS.get(d["provider"], PRESETS["custom"])
        return {**d, "kind": p["kind"], "model": d["model"] or p["model"], "base_url": (d["base_url"] or p["base_url"]).rstrip("/"),
                "no_key": bool(p.get("no_key"))}

    def ready(self) -> str | None:
        r = self.resolved()
        if not r["base_url"] or not r["model"]:
            return "Choose a model first (Settings, AI models)."
        if not r["api_key"] and not r["no_key"]:
            return "Add your API key first (Settings, AI models)."
        return None

    def public(self) -> dict:
        d = self.data; r = self.resolved()
        return {"provider": d["provider"], "model": d["model"], "base_url": d["base_url"], "workspace": d["workspace"],
                "auto_approve": d["auto_approve"], "web": d["web"], "commands": d["commands"], "search_provider": d["search_provider"], "has_search_key": bool(d["search_key"]),
                "search_hint": ("…" + d["search_key"][-4:]) if len(d["search_key"]) > 8 else "", "has_key": bool(d["api_key"]), "key_hint": ("…" + d["api_key"][-4:]) if len(d["api_key"]) > 8 else "",
                "ready": self.ready() is None, "problem": self.ready() or "", "default_model": PRESETS.get(d["provider"], PRESETS["custom"])["model"],
                "presets": {k: {"label": v["label"], "model": v["model"], "base_url": v["base_url"], "no_key": bool(v.get("no_key"))} for k, v in PRESETS.items()}}
