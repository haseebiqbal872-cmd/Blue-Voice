"""One chat call, three wire formats: OpenAI-compatible, Anthropic, Gemini.

Internal message format (OpenAI-like):
  {"role":"user","content":str} | {"role":"assistant","content":str,"tool_calls":[{"id","name","args"}]}
  | {"role":"tool","tool_call_id":str,"name":str,"content":str}
Tools: [{"name","description","parameters":<json schema>}]
`chat()` returns {"text": str, "tool_calls": [{"id","name","args"}]}.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class LLMError(Exception):
    pass


def _open(url: str, headers: dict, body: dict, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json", "User-Agent": "BlueVoice/1.0", **headers}), timeout=timeout)


def _stream(url: str, headers: dict, body: dict, timeout: int = 120):
    """Server-sent events: yields each `data:` line as parsed JSON."""
    try:
        with _open(url, headers, body, timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    return
                try:
                    yield json.loads(d)
                except ValueError:
                    continue
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        _raise(e, url)


def _post(url: str, headers: dict, body: dict, timeout: int = 120) -> dict:
    try:
        with _open(url, headers, body, timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        _raise(e, url)


def _raise(e, url: str):
    if isinstance(e, urllib.error.HTTPError):
        detail = ""
        try:
            j = json.loads(e.read().decode("utf-8", "replace"))
            detail = (j.get("error", {}) or {}).get("message") if isinstance(j.get("error"), dict) else str(j.get("error") or j.get("message") or "")
        except Exception:
            pass
        hint = {401: "The API key was not accepted.", 403: "This key is not allowed to use that model.", 404: "That model or address was not found.",
                429: "Too many requests or no credit left on this key."}.get(e.code, "")
        raise LLMError(f"{hint} ({e.code}) {detail or ''}".strip())
    elif isinstance(e, urllib.error.URLError):
        raise LLMError(f"Could not reach {url.split('/')[2]}. Check your internet connection or the address. ({e.reason})")
    raise LLMError("The model took too long to answer.")


# ---------------------------------------------------------------- OpenAI-compatible
def _openai(cfg, system, messages, tools, on_text=None):
    msgs = [{"role": "system", "content": system}]
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            msgs.append({"role": "assistant", "content": m.get("content") or None,
                         "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": json.dumps(c["args"])}} for c in m["tool_calls"]]})
        elif m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    body = {"model": cfg["model"], "messages": msgs}
    if tools:
        body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in tools]
    h = {"Authorization": "Bearer " + cfg["api_key"]} if cfg["api_key"] else {}
    if on_text:
        body["stream"] = True
        text, acc = "", {}
        for j in _stream(cfg["base_url"] + "/chat/completions", h, body):
            try:
                d = j["choices"][0].get("delta") or {}
            except Exception:
                continue
            if d.get("content"):
                text += d["content"]; on_text(d["content"])
            for c in d.get("tool_calls") or []:
                a = acc.setdefault(c.get("index", 0), {"id": "", "name": "", "args": ""})
                a["id"] = c.get("id") or a["id"]
                f = c.get("function") or {}
                a["name"] += f.get("name") or ""; a["args"] += f.get("arguments") or ""
        calls = []
        for i in sorted(acc):
            try:
                args = json.loads(acc[i]["args"] or "{}")
            except Exception:
                args = {}
            calls.append({"id": acc[i]["id"] or f"call{i}", "name": acc[i]["name"], "args": args if isinstance(args, dict) else {}})
        return {"text": text, "tool_calls": calls}
    j = _post(cfg["base_url"] + "/chat/completions", h, body)
    try:
        m = j["choices"][0]["message"]
    except Exception:
        raise LLMError("The model sent an answer in a format I do not understand.")
    calls = []
    for c in m.get("tool_calls") or []:
        try:
            args = json.loads(c["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        calls.append({"id": c.get("id") or f"call{len(calls)}", "name": c["function"]["name"], "args": args if isinstance(args, dict) else {}})
    return {"text": m.get("content") or "", "tool_calls": calls}


# ---------------------------------------------------------------- Anthropic
def _anthropic(cfg, system, messages, tools, on_text=None):
    msgs = []
    for m in messages:
        if m["role"] == "assistant":
            blocks = ([{"type": "text", "text": m["content"]}] if m.get("content") else []) + \
                     [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]} for c in m.get("tool_calls") or []]
            msgs.append({"role": "assistant", "content": blocks or [{"type": "text", "text": "(no text)"}]})
        elif m["role"] == "tool":
            blk = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
            if msgs and msgs[-1]["role"] == "user" and isinstance(msgs[-1]["content"], list):
                msgs[-1]["content"].append(blk)                  # results of one turn go in one user message
            else:
                msgs.append({"role": "user", "content": [blk]})
        else:
            msgs.append({"role": "user", "content": m["content"]})
    body = {"model": cfg["model"], "max_tokens": 4096, "system": system, "messages": msgs}
    if tools:
        body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
    hd = {"x-api-key": cfg["api_key"], "anthropic-version": "2023-06-01"}
    if on_text:
        body["stream"] = True
        text, blocks = "", {}
        for j in _stream(cfg["base_url"] + "/v1/messages", hd, body):
            t, i = j.get("type"), j.get("index", 0)
            if t == "content_block_start" and (j.get("content_block") or {}).get("type") == "tool_use":
                cb = j["content_block"]; blocks[i] = {"id": cb["id"], "name": cb["name"], "args": ""}
            elif t == "content_block_delta":
                d = j.get("delta") or {}
                if d.get("type") == "text_delta":
                    text += d.get("text", ""); on_text(d.get("text", ""))
                elif d.get("type") == "input_json_delta" and i in blocks:
                    blocks[i]["args"] += d.get("partial_json", "")
            elif t == "error":
                raise LLMError((j.get("error") or {}).get("message", "The model stopped with an error."))
        calls = []
        for i in sorted(blocks):
            try:
                args = json.loads(blocks[i]["args"] or "{}")
            except Exception:
                args = {}
            calls.append({"id": blocks[i]["id"], "name": blocks[i]["name"], "args": args if isinstance(args, dict) else {}})
        return {"text": text, "tool_calls": calls}
    j = _post(cfg["base_url"] + "/v1/messages", hd, body)
    text, calls = "", []
    for b in j.get("content") or []:
        if b.get("type") == "text":
            text += b.get("text", "")
        elif b.get("type") == "tool_use":
            calls.append({"id": b["id"], "name": b["name"], "args": b.get("input") or {}})
    return {"text": text, "tool_calls": calls}


# ---------------------------------------------------------------- Gemini
def _gemini(cfg, system, messages, tools, on_text=None):
    contents = []
    for m in messages:
        if m["role"] == "assistant":
            parts = ([{"text": m["content"]}] if m.get("content") else []) + [{"functionCall": {"name": c["name"], "args": c["args"]}} for c in m.get("tool_calls") or []]
            contents.append({"role": "model", "parts": parts or [{"text": "(no text)"}]})
        elif m["role"] == "tool":
            part = {"functionResponse": {"name": m["name"], "response": {"result": m["content"]}}}
            if contents and contents[-1]["role"] == "user" and contents[-1]["parts"] and "functionResponse" in contents[-1]["parts"][0]:
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
        else:
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents}
    if tools:
        body["tools"] = [{"functionDeclarations": [{"name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in tools]}]
    gh = {"x-goog-api-key": cfg["api_key"]}
    if on_text:
        text, calls = "", []
        for j in _stream(f"{cfg['base_url']}/v1beta/models/{cfg['model']}:streamGenerateContent?alt=sse", gh, body):
            try:
                parts = j["candidates"][0]["content"]["parts"]
            except Exception:
                continue
            for p in parts:
                if "text" in p:
                    text += p["text"]; on_text(p["text"])
                elif "functionCall" in p:
                    fc = p["functionCall"]; calls.append({"id": f"g{len(calls)}", "name": fc["name"], "args": fc.get("args") or {}})
        if not text and not calls:
            raise LLMError("The model sent no answer. It may have refused the request.")
        return {"text": text, "tool_calls": calls}
    j = _post(f"{cfg['base_url']}/v1beta/models/{cfg['model']}:generateContent", gh, body)
    text, calls = "", []
    try:
        for p in j["candidates"][0]["content"]["parts"]:
            if "text" in p:
                text += p["text"]
            elif "functionCall" in p:
                fc = p["functionCall"]; calls.append({"id": f"g{len(calls)}", "name": fc["name"], "args": fc.get("args") or {}})
    except Exception:
        raise LLMError("The model sent no answer. It may have refused the request.")
    return {"text": text, "tool_calls": calls}


def chat(cfg: dict, system: str, messages: list, tools: list, on_text=None) -> dict:
    """With `on_text`, the answer is streamed: on_text(piece) is called as text arrives; the full result is still returned."""
    fn = {"openai": _openai, "anthropic": _anthropic, "gemini": _gemini}[cfg["kind"]]
    return fn(cfg, system, messages, tools, on_text)
