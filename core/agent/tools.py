"""Tools the agent may use. Files are confined to one folder the user chose; changes ask first; the web tool
refuses addresses on this computer or home network."""
from __future__ import annotations

import html
import json
import ipaddress
import os
import re
import socket
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

MAX_READ = 60_000
MAX_WEB = 40_000


class ToolError(Exception):
    pass


def _schema(props: dict, req: list) -> dict:
    return {"type": "object", "properties": props, "required": req}


S = {"type": "string"}
SPECS = {
    "list_files": ("List the files and folders in a folder of the workspace.", _schema({"path": {**S, "description": "Folder, relative to the workspace. Use . for the top."}}, ["path"]), False),
    "read_file": ("Read a text file from the workspace.", _schema({"path": S}, ["path"]), False),
    "write_file": ("Create or replace a text file in the workspace.", _schema({"path": S, "content": S}, ["path", "content"]), True),
    "edit_file": ("Replace one exact piece of text in a workspace file.", _schema({"path": S, "old": S, "new": S}, ["path", "old", "new"]), True),
    "delete_file": ("Delete one file from the workspace.", _schema({"path": S}, ["path"]), True),
    "run_command": ("Run one shell command inside the workspace folder (for example to run tests or a script). The person must approve each command. It stops after 60 seconds.", _schema({"command": S}, ["command"]), True),
    "web_fetch": ("Download a web page and return its text.", _schema({"url": S}, ["url"]), False),
    "web_search": ("Search the web and return the top results with their links.", _schema({"query": S}, ["query"]), False),
}


class Tools:
    def __init__(self, workspace: str, web: bool = True, commands: bool = False, search_provider: str = "", search_key: str = "", fetch=None):
        self.root = os.path.realpath(workspace) if workspace else ""
        self.web, self.commands = web, commands
        self.search_provider, self.search_key = (search_provider or "").lower(), search_key or ""
        self._fetch = fetch or self._api

    def specs(self) -> list[dict]:
        out = []
        for n, (d, p, _) in SPECS.items():
            if n.startswith("web") and not self.web:
                continue
            if not n.startswith("web") and not self.root:
                continue
            if n == "run_command" and not self.commands:
                continue
            out.append({"name": n, "description": d, "parameters": p})
        return out

    def needs_approval(self, name: str) -> bool:
        return SPECS.get(name, ("", {}, False))[2]

    def describe(self, name: str, a: dict) -> str:
        p = a.get("path", "")
        return {"write_file": f"Write the file {p} ({len(a.get('content', ''))} characters)", "edit_file": f"Change the file {p}",
                "delete_file": f"Delete the file {p}", "run_command": f"Run this command in your workspace folder: {a.get('command', '')}"}.get(name, name)

    # -------------------------------------------------------------- files
    def _safe(self, rel: str) -> str:
        if not self.root:
            raise ToolError("No workspace folder is set. The user must choose one in Settings.")
        rel = (rel or ".").replace("\\", "/").lstrip("/")
        p = os.path.realpath(os.path.join(self.root, rel))
        if p != self.root and not p.startswith(self.root + os.sep):
            raise ToolError("That path is outside the workspace folder.")
        return p

    def run(self, name: str, a: dict) -> str:
        if name not in SPECS:
            raise ToolError(f"There is no tool called {name}.")
        return getattr(self, "t_" + name)(**{k: a.get(k, "") for k in SPECS[name][1]["properties"]})

    def t_run_command(self, command):
        if not self.commands:
            raise ToolError("Running commands is turned off in Settings.")
        if not self.root:
            raise ToolError("No workspace folder is set.")
        command = (command or "").strip()
        if not command:
            raise ToolError("No command given.")
        try:
            r = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            return "The command ran for 60 seconds and was stopped."
        out = (r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")).strip()
        if len(out) > 8000:
            out = out[:4000] + "\n... (cut) ...\n" + out[-4000:]
        return f"Exit code {r.returncode}.\n{out}" if out else f"Exit code {r.returncode}. (no output)"

    def t_list_files(self, path):
        p = self._safe(path)
        if not os.path.isdir(p):
            raise ToolError("That folder does not exist.")
        rows = []
        for n in sorted(os.listdir(p))[:300]:
            f = os.path.join(p, n)
            rows.append(n + ("/" if os.path.isdir(f) else f"  ({os.path.getsize(f)} bytes)"))
        return "\n".join(rows) or "(empty folder)"

    def t_read_file(self, path):
        p = self._safe(path)
        if not os.path.isfile(p):
            raise ToolError("That file does not exist.")
        with open(p, "rb") as f:
            raw = f.read(MAX_READ + 1)
        if b"\0" in raw[:2000]:
            raise ToolError("That looks like a binary file, not text.")
        t = raw[:MAX_READ].decode("utf-8", "replace")
        return t + ("\n[file continues; only the first part is shown]" if len(raw) > MAX_READ else "")

    def t_write_file(self, path, content):
        p = self._safe(path)
        if os.path.isdir(p):
            raise ToolError("That is a folder.")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}."

    def t_edit_file(self, path, old, new):
        p = self._safe(path)
        if not os.path.isfile(p):
            raise ToolError("That file does not exist.")
        t = open(p, encoding="utf-8").read()
        if not old or t.count(old) != 1:
            raise ToolError(f"The text to replace must appear exactly once (it appears {t.count(old) if old else 0} times).")
        open(p, "w", encoding="utf-8", newline="").write(t.replace(old, new))
        return f"Changed {path}."

    def t_delete_file(self, path):
        p = self._safe(path)
        if not os.path.isfile(p):
            raise ToolError("That file does not exist (only single files can be deleted).")
        os.remove(p)
        return f"Deleted {path}."

    # -------------------------------------------------------------- web
    @staticmethod
    def _public(url: str) -> str:
        u = urllib.parse.urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise ToolError("Only http and https addresses are allowed.")
        try:
            infos = socket.getaddrinfo(u.hostname, None)
        except OSError:
            raise ToolError("That address could not be found.")
        for i in infos:
            ip = ipaddress.ip_address(i[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
                raise ToolError("That address is on this computer or a private network, so it is blocked.")
        return url

    @staticmethod
    def _text(raw: str) -> str:
        raw = re.sub(r"(?is)<(script|style|noscript|svg).*?</\1>", " ", raw)
        raw = re.sub(r"(?s)<[^>]+>", " ", raw)
        return re.sub(r"\s+", " ", html.unescape(raw)).strip()

    def _get(self, url: str) -> str:
        class NoPrivateRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(s, req, fp, code, msg, headers, newurl):
                Tools._public(newurl)
                return super().redirect_request(req, fp, code, msg, headers, newurl)
        self._public(url)
        op = urllib.request.build_opener(NoPrivateRedirect)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BlueVoice/1.0", "Accept": "text/html,text/plain,*/*"})
        try:
            with op.open(req, timeout=20) as r:
                ct = r.headers.get("Content-Type", "")
                if not re.search(r"text|json|xml|html", ct, re.I):
                    raise ToolError(f"That page is not text ({ct}).")
                return r.read(1_500_000).decode("utf-8", "replace")
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Could not open that page: {e}")

    def t_web_fetch(self, url):
        t = self._text(self._get(url))
        return t[:MAX_WEB] + (" [page continues]" if len(t) > MAX_WEB else "")

    @staticmethod
    def _api(url: str, headers: dict, body: dict | None = None):
        req = urllib.request.Request(url, headers={"User-Agent": "BlueVoice/1.0", "Accept": "application/json", **headers},
                                     data=json.dumps(body).encode() if body is not None else None, method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise ToolError({401: "The search key was not accepted.", 403: "The search key was not accepted.", 429: "The search service says there were too many searches or the free quota is used up."}.get(e.code, f"The search service answered with an error ({e.code})."))
        except Exception as e:
            raise ToolError(f"Could not reach the search service: {e}")

    def _keyed_search(self, query: str) -> str:
        if self.search_provider == "brave":
            j = self._fetch("https://api.search.brave.com/res/v1/web/search?count=6&q=" + urllib.parse.quote(query), {"X-Subscription-Token": self.search_key})
            rows = [(r.get("title", ""), r.get("url", ""), r.get("description", "")) for r in (j.get("web") or {}).get("results", [])]
            head = ""
        else:
            j = self._fetch("https://api.tavily.com/search", {"Authorization": "Bearer " + self.search_key, "Content-Type": "application/json"},
                            {"query": query, "max_results": 6, "include_answer": True})
            rows = [(r.get("title", ""), r.get("url", ""), r.get("content", "")) for r in j.get("results", [])]
            head = f"Summary: {j['answer']}\n" if j.get("answer") else ""
        lines = [f"{i}. {self._text(t)}\n   {u}\n   {self._text(d)[:400]}" for i, (t, u, d) in enumerate(rows[:6], 1)]
        return head + "\n".join(lines) if lines else "No results found."

    def t_web_search(self, query):
        if self.search_key and self.search_provider in ("brave", "tavily"):
            try:
                return self._keyed_search(query)
            except ToolError as e:
                note = f"(Your {self.search_provider.title()} search failed: {e} Using the free search instead.)\n"
                return note + self._ddg(query)
        return self._ddg(query)

    def _ddg(self, query):
        page = self._get("https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(query))
        out = []
        for m in re.finditer(r'(?s)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?(?:class="result__snippet"[^>]*>(.*?)</a>)?', page):
            href = m.group(1)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            if "uddg" in q:
                href = q["uddg"][0]
            out.append(f"{len(out) + 1}. {self._text(m.group(2))}\n   {href}\n   {self._text(m.group(3) or '')}")
            if len(out) >= 6:
                break
        return "\n".join(out) or "No results found."
