"""Sign in with GitHub (device flow) or Google (browser + PKCE, back to this computer). Nothing is sent anywhere except to
GitHub / Google themselves. We only ask for your name and email, and keep no token after sign-in."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from core import oauth_ids

UA = "BlueVoice"


def _http(method: str, url: str, headers: dict | None = None, form: dict | None = None):
    data = urllib.parse.urlencode(form).encode() if form is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": UA, "Accept": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")
        except Exception:
            return e.code, {}


class SignInError(Exception):
    pass


class OAuth:
    def __init__(self, data_dir: str, http=None):
        self.dir, self.http = data_dir, http or _http
        self.pending: dict[str, dict] = {}              # google: state -> {verifier, redirect, t}
        self.gh_cancel = threading.Event()

    # ---------------------------------------------------------------- setup
    def config(self) -> dict:
        c = {"github_client_id": oauth_ids.GITHUB_CLIENT_ID, "google_client_id": oauth_ids.GOOGLE_CLIENT_ID, "google_client_secret": oauth_ids.GOOGLE_CLIENT_SECRET}
        try:
            c.update({k: v for k, v in json.load(open(os.path.join(self.dir, "oauth.json"), encoding="utf-8")).items() if v})
        except Exception:
            pass
        for k, env in (("github_client_id", "BV_GITHUB_CLIENT_ID"), ("google_client_id", "BV_GOOGLE_CLIENT_ID"), ("google_client_secret", "BV_GOOGLE_CLIENT_SECRET")):
            if os.environ.get(env):
                c[k] = os.environ[env]
        return c

    def available(self) -> dict:
        c = self.config()
        return {"github": bool(c["github_client_id"]), "google": bool(c["google_client_id"])}

    # ---------------------------------------------------------------- GitHub
    def github_start(self) -> dict:
        cid = self.config()["github_client_id"]
        if not cid:
            raise SignInError("GitHub sign-in is not set up in this copy of Blue Voice yet.")
        code, j = self.http("POST", "https://github.com/login/device/code", None, {"client_id": cid, "scope": "read:user user:email"})
        if code != 200 or "device_code" not in j:
            raise SignInError("GitHub did not start the sign-in. " + str(j.get("error_description") or j.get("error") or ""))
        return {"device_code": j["device_code"], "user_code": j["user_code"], "uri": j.get("verification_uri", "https://github.com/login/device"),
                "interval": max(int(j.get("interval", 5)), 1), "expires": int(j.get("expires_in", 900))}

    def github_wait(self, st: dict, sleep=time.sleep) -> dict:
        """Polls until the person approves in the browser. Returns {"name","email"}. Raises SignInError."""
        cid, end, wait = self.config()["github_client_id"], time.time() + st["expires"], st["interval"]
        while time.time() < end and not self.gh_cancel.is_set():
            sleep(wait)
            code, j = self.http("POST", "https://github.com/login/oauth/access_token", None,
                                {"client_id": cid, "device_code": st["device_code"], "grant_type": "urn:ietf:params:oauth:grant-type:device_code"})
            err = j.get("error")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                wait += 5; continue
            if err == "access_denied":
                raise SignInError("You said no on GitHub, so nothing was signed in.")
            if err in ("expired_token", "incorrect_device_code"):
                raise SignInError("The code ran out of time. Try again.")
            if err or "access_token" not in j:
                raise SignInError("GitHub could not finish the sign-in.")
            return self._github_user(j["access_token"])
        raise SignInError("Sign-in was cancelled." if self.gh_cancel.is_set() else "The code ran out of time. Try again.")

    def _github_user(self, token: str) -> dict:
        h = {"Authorization": "Bearer " + token, "X-GitHub-Api-Version": "2022-11-28"}
        c, u = self.http("GET", "https://api.github.com/user", h)
        if c != 200:
            raise SignInError("Could not read your GitHub name.")
        email = u.get("email") or ""
        if not email:
            c, es = self.http("GET", "https://api.github.com/user/emails", h)
            if c == 200 and isinstance(es, list):
                email = next((e["email"] for e in es if e.get("primary") and e.get("verified")), "")
        return {"name": u.get("name") or u.get("login") or "GitHub user", "email": email or f"{u.get('login', 'user')}@users.noreply.github.com", "provider": "github"}

    # ---------------------------------------------------------------- Google
    def google_url(self, redirect: str) -> str:
        c = self.config()
        if not c["google_client_id"]:
            raise SignInError("Google sign-in is not set up in this copy of Blue Voice yet.")
        now = time.time()
        for k in [k for k, v in self.pending.items() if now - v["t"] > 600]:
            self.pending.pop(k, None)
        state, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(48)
        self.pending[state] = {"verifier": verifier, "redirect": redirect, "t": now}
        chal = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        q = {"client_id": c["google_client_id"], "redirect_uri": redirect, "response_type": "code", "scope": "openid email profile",
             "state": state, "code_challenge": chal, "code_challenge_method": "S256", "prompt": "select_account"}
        return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(q)

    def google_finish(self, state: str, code: str) -> dict:
        p = self.pending.pop(state or "", None)
        if not p or time.time() - p["t"] > 600:
            raise SignInError("This sign-in link is not valid any more. Start again in Blue Voice.")
        c = self.config()
        form = {"client_id": c["google_client_id"], "code": code, "code_verifier": p["verifier"], "grant_type": "authorization_code", "redirect_uri": p["redirect"]}
        if c["google_client_secret"]:
            form["client_secret"] = c["google_client_secret"]
        st, j = self.http("POST", "https://oauth2.googleapis.com/token", None, form)
        if st != 200 or "access_token" not in j:
            raise SignInError("Google did not accept the sign-in.")
        st, u = self.http("GET", "https://openidconnect.googleapis.com/v1/userinfo", {"Authorization": "Bearer " + j["access_token"]})
        if st != 200 or not u.get("email"):
            raise SignInError("Could not read your Google name and email.")
        return {"name": u.get("name") or u["email"].split("@")[0], "email": u["email"], "provider": "google"}
