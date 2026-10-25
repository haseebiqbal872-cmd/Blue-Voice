"""Accounts, sign packs and the update check."""
import json, os, sys, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pytest
from core.gs.teach import TeachStore
from core import packs


class FakeHTTP:
    def __init__(self, script): self.script, self.calls = list(script), []
    def __call__(self, method, url, headers=None, form=None):
        self.calls.append((method, url, form)); return self.script.pop(0)


def test_github_device_flow(tmp_path, monkeypatch):
    from core.oauth import OAuth, SignInError
    monkeypatch.setenv("BV_GITHUB_CLIENT_ID", "cid")
    h = FakeHTTP([(200, {"device_code": "dc", "user_code": "ABCD-1234", "verification_uri": "https://github.com/login/device", "interval": 1, "expires_in": 60}),
                  (200, {"error": "authorization_pending"}), (200, {"error": "slow_down"}), (200, {"access_token": "tok"}),
                  (200, {"login": "haseeb", "name": None, "email": None}), (200, [{"email": "a@b.co", "primary": True, "verified": True}])])
    o = OAuth(str(tmp_path), h)
    st = o.github_start(); assert st["user_code"] == "ABCD-1234" and h.calls[0][2]["client_id"] == "cid"
    sleeps = []
    assert o.github_wait(st, sleeps.append) == {"name": "haseeb", "email": "a@b.co", "provider": "github"}
    assert sleeps == [1, 1, 6]                                    # slow_down adds 5 seconds
    o2 = OAuth(str(tmp_path), FakeHTTP([(200, {"error": "access_denied"})]))
    with pytest.raises(SignInError, match="said no"): o2.github_wait({"device_code": "x", "interval": 1, "expires": 60}, lambda s: None)
    monkeypatch.delenv("BV_GITHUB_CLIENT_ID")
    with pytest.raises(SignInError, match="not set up"): OAuth(str(tmp_path), h).github_start()


def test_google_pkce_flow(tmp_path, monkeypatch):
    import urllib.parse, hashlib, base64
    from core.oauth import OAuth, SignInError
    monkeypatch.setenv("BV_GOOGLE_CLIENT_ID", "gid"); monkeypatch.setenv("BV_GOOGLE_CLIENT_SECRET", "gs")
    h = FakeHTTP([(200, {"access_token": "t"}), (200, {"name": "Haseeb", "email": "h@gmail.com"})])
    o = OAuth(str(tmp_path), h)
    url = o.google_url("http://127.0.0.1:8765/oauth/google"); q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
    assert q["code_challenge_method"] == "S256" and q["client_id"] == "gid" and q["redirect_uri"].endswith("/oauth/google")
    ver = o.pending[q["state"]]["verifier"]
    assert q["code_challenge"] == base64.urlsafe_b64encode(hashlib.sha256(ver.encode()).digest()).rstrip(b"=").decode()
    with pytest.raises(SignInError): o.google_finish("wrong", "c")
    assert o.google_finish(q["state"], "code") == {"name": "Haseeb", "email": "h@gmail.com", "provider": "google"}
    assert h.calls[0][2]["code_verifier"] == ver and h.calls[0][2]["client_secret"] == "gs"
    with pytest.raises(SignInError): o.google_finish(q["state"], "code")      # one use only


def vec(i, d=4):
    v = [0.0] * d; v[i % d] = 1.0; return v


def test_pack_roundtrip_and_validation(tmp_path):
    a = TeachStore(str(tmp_path / "a.json")); a.words = {"hello": [vec(0), vec(1), vec(2)]}
    pk = json.loads(json.dumps(packs.export_pack(a, "m1", 4)))
    b = TeachStore(str(tmp_path / "b.json"))
    assert packs.import_pack(b, pk, "m1", 4) == (1, 3) and len(b.words["hello"]) == 3
    assert TeachStore(str(tmp_path / "b.json")).words["hello"][0] == [1.0, 0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="different recognition model"): packs.import_pack(b, pk, "m2", 4)
    with pytest.raises(ValueError, match="not a Blue Voice"): packs.import_pack(b, {"x": 1}, "m1", 4)
    bad = json.loads(json.dumps(pk)); bad["words"]["hello"][0] = [0, 0, 0, 0]
    with pytest.raises(ValueError, match="damaged"): packs.import_pack(b, bad, "m1", 4)
    bad["words"]["hello"][0] = [1, 2]
    with pytest.raises(ValueError, match="damaged"): packs.import_pack(b, bad, "m1", 4)
    bad["words"]["hello"][0] = [float("nan"), 0, 0, 0]
    with pytest.raises(ValueError, match="damaged"): packs.import_pack(b, bad, "m1", 4)
    for _ in range(3): packs.import_pack(b, pk, "m1", 4)
    assert len(b.words["hello"]) == 6                      # capped at MAX_TAKES


def test_update_check_without_repo():
    from server import Controller
    r = Controller().check_update()
    assert r["available"] is False and "not switched on" in r["error"]
