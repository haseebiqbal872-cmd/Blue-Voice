"""Mechanical test of the record-your-own-word engine on the 6 real clips shipped with GestureSign.
Each clip is turned into 'takes' by re-timing and a little tracker noise (NOT independent real takes)."""
import json, os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from core.sign_words import WordRecognizer
from core.gs.teach import TeachStore

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "gs")
rec = WordRecognizer(R)
d = json.load(open(os.path.join(R, "replay_clips.json"))); pts = d["points"]


def clip(c):
    fr = np.full((len(c["frames"]), 543, 3), np.nan, np.float32)
    for t, row in enumerate(c["frames"]):
        fr[t, pts, :2] = np.array([np.nan if v is None else v for v in row], np.float32).reshape(-1, 2)
    return fr


def variant(fr, rng, fps_factor, noise):
    n = len(fr); idx = np.clip(np.round(np.arange(0, n - 1, fps_factor)).astype(int), 0, n - 1)
    v = fr[idx].copy(); v[..., :2] += rng.normal(0, noise, v[..., :2].shape).astype(np.float32)
    shift = rng.normal(0, 0.02, 2).astype(np.float32); v[..., :2] += shift
    return v


def test_engine():
    rng = np.random.default_rng(1)
    store = TeachStore(os.path.join(tempfile.mkdtemp(), "w.json"))
    clips = [(c["title"], clip(c)) for c in d["clips"]]
    for name, fr in clips[:5]:                       # teach 5 words, keep the 6th unknown
        for f in (0.9, 1.0, 1.1):
            store.add_take(name, rec.embed(variant(fr, rng, f, 0.002), 30))
    assert len(store.ready_words()) == 5
    hits, tot, sims = 0, 0, []
    for name, fr in clips[:5]:
        for f in (0.95, 1.05, 1.2):
            r = rec.read(variant(fr, rng, f, 0.004), 30, store); tot += 1
            hits += (r["source"] == "yours" and r["top"][0][0] == name)
    unk = []
    name, fr = clips[5]
    for f in (0.95, 1.05, 1.2):
        r = rec.read(variant(fr, rng, f, 0.004), 30, store); unk.append((r["source"], r["top"][0][0]))
    print(f"taught words recognised: {hits}/{tot}   unknown sign ('{name}') -> {unk}")
    store2 = TeachStore(store.path); assert store2.ready_words() == store.ready_words()   # persistence
    assert hits >= tot - 1


if __name__ == "__main__":
    test_engine()
