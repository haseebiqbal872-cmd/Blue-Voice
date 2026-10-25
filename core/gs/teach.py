"""Record-your-own-words: nearest-prototype matching on the model's 128-number sign fingerprints.

Port of GestureSign's "Teach your own" rule (MIT, see LICENSE_GestureSign.txt). A word's prototype is the normalised
mean of its recorded takes; a new sign matches the closest prototype only if it is similar enough AND clearly
closer than the runner-up, otherwise the answer is "not sure". Everything is stored in one small JSON file per user.
"""
from __future__ import annotations

import json
import os
import re
import time

import numpy as np

MIN_TAKES = 3
MAX_TAKES = 6
DEFAULT_THRESH = {"minSimilarity": 0.55, "margin": 0.05}   # on a real test clip: true repeats scored 75-96%, a wrong sign 31%


def _unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class TeachStore:
    def __init__(self, path: str):
        self.path = path
        self.words: dict[str, list[list[float]]] = {}
        self.stamps: dict[str, float] = {}
        if os.path.isfile(path):
            try:
                raw = json.load(open(path, encoding="utf-8"))
                self.words = raw.get("words", {}); self.stamps = raw.get("stamps", {})
            except Exception:
                self.words = {}

    # ---- storage ---------------------------------------------------------
    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        json.dump({"version": 1, "words": self.words, "stamps": self.stamps}, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, self.path)

    @staticmethod
    def clean(name: str) -> str:
        return re.sub(r"\s+", " ", name.strip())[:40]

    def add_take(self, name: str, embedding) -> int:
        name = self.clean(name)
        if not name:
            raise ValueError("The word needs a name.")
        takes = self.words.setdefault(name, [])
        if len(takes) >= MAX_TAKES:
            takes.pop(0)                                   # keep the newest takes
        takes.append([float(x) for x in _unit(embedding)])
        self.stamps[name] = time.time()
        self.save()
        return len(takes)

    def delete(self, name: str) -> bool:
        name = self.clean(name)
        ok = self.words.pop(name, None) is not None
        self.stamps.pop(name, None)
        if ok:
            self.save()
        return ok

    def ready_words(self) -> list[str]:
        return [w for w, t in self.words.items() if len(t) >= MIN_TAKES]

    def summary(self) -> list[dict]:
        return [{"word": w, "takes": len(t), "ready": len(t) >= MIN_TAKES, "added": self.stamps.get(w)} for w, t in self.words.items()]

    # ---- matching --------------------------------------------------------
    def match(self, embedding, thresholds: dict | None = None) -> dict:
        th = {**DEFAULT_THRESH, **(thresholds or {})}
        ready = self.ready_words()
        if not ready:
            return {"kind": "empty"}
        e = _unit(embedding)
        scored = sorted(((float(e @ _unit(np.mean(self.words[w], axis=0))), w) for w in ready), reverse=True)
        best, second = scored[0], (scored[1] if len(scored) > 1 else None)
        run_up = second[0] if second else -1.0
        ok = best[0] >= th["minSimilarity"] and (second is None or best[0] - run_up >= th["margin"])
        return {"kind": "match" if ok else "not-sure", "word": best[1], "similarity": best[0], "runner_up": run_up}
