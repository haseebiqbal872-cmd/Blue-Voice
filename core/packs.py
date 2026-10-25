"""Sign packs: share your recorded signs as a small file, or add signs someone else recorded.
A pack only works with the same recognition model, so it carries the model version and is refused otherwise."""
from __future__ import annotations

import math

from core.gs.teach import MAX_TAKES, TeachStore

FORMAT = "bluevoice-signs"
MAX_WORDS = 300


def export_pack(store: TeachStore, model: str, dim: int) -> dict:
    return {"format": FORMAT, "version": 1, "model": model, "dim": dim,
            "words": {w: t for w, t in store.words.items() if t}}


def import_pack(store: TeachStore, pack, model: str, dim: int) -> tuple[int, int]:
    """Merges a pack into the store. Returns (words added or updated, takes added). Raises ValueError with a plain message."""
    if not isinstance(pack, dict) or pack.get("format") != FORMAT:
        raise ValueError("This is not a Blue Voice sign pack.")
    if pack.get("model") != model or pack.get("dim") != dim:
        raise ValueError("This pack was made for a different recognition model, so it can't be used here.")
    words = pack.get("words")
    if not isinstance(words, dict) or not words:
        raise ValueError("The pack has no signs in it.")
    if len(words) > MAX_WORDS:
        raise ValueError(f"A pack can hold at most {MAX_WORDS} signs.")
    clean: dict[str, list[list[float]]] = {}
    for w, takes in words.items():
        name = TeachStore.clean(str(w))
        if not name or not isinstance(takes, list):
            raise ValueError("The pack is damaged.")
        rows = []
        for t in takes[-MAX_TAKES:]:
            if not isinstance(t, list) or len(t) != dim or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in t):
                raise ValueError("The pack is damaged.")
            n = math.sqrt(sum(x * x for x in t))
            if n == 0:
                raise ValueError("The pack is damaged.")
            rows.append([float(x) / n for x in t])
        if rows:
            clean[name] = rows
    nt = 0
    for name, rows in clean.items():
        cur = store.words.setdefault(name, [])
        cur.extend(rows); del cur[:-MAX_TAKES]
        nt += len(rows)
    store.save()
    return len(clean), nt
