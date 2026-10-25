"""Open-palm detector: all five fingers straight, hand held still. Used to start a recording hands-free."""
from __future__ import annotations

import numpy as np

from core.gs import landmarks as lm

HOLD_MS = 800
FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))      # (tip, middle joint)


def palm_wrist(frame: np.ndarray):
    """Wrist position of a hand showing five extended fingers, else None."""
    for start in (lm.LEFT_HAND_START, lm.RIGHT_HAND_START):
        h = frame[start:start + lm.N_HAND, :2]
        if not np.isfinite(h).all():
            continue
        d = lambda a, b: float(np.hypot(*(h[a] - h[b])))
        size = d(0, 9)                                # wrist to middle knuckle
        if size < 1e-4:
            continue
        if not all(d(t, 0) > 1.25 * d(p, 0) for t, p in FINGERS):
            continue
        if d(4, 17) < 1.1 * d(3, 17):                 # thumb out, away from the little finger
            continue
        if d(8, 0) < 1.6 * size:                      # fingers really long, not just slightly open
            continue
        return h[0].copy()
    return None


def open_palm(frame: np.ndarray) -> bool:
    return palm_wrist(frame) is not None


class PalmHold:
    """Feed frames; `push` returns True once an open palm has been held for HOLD_MS."""

    def __init__(self, hold_ms: float = HOLD_MS):
        self.hold_ms, self.since, self.anchor = hold_ms, None, None

    def reset(self):
        self.since, self.anchor = None, None

    def push(self, frame: np.ndarray, t_ms: float) -> bool:
        w = palm_wrist(frame)
        if w is not None and self.anchor is not None and float(np.hypot(*(w - self.anchor))) > 0.06:
            self.since = None                         # the hand moved: it is not being held still
        if w is None:
            self.since = None
            return False
        if self.since is None:
            self.since, self.anchor = t_ms, w
        return t_ms - self.since >= self.hold_ms
