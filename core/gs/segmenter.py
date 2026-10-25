"""Cuts a continuous landmark stream into single-sign segments.

Python port of GestureSign's segmenter (MIT, see LICENSE_GestureSign.txt). A segment starts when a hand
is present and moving and ends when the hands are gone, or held still, for `gap_ms`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.gs import landmarks as lm
from core.gs.features import compute_anchor


@dataclass
class SegOpts:
    gap_ms: float = 300
    move_speed: float = 0.6      # shoulder widths per second
    min_frames: int = 8
    max_ms: float = 4000
    preroll_ms: float = 200
    leave_ms: float = 500


@dataclass
class Segment:
    frames: np.ndarray           # (T,543,3)
    fps: float
    duration_ms: float
    reason: str
    start_ms: float = 0.0


def hands_missing(frame: np.ndarray) -> bool:
    left = np.isfinite(frame[lm.LEFT_HAND_START:lm.LEFT_HAND_START + lm.N_HAND, 0]).any()
    right = np.isfinite(frame[lm.RIGHT_HAND_START:lm.RIGHT_HAND_START + lm.N_HAND, 0]).any()
    return not (left or right)


def hand_speed(prev, nxt) -> float:
    dt = (nxt[1] - prev[1]) / 1000.0
    if not dt > 0:
        return 0.0
    scale = compute_anchor(nxt[0][None]).scale
    tot, cnt = 0.0, 0
    for start in (lm.LEFT_HAND_START, lm.RIGHT_HAND_START):
        d = np.hypot(*(nxt[0][start:start + lm.N_HAND, :2] - prev[0][start:start + lm.N_HAND, :2]).T)
        d = d[np.isfinite(d)]
        tot += float(d.sum()); cnt += len(d)
    return tot / cnt / scale / dt if cnt else 0.0


class Segmenter:
    def __init__(self, **kw):
        self.o = SegOpts(**kw)
        self.reset()

    def reset(self):
        self.samples = []; self.recent = []; self.last = None
        self.speed = 0.0; self.quiet_since = None; self.quiet_reason = "absent"
        self.from_rest = False; self.state = "idle"

    def push(self, frame, t):
        """Feed one frame (543,3) at time t (ms). Returns ('start'|'end'|'discard', payload) or None."""
        s = (frame, t)
        present = not hands_missing(frame)
        if present and self.last is not None and not hands_missing(self.last[0]):
            self.speed = 0.5 * self.speed + 0.5 * hand_speed(self.last, s)
        elif not present:
            self.speed = 0.0
        self.last = s
        moving = present and self.speed >= self.o.move_speed
        if self.state != "signing":
            if not present:
                self.state = "idle"
            self.recent = [x for x in self.recent + [s] if t - x[1] <= self.o.preroll_ms] if present else []
            if not moving:
                return None
            self.from_rest = self.state == "resting"
            self.state = "signing"
            self.samples = list(self.recent)
            self.quiet_since = None
            return ("start", t)
        self.samples.append(s)
        if t - self.samples[0][1] >= self.o.max_ms:
            ev = self._finish("max", len(self.samples)); self.recent = []
            return ev
        if moving:
            self.quiet_since = None
            return None
        if self.quiet_since is None:
            self.quiet_since = t; self.quiet_reason = "still" if present else "absent"
        if t - self.quiet_since < self.o.gap_ms:
            return None
        keep = sum(1 for x in self.samples if x[1] < self.quiet_since)
        reason = self.quiet_reason
        if reason == "absent" and self.from_rest:
            kept = self.samples[:keep]
            span = (kept[-1][1] - kept[0][1]) if kept else 0
            if span < self.o.leave_ms:
                self._finish(reason, 0)
                return ("discard", "hands-leaving")
        ev = self._finish(reason, keep + 1 if reason == "still" else keep)
        self.state = "resting" if present else "idle"
        return ev

    def flush(self):
        if self.state != "signing":
            return None
        keep = len(self.samples) if self.quiet_since is None else sum(1 for x in self.samples if x[1] < self.quiet_since)
        return self._finish("flush", keep)

    def _finish(self, reason, keep):
        kept = self.samples[:max(0, keep)]
        self.samples = []; self.quiet_since = None; self.state = "idle"
        if len(kept) < self.o.min_frames:
            return ("discard", "too-short")
        dur = kept[-1][1] - kept[0][1]
        fps = (len(kept) - 1) * 1000 / dur if dur > 0 else 30.0
        return ("end", Segment(np.stack([x[0] for x in kept]), fps, dur, reason, kept[0][1]))
