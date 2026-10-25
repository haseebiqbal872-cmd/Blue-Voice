"""
Static hand-shape (letter / number) classifier.

Landmarks -> normalised feature vector -> scikit-learn model trained from
*your own* labelled data (see tools/collect.py, tools/extract_from_images.py,
tools/train.py).  No model ships with the app: until ``models/letters.joblib``
exists, :class:`LetterClassifier.available` is False and no letters are produced.

Only handshapes that do not need motion can be classified this way
(most ASL letters and numbers; J and Z need motion and are not covered).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "letters.joblib")


def normalise(points: Sequence[Sequence[float]]) -> np.ndarray:
    """21 (x, y, z) landmarks -> 63 features, invariant to position and size."""
    p = np.asarray(points, dtype=np.float32).reshape(21, 3).copy()
    p -= p[0]                                   # wrist at the origin
    scale = float(np.linalg.norm(p[9][:2])) or 1.0   # wrist -> middle-finger base
    return (p / scale).reshape(-1)


_I, _J = np.triu_indices(21, 1)


def features(points) -> np.ndarray:
    """Rotation-, mirror-, position- and size-invariant features: all 210
    pairwise landmark distances (x, y) divided by the wrist->middle-base length.
    Accepts 21x3 points or a flat 63 vector.  (Measured: far better across
    different datasets than raw coordinates.)"""
    p = np.asarray(points, dtype=np.float32).reshape(-1, 21, 3)[:, :, :2]
    d = np.linalg.norm(p[:, _I, :] - p[:, _J, :], axis=2)
    s = np.linalg.norm(p[:, 9, :] - p[:, 0, :], axis=1, keepdims=True)
    s[s == 0] = 1.0
    return (d / s).astype(np.float32)


class LetterClassifier:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = os.path.abspath(path or MODEL_PATH)
        self._model = None
        self.labels: List[str] = []
        self._load()

    def _load(self) -> None:
        if not os.path.isfile(self.path):
            return
        try:
            import joblib
            data = joblib.load(self.path)
            self._model = data["model"]
            self.labels = list(data["labels"])
        except Exception:
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def predict(self, points) -> Tuple[Optional[str], float]:
        if not self.available:
            return None, 0.0
        x = features(points)
        proba = self._model.predict_proba(x)[0]
        i = int(np.argmax(proba))
        return str(self._model.classes_[i]), float(proba[i])


class LetterCommitter:
    """Turns noisy per-frame predictions into single committed characters.

    A letter is committed when it has been the confident prediction for
    ``stable_frames`` frames in a row.  The same letter is not repeated until
    the hand changes shape or leaves the view (so holding a letter types it once).
    """

    def __init__(self, stable_frames: int = 12, min_conf: float = 0.85) -> None:
        self.stable_frames = stable_frames
        self.min_conf = min_conf
        self.reset()

    def reset(self) -> None:
        self._cand: Optional[str] = None
        self._count = 0
        self._last_committed: Optional[str] = None

    def update(self, label: Optional[str], conf: float) -> Optional[str]:
        if label is None:
            self._cand, self._count, self._last_committed = None, 0, None
            return None
        if conf < self.min_conf:
            self._cand, self._count = None, 0
            return None
        if label != self._cand:
            self._cand, self._count = label, 1
        else:
            self._count += 1
        if label == self._last_committed:
            return None
        if self._count >= self.stable_frames:
            self._last_committed = label
            return label
        return None
