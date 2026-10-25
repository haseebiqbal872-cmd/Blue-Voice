"""Word-level sign recogniser (250 ASL signs) using the GestureSign ONNX model (MIT).

    rec = WordRecognizer("models/gs")
    res = rec.predict(frames, fps)       # frames: (T,543,3) MediaPipe-Holistic layout, NaN = not seen
    res["kind"]  -> "sign" | "not-sure" | "nothing";  res["top"] -> [(word, probability), x3]

Credit: GestureSign (Shashank Yadav, MIT); model trained on Google "Isolated Sign Language Recognition" (CC BY 4.0).
"""
from __future__ import annotations

import json
import os

import numpy as np

from core.gs.features import sequence_features
from core.gs.resample import fit_length, resample


class WordRecognizer:
    def __init__(self, model_dir: str = "models/gs"):
        import onnxruntime as ort
        self.meta = json.load(open(os.path.join(model_dir, "model-meta.json"), encoding="utf-8"))
        labels = json.load(open(os.path.join(model_dir, "labels.json"), encoding="utf-8"))["signs"]
        self.words = [s["label"] for s in labels]
        self.shipped = np.array([s["shipped"] for s in labels])
        self.sess = ort.InferenceSession(os.path.join(model_dir, "model.onnx"), providers=["CPUExecutionProvider"])
        self.temp = self.meta["calibration"]["temperature"]
        self.not_sure = self.meta["calibration"]["notSureBelow"]
        self.max_frames = self.meta["maxFrames"]; self.min_frames = self.meta["minFrames"]
        self.fps = self.meta["fps"]; self.fset = self.meta["featureSet"]

    def run(self, frames: np.ndarray, fps: float):
        """-> (logits (250,), embedding (128,) unit length). Both come from one model pass."""
        x = resample(frames.astype(np.float32), fps, self.fps)
        feats = sequence_features(fit_length(x, self.max_frames), self.fset)
        logits, emb = self.sess.run(["logits", "embedding"], {"frames": feats[None].astype(np.float32)})
        return logits[0], emb[0]

    def embed(self, frames: np.ndarray, fps: float) -> np.ndarray:
        return self.run(frames, fps)[1]

    def probs(self, frames: np.ndarray, fps: float) -> np.ndarray:
        logits, _ = self.run(frames, fps)
        z = np.where(self.shipped, logits / self.temp, -np.inf)
        e = np.exp(z - z.max())
        return e / e.sum()

    def read(self, frames: np.ndarray, fps: float, store=None, k: int = 3) -> dict:
        """One sign -> the user's own recorded words first, then the built-in 250.

        {"source": "yours"|"built-in", "kind": "sign"|"not-sure"|"nothing", "top": [(word, score), ...]}
        """
        if len(frames) < self.min_frames:
            return {"source": "built-in", "kind": "nothing", "top": []}
        logits, emb = self.run(frames, fps)
        if store is not None and store.ready_words():
            m = store.match(emb)
            if m["kind"] == "match":
                return {"source": "yours", "kind": "sign", "top": [(m["word"], m["similarity"])]}
        z = np.where(self.shipped, logits / self.temp, -np.inf)
        e = np.exp(z - z.max()); p = e / e.sum()
        order = np.argsort(-p)[:k]
        top = [(self.words[i], float(p[i])) for i in order]
        return {"source": "built-in", "kind": "sign" if top[0][1] >= self.not_sure else "not-sure", "top": top}

    def predict(self, frames: np.ndarray, fps: float, k: int = 3) -> dict:
        if len(frames) < self.min_frames:
            return {"kind": "nothing", "top": []}
        p = self.probs(frames, fps)
        order = np.argsort(-p)[:k]
        top = [(self.words[i], float(p[i])) for i in order]
        kind = "sign" if top[0][1] >= self.not_sure else "not-sure"
        return {"kind": kind, "top": top}
