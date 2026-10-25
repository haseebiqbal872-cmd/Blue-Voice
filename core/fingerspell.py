"""Fingerspelling recogniser: hand landmarks over time -> text.

Loads the trained models from models/fingerspell_*.pt (trained on Google's FSboard data), combines
them, and optionally snaps each word to the closest real English word (needs `pip install wordfreq`).
"""
from __future__ import annotations
import glob, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "tools", "fsboard"))


def normalise_hand(frames, mirror: bool):
    """frames: list of (21,3) arrays or None -> (T,63) float32, same recipe as training (wrist-centred, palm-scaled)."""
    T = len(frames)
    h = np.full((T, 21, 3), np.nan, np.float32)
    for i, f in enumerate(frames):
        if f is not None:
            h[i] = np.asarray(f, np.float32)
    if mirror:
        h[:, :, 0] *= -1
    present = ~np.isnan(h[:, 0, 0])
    h -= h[:, :1, :]
    scale = np.linalg.norm(h[:, 9, :2], axis=1)
    scale[~present | (scale < 1e-6)] = 1.0
    h /= scale[:, None, None]
    h[~present] = 0.0
    h = np.clip(np.nan_to_num(h), -6, 6)
    return h.reshape(T, 63).astype(np.float32)


class Fingerspeller:
    def __init__(self, model_dir: str | None = None, device: str = "cpu"):
        import torch
        from train_ctc import Net
        self.torch, self.dev = torch, torch.device(device)
        model_dir = model_dir or os.path.join(_HERE, "..", "models")
        paths = sorted(glob.glob(os.path.join(model_dir, "fingerspell_*.pt")))
        if not paths:
            raise FileNotFoundError(f"No fingerspell_*.pt model files in {os.path.abspath(model_dir)}")
        self.models, self.chars = [], None
        for p in paths:
            ck = torch.load(p, map_location=self.dev)
            m = Net(len(ck["chars"]), ck["hid"], ck.get("arch", "gru")).to(self.dev)
            m.load_state_dict(ck["model"]); m.eval()
            self.models.append(m); self.chars = ck["chars"]
        self.paths = paths
        self.idx = {c: i for i, c in enumerate(self.chars)}
        self._words = None

    # ---- inference -------------------------------------------------------------------------------------------
    def _logprobs(self, feats):
        torch = self.torch
        X = torch.from_numpy(feats)[None].to(self.dev); L = torch.tensor([len(feats)])
        from train_ctc import Net
        n = int(Net.out_len(L)[0])
        with torch.no_grad():
            return [m(X, L).log_softmax(-1)[0, :n].cpu().numpy() for m in self.models]

    def _greedy(self, lp):
        a = lp.argmax(1); prev = -1; out = []
        for c in a:
            if c != 0 and c != prev: out.append(self.chars[c])
            prev = c
        return "".join(out)

    def transcribe_feats(self, feats):
        """(T,63) -> (text, confidence 0..1). Candidates from every model, re-scored by all models together."""
        from decode_lm import beam_decode, ctc_ll
        if len(feats) < 8:
            return "", 0.0
        lps = self._logprobs(feats)
        cands = set()
        for lp in lps:
            cands.add(self._greedy(lp))
            cands.update(beam_decode(lp, self.chars, None, 0.0, 0.0, beam=12, nbest=6))
        cands.discard("")
        if not cands:
            return "", 0.0
        scored = [(sum(ctc_ll(lp, h, self.idx) for lp in lps) / len(lps), h) for h in cands]
        ll, best = max(scored)
        return best, float(np.exp(ll / max(1, lp_len(lps[0]))))

    def transcribe_landmarks(self, frames):
        """frames: list of (21,3) arrays / None (one per video frame). Tries the hand as-is and mirrored
        (camera conventions differ) and keeps whichever the models are more confident about."""
        best = ("", 0.0, False)
        for mirror in (False, True):
            text, conf = self.transcribe_feats(normalise_hand(frames, mirror))
            if conf > best[1] or not best[0]:
                best = (text, conf, mirror)
        return best

    # ---- word suggestion -------------------------------------------------------------------------------------
    def _load_words(self):
        if self._words is None:
            try:
                from wordfreq import top_n_list
                self._words = [w for w in top_n_list("en", 60000) if w.isalpha()]
            except Exception:
                self._words = []
        return self._words

    def suggest(self, word: str, n: int = 3):
        """Closest real words to a recognised word (edit distance, preferring common words)."""
        from train_ctc import edit_distance
        w = word.lower().strip()
        words = self._load_words()
        if not w or not words:
            return []
        lim = 1 if len(w) <= 4 else 2
        out = []
        for rank, c in enumerate(words):
            if abs(len(c) - len(w)) > lim: continue
            d = edit_distance(w, c)
            if d <= lim: out.append((d, rank, c))
        return [c for _, _, c in sorted(out)[:n]]


def lp_len(lp):
    return lp.shape[0]
