"""Train and MEASURE the letter classifier.

    python tools/train.py [--csv data/landmarks.csv] [--keep-jz]

Honest accuracy = leave-one-SESSION-out: train on all sessions except one, test on the
one left out, repeat for each session, report each and the mean.  A session is a source
(person / camera / day / dataset).  The random-frame number is printed only as a warning.
J and Z need motion, so they are skipped unless --keep-jz.
Target: 90-95%+ on held-out sessions.  Letters under 90% are listed so you know what to fix.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load, DATA
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from core.letter_classifier import features


def make():
    return make_pipeline(StandardScaler(), MLPClassifier((256, 128), max_iter=300, random_state=0))


def evaluate(F, y, g):
    out = {"per_session": {}, "weak": {}}
    sessions = list(np.unique(g))
    if len(sessions) >= 2:
        weak_acc = {}
        for s in sessions:
            te = g == s
            m = make().fit(F[~te], y[~te]); p = m.predict(F[te])
            out["per_session"][str(s)] = float((p == y[te]).mean())
            for l in np.unique(y[te]):
                weak_acc.setdefault(str(l), []).append(float((p[y[te] == l] == l).mean()))
        out["session_acc"] = float(np.mean(list(out["per_session"].values())))
        out["weak"] = {l: float(np.mean(v)) for l, v in weak_acc.items() if np.mean(v) < 0.9}
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=0, stratify=y)
    out["random_acc"] = float((make().fit(F[tr], y[tr]).predict(F[te]) == y[te]).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DATA)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "letters.joblib"))
    ap.add_argument("--keep-jz", action="store_true")
    a = ap.parse_args()
    X, y, g = load(a.csv)
    if not a.keep_jz:
        k = ~np.isin(y, ["J", "Z"]); X, y, g = X[k], y[k], g[k]
    F = features(X)
    print(f"{len(y)} samples, {len(set(y))} labels, {len(set(g))} sessions")
    r = evaluate(F, y, g)
    if "session_acc" in r:
        for s, v in r["per_session"].items():
            print(f"  held out '{s}': {v*100:.1f}%")
        print(f"HELD-OUT SESSION accuracy (mean): {r['session_acc']*100:.1f}%   <- the number that matters")
        if r["weak"]:
            print("letters under 90% on held-out sessions:", {k: round(v * 100) for k, v in sorted(r["weak"].items())})
    else:
        print("Only one session: cannot measure real accuracy. Add another session (other person / camera / day).")
    print(f"random-frame accuracy: {r['random_acc']*100:.1f}%  (optimistic, do not rely on it)")
    m = make().fit(F, y)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    joblib.dump({"model": m, "labels": sorted(set(y))}, a.out); print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
