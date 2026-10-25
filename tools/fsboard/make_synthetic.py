"""Fake FSboard-format shards (for testing the pipeline ONLY - not real signing).

    python tools/fsboard/make_synthetic.py data/synth

Each symbol is a fixed random hand pose; a clip holds each symbol for a few frames with noise,
position/size jitter and dropped frames.  Train/val/test use different 'signers' (noise styles).
"""
import sys, os, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seqex

SYMS = "abcdefghijklmnopqrstuvwxyz0123456789"


def make(out, n_train=400, n_val=80, n_test=80, seed=0):
    import pyarrow as pa, pyarrow.feather as pf
    rng = np.random.default_rng(seed); random.seed(seed)
    proto = {c: rng.normal(0, 1, (21, 3)).astype(np.float32) for c in SYMS}
    for c in proto:
        proto[c][0] = 0; proto[c][9] = [0, -1, 0]          # wrist at origin, palm size 1
    os.makedirs(out, exist_ok=True)

    def clip(signer_noise):
        n = random.randint(4, 9); text = "".join(random.choice(SYMS) for _ in range(n))
        frames = []
        for ch in text:
            for _ in range(random.randint(5, 9)):
                p = proto[ch] + rng.normal(0, signer_noise, (21, 3))
                s = random.uniform(0.15, 0.35); off = rng.uniform(0.2, 0.8, 2)
                q = p * s; q[:, 0] += off[0]; q[:, 1] += off[1]
                frames.append(q)
            if random.random() < 0.3:
                frames.append(None)
        hands = {f"{ax}_right_hand": [([float(v) for v in f[:, j]] if f is not None else []) for f in frames] for j, ax in enumerate("xyz")}
        return seqex.build(f"v{random.random()}", text, hands, len(frames))

    for split, n, noise in [("train", n_train, 0.05), ("val", n_val, 0.08), ("test", n_test, 0.08)]:
        shards = 2 if split == "train" else 1
        for k in range(shards):
            rows = [clip(noise) for _ in range(n // shards)]
            pf.write_feather(pa.table({"serialized": pa.array(rows, pa.binary())}), os.path.join(out, f"synth_v3-{split}.arrow-{k:05d}-of-{shards:05d}"))
    return out


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data/synth")
