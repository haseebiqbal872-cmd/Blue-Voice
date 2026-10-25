"""FSboard landmark shards (.arrow) -> compact hand-only features (.npz).

    python tools/fsboard/prepare.py data/fsboard/**/*.arrow-* [--out data/fsboard_prepared]

Keeps only the signing hand (the one visible in more frames; a left hand is mirrored so every
sequence looks right-handed), wrist-centred and size-normalised.  Face/pose are dropped, so a
~600 MB shard becomes a small file.  Output per shard: X (frames,63) float16, offsets, prompts.
"""
import sys, os, glob, argparse, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import seqex


def hand_array(lists, side, T):
    """-> (T,21,3) float32 with NaN for missing frames."""
    out = np.full((T, 21, 3), np.nan, np.float32)
    for j, axis in enumerate("xyz"):
        rows = lists.get(f"{axis}_{side}_hand", [])
        for t, r in enumerate(rows[:T]):
            if len(r) == 21:
                out[t, :, j] = r
    return out


def normalise_seq(h, mirror):
    """(T,21,3) with NaN -> (T,63) wrist-centred, scaled by palm size, zeros where missing."""
    h = h.copy()
    if mirror:
        h[:, :, 0] *= -1
    present = ~np.isnan(h[:, 0, 0])
    h -= h[:, :1, :]                       # wrist at origin
    scale = np.linalg.norm(h[:, 9, :2], axis=1)
    scale[~present | (scale < 1e-6)] = 1.0
    h /= scale[:, None, None]
    h[~present] = 0.0
    h = np.clip(np.nan_to_num(h), -6, 6)
    return h.reshape(len(h), 63), present


def process_example(serialized):
    ctx, lists = seqex.parse(serialized)
    T = int(ctx.get("num_frames", 0)) or max((len(v) for v in lists.values()), default=0)
    if T == 0:
        return None
    L, R = hand_array(lists, "left", T), hand_array(lists, "right", T)
    nl, nr = int((~np.isnan(L[:, 0, 0])).sum()), int((~np.isnan(R[:, 0, 0])).sum())
    if max(nl, nr) < 5:
        return None
    left = nl > nr
    feats, present = normalise_seq(L if left else R, mirror=left)
    return ctx.get("prompt", ""), ctx.get("video_name", ""), feats.astype(np.float16), left


def prepare_shard(path, out_dir):
    import pyarrow.feather as pf
    table = pf.read_table(path, columns=["serialized"])
    X, off, prompts, names, lefts = [], [0], [], [], []
    for chunk in table.columns[0].chunks:
        for item in chunk:
            r = process_example(item.as_py())
            if r is None:
                continue
            prompt, name, f, left = r
            X.append(f); off.append(off[-1] + len(f)); prompts.append(prompt); names.append(name); lefts.append(left)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(path).replace(".arrow-", ".npz-") if ".arrow-" in path else os.path.basename(path) + ".npz")
    if not out.endswith(".npz") and ".npz-" not in out:
        out += ".npz"
    np.savez_compressed(out, X=np.concatenate(X) if X else np.zeros((0, 63), np.float16),
                        offsets=np.array(off, np.int64), prompts=np.array(prompts), names=np.array(names),
                        left=np.array(lefts))
    return out, len(prompts), off[-1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--out", default="data/fsboard_prepared")
    a = ap.parse_args()
    files = []
    for f in a.files:
        files += glob.glob(f, recursive=True) if any(c in f for c in "*?") else [f]
    for f in sorted(set(files)):
        if not re.search(r"\.arrow-\d{5}-of-\d{5}$", f):
            continue
        out, n, frames = prepare_shard(f, a.out)
        print(f"{os.path.basename(f)} -> {out}: {n} clips, {frames} frames", flush=True)


if __name__ == "__main__":
    main()
