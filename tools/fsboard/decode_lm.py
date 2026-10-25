"""Ensemble the trained models and decode with a character language model (CTC prefix beam search).

    python decode_lm.py --models a.pt b.pt --train "prep/*-train.npz*" --val "prep/*-validation.npz*" --test "prep/*-test.npz*"

The language-model weight is tuned on VALIDATION clips only; the TEST number is then measured once.
"""
import argparse, math, os, sys, random, time
from collections import defaultdict
from multiprocessing import Pool
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ctc import Net, Vocab, load, edit_distance, batches  # noqa

BOS, EOS = "^", "$"


class CharLM:
    """Interpolated absolute-discount n-gram over characters."""
    def __init__(self, texts, n=5, d=0.75):
        self.n, self.d = n, d
        self.cnt = [defaultdict(lambda: defaultdict(int)) for _ in range(n)]
        for t in texts:
            s = BOS * (n - 1) + t + EOS
            for i in range(n - 1, len(s)):
                for k in range(n):
                    self.cnt[k][s[i - k:i]][s[i]] += 1
        self.tot = [{h: sum(v.values()) for h, v in c.items()} for c in self.cnt]
        self.V = len({c for c in self.cnt[0][""]}) or 1
        self.cache = {}

    def prob(self, hist, c):
        key = (hist[-(self.n - 1):], c)
        if key in self.cache: return self.cache[key]
        hist = (BOS * (self.n - 1) + hist)[-(self.n - 1):]
        p = 1.0 / (self.V + 1)
        for k in range(0, self.n):
            h = hist[len(hist) - k:] if k else ""
            tot = self.tot[k].get(h)
            if not tot: continue
            c_hc = self.cnt[k][h].get(c, 0)
            p = max(c_hc - self.d, 0) / tot + (self.d * len(self.cnt[k][h]) / tot) * p
        r = math.log(max(p, 1e-12)); self.cache[key] = r
        return r


def lse(a, b):
    if a == -1e30: return b
    if b == -1e30: return a
    m = max(a, b); return m + math.log(math.exp(a - m) + math.exp(b - m))


def beam_decode(lp, chars, lm, alpha, beta, beam=12, topk=6, nbest=1):
    """lp: (T,V) log-probs, blank=0. Returns best string."""
    NEG = -1e30
    beams = {"": (0.0, NEG, 0.0)}      # prefix -> (log p_blank, log p_nonblank, lm score)
    for t in range(lp.shape[0]):
        row = lp[t]
        cand = np.argsort(-row)[:topk]
        nxt = defaultdict(lambda: [NEG, NEG, 0.0])
        for pre, (pb, pnb, lms) in beams.items():
            tot = lse(pb, pnb)
            for c in cand:
                p = float(row[c])
                if c == 0:
                    e = nxt[pre]; e[0] = lse(e[0], tot + p); e[2] = lms
                    continue
                ch = chars[c]
                last = pre[-1] if pre else None
                if ch == last:
                    e = nxt[pre]; e[1] = lse(e[1], pnb + p); e[2] = lms          # repeat collapses
                    npre = pre + ch; l = (lm.prob(pre, ch) * alpha + beta) if lm else 0.0
                    e2 = nxt[npre]; e2[1] = lse(e2[1], pb + p); e2[2] = lms + l
                else:
                    npre = pre + ch; l = (lm.prob(pre, ch) * alpha + beta) if lm else 0.0
                    e2 = nxt[npre]; e2[1] = lse(e2[1], tot + p); e2[2] = lms + l
        scored = sorted(nxt.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2], reverse=True)[:beam]
        beams = {k: tuple(v) for k, v in scored}
    ranked = sorted(beams.items(), key=lambda kv: lse(kv[1][0], kv[1][1]) + kv[1][2], reverse=True)
    if nbest > 1: return [k for k, _ in ranked[:nbest]]
    return ranked[0][0]


def lm_text(lm, text):
    s, h = 0.0, ""
    for c in text + EOS:
        s += lm.prob(h, c); h += c
    return s


def ctc_ll(lp, text, idx):
    """log-likelihood of a text under one model's (T,V) log-probs."""
    import torch.nn.functional as F
    ids = [idx[c] for c in text if c in idx]
    if not ids: return -1e4
    l = F.ctc_loss(torch.from_numpy(lp)[:, None, :], torch.tensor([ids]), [lp.shape[0]], [len(ids)], blank=0, reduction="sum", zero_infinity=True)
    return -float(l)


def joint_prepare(lps, chars, lm, beam=16, nbest=8):
    """Candidates for ONE clip with, for each, its log-likelihood under every model and its LM score."""
    idx = {c: i for i, c in enumerate(chars)}
    cands = set()
    for lp in lps:
        cands.update(beam_decode(lp, chars, lm, 0.4, 1.0, beam=beam, nbest=nbest))
        cands.add("".join(chars[c] for c, p in zip(lp.argmax(1), [-1] + list(lp.argmax(1))[:-1]) if c != 0 and c != p))
    return [(h, sum(ctc_ll(lp, h, idx) for lp in lps) / len(lps), lm_text(lm, h)) for h in cands]


def joint_pick(prepared, alpha, beta):
    return [max(c, key=lambda t: t[1] + alpha * t[2] + beta * len(t[0]))[0] for c in prepared]


_G = {}
def _work(args):
    i, alpha, beta = args
    if _G.get("joint"):
        return joint_prepare([l[i] for l in _G["lps"]], _G["chars"], _G["lm"])
    return beam_decode(_G["lps"][i], _G["chars"], _G["lm"], alpha, beta)


@torch.no_grad()
def get_logprobs(models, seqs, dev):
    out = [None] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    for k in range(0, len(order), 32):
        ids = order[k:k + 32]; chunk = [seqs[i] for i in ids]
        xs = [torch.from_numpy(x.astype(np.float32)) for x, _ in chunk]
        L = torch.tensor([len(x) for x in xs]); X = torch.zeros(len(xs), int(L.max()), 63)
        for j, x in enumerate(xs): X[j, :len(x)] = x
        lps = torch.stack([m(X.to(dev), L).log_softmax(-1) for m in models]).mean(0).log_softmax(-1).cpu().numpy()
        for j, i in enumerate(ids): out[i] = lps[j, :int(Net.out_len(L[j]))].astype(np.float32)
    return out


def score(preds, seqs, chars):
    keep = set(chars); e = n = ex = 0
    for p, (_, t) in zip(preds, seqs):
        ref = "".join(c for c in t if c in keep); e += edit_distance(p, ref); n += max(1, len(ref)); ex += p == ref
    return 1 - e / max(1, n), ex / max(1, len(seqs))


def decode_all(lps, chars, lm, alpha, beta, procs, joint=False):
    _G.update(lps=lps, chars=chars, lm=lm, joint=joint)
    n = len(lps[0]) if joint else len(lps)
    with Pool(procs) as pool:
        return pool.map(_work, [(i, alpha, beta) for i in range(n)], chunksize=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--val-clips", type=int, default=300)
    ap.add_argument("--order", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, chars = [], None
    for p in a.models:
        ck = torch.load(p, map_location=dev); m = Net(len(ck["chars"]), ck["hid"], ck.get("arch", "gru")).to(dev)
        m.load_state_dict(ck["model"]); m.eval(); models.append(m); chars = ck["chars"]
    print("models:", a.models, flush=True)
    tr = load(a.train); va = load(a.val); te = load(a.test)
    lm = CharLM([t for _, t in tr], n=a.order)
    greedy = lambda lps: ["".join(chars[c] for c, prev in zip(r.argmax(1), [-1] + list(r.argmax(1))[:-1]) if c != 0 and c != prev) for r in lps]
    names = [os.path.basename(p) for p in a.models]
    random.seed(0); vsub = random.sample(va, min(a.val_clips, len(va)))
    # per-model log-probs (each model alone; they are NOT averaged: CTC spikes of different models are not aligned)
    vl = [get_logprobs([m], vsub, dev) for m in models]; tl = [get_logprobs([m], te, dev) for m in models]
    for n_, v_ in zip(names, vl): print(f"validation subset, greedy, {n_}: {score(greedy(v_), vsub, chars)[0]*100:.1f}%", flush=True)
    # candidate methods, all tuned on VALIDATION only
    methods = []                      # (val_acc, label, models idx list, alpha, beta, joint)
    for k, n_ in enumerate(names):
        methods.append((score(greedy(vl[k]), vsub, chars)[0], f"{n_} greedy", [k], 0.0, 0.0, False))
        for alpha in (0.3, 0.6):
            for beta in (0.5, 1.0):
                acc = score(decode_all(vl[k], chars, lm, alpha, beta, a.procs), vsub, chars)[0]
                print(f"  {n_} + LM alpha {alpha} beta {beta}: {acc*100:.1f}%", flush=True)
                methods.append((acc, f"{n_} + LM", [k], alpha, beta, False))
    if len(models) > 1:
        vprep = decode_all(vl, chars, lm, 0, 0, a.procs, joint=True)
        for alpha in (0.1, 0.2, 0.3, 0.45, 0.6, 0.8):
            for beta in (0.0, 0.5, 1.0, 1.5, 2.0):
                acc = score(joint_pick(vprep, alpha, beta), vsub, chars)[0]
                print(f"  JOINT all models + LM alpha {alpha} beta {beta}: {acc*100:.1f}%", flush=True)
                methods.append((acc, "joint", list(range(len(models))), alpha, beta, True))
    acc, label, ks, alpha, beta, joint = max(methods, key=lambda m: m[0])
    print(f"chosen on validation: {label}  alpha {alpha} beta {beta}  ({acc*100:.1f}%)", flush=True)
    sel = [tl[k] for k in ks]
    if joint: preds = joint_pick(decode_all(sel, chars, lm, 0, 0, a.procs, joint=True), alpha, beta)
    elif alpha > 0: preds = decode_all(sel[0], chars, lm, alpha, beta, a.procs)
    else: preds = greedy(sel[0])
    g, gx = score(greedy(tl[ks[0]]), te, chars)
    c, cx = score(preds, te, chars)
    print("TEST first chosen model greedy        character accuracy: %.1f%%   exact: %.1f%%" % (100 * g, 100 * gx), flush=True)
    print("TEST final method (%s) character accuracy: %.1f%%   exact: %.1f%%   (%d clips)  <- the number that matters" % (label, 100 * c, 100 * cx, len(te)), flush=True)


if __name__ == "__main__":
    main()
