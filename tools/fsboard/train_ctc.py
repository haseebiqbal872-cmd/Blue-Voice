"""Train a fingerspelling recogniser (hand landmarks over time -> text) with CTC.

    python tools/fsboard/train_ctc.py --train "data/fsboard_prepared/*train*.npz*" \
        --val "data/fsboard_prepared/*val*.npz*" --test "data/fsboard_prepared/*test*.npz*" \
        --minutes 60

Reports CHARACTER ACCURACY = 1 - character error rate on signers the model never saw
(FSboard's val/test splits use different signers than train).  This is the honest number.
Writes logs/train.log and the best model to models/fingerspell.pt.
"""
import argparse, copy, glob, json, math, os, random, sys, time
import numpy as np
import torch
import torch.nn as nn

_I, _J = np.triu_indices(21, 1)
_I, _J = torch.tensor(_I), torch.tensor(_J)


def load(patterns):
    seqs = []
    for pat in patterns.split(","):
        for p in sorted(glob.glob(pat)):
            z = np.load(p, allow_pickle=False)
            X, off, prompts = z["X"], z["offsets"], z["prompts"]
            for i in range(len(prompts)):
                t = str(prompts[i]).lower()
                if t and off[i + 1] - off[i] >= 8:
                    seqs.append((X[off[i]:off[i + 1]], t))
    return seqs


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def featurize(c):
    """c: (B,T,63) normalised coords -> (B,T,336): coords + 210 pairwise distances + velocity."""
    p = c.view(c.shape[0], c.shape[1], 21, 3)[..., :2]
    d = (p[:, :, _I, :] - p[:, :, _J, :]).norm(dim=-1)
    v = torch.zeros_like(c); v[:, 1:] = c[:, 1:] - c[:, :-1]
    return torch.cat([c, d, v], dim=-1)


class Net(nn.Module):
    def __init__(self, nv, hid=192, arch="gru"):
        super().__init__()
        self.arch = arch
        self.inp = nn.Sequential(nn.Linear(63 + 210 + 63, hid), nn.GELU(), nn.Dropout(0.1))
        self.conv = nn.Sequential(nn.Conv1d(hid, hid, 5, stride=2, padding=2), nn.GELU(),
                                  nn.Conv1d(hid, hid, 5, padding=2), nn.GELU())
        if arch == "tf":
            layer = nn.TransformerEncoderLayer(hid, 4, hid * 4, dropout=0.15, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, 5, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(hid)
            self.out = nn.Linear(hid, nv)
        else:
            self.gru = nn.GRU(hid, hid, 2, batch_first=True, bidirectional=True, dropout=0.2)
            self.out = nn.Linear(2 * hid, nv)

    @staticmethod
    def out_len(l):
        return (l - 1) // 2 + 1

    def forward(self, c, L=None):
        x = self.inp(featurize(c))
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        if self.arch == "tf":
            T, H = x.shape[1], x.shape[2]
            pos = torch.arange(T, device=x.device)[:, None] / (10000 ** (torch.arange(0, H, 2, device=x.device) / H))[None]
            pe = torch.zeros(T, H, device=x.device); pe[:, 0::2] = torch.sin(pos); pe[:, 1::2] = torch.cos(pos)
            x = x + pe
            mask = None
            if L is not None:
                n = Net.out_len(L.to(x.device)).clamp(min=1, max=T)
                mask = torch.arange(T, device=x.device)[None] >= n[:, None]
            return self.out(self.norm(self.enc(x, src_key_padding_mask=mask)))
        if L is not None:      # pack so padding never reaches the backward GRU direction
            n = Net.out_len(L.cpu()).clamp(min=1, max=x.shape[1])
            pk = nn.utils.rnn.pack_padded_sequence(x, n, batch_first=True, enforce_sorted=False)
            y, _ = self.gru(pk)
            x, _ = nn.utils.rnn.pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
        else:
            x, _ = self.gru(x)
        return self.out(x)                      # (B,T',V)


def augment(x):
    """x (T,63) float tensor.  Rotation, scale, noise and speed changes."""
    T = x.shape[0]
    p = x.view(T, 21, 3).clone()
    a = math.radians(random.uniform(-20, 20)); c, s = math.cos(a), math.sin(a)
    xy = p[..., :2] @ torch.tensor([[c, -s], [s, c]], dtype=p.dtype).T
    sh = random.uniform(-0.15, 0.15)
    xy = xy @ torch.tensor([[1.0, 0.0], [sh, 1.0]], dtype=p.dtype)
    p[..., :2] = xy * torch.tensor([random.uniform(0.88, 1.12), random.uniform(0.88, 1.12)], dtype=p.dtype)
    p += torch.randn_like(p) * 0.01
    x = p.view(T, 63)
    f = random.uniform(0.8, 1.25)
    n = max(8, int(T * f))
    idx = torch.linspace(0, T - 1, n)
    lo = idx.floor().long(); hi = idx.ceil().long().clamp(max=T - 1); w = (idx - lo)[:, None]
    x = x[lo] * (1 - w) + x[hi] * w
    for _ in range(random.choice([0, 0, 1, 2])):          # time masking: blank out a short span
        n = x.shape[0]; ln = random.randint(1, max(1, n // 12)); st = random.randint(0, max(0, n - ln))
        x = x.clone(); x[st:st + ln] = 0
    return x


class Vocab:
    def __init__(self, chars):
        self.chars = ["<blank>"] + sorted(set(chars)); self.idx = {c: i for i, c in enumerate(self.chars)}

    def enc(self, t): return [self.idx[c] for c in t if c in self.idx]

    def decode(self, ids):
        out, prev = [], 0
        for i in ids:
            if i != prev and i != 0: out.append(self.chars[i])
            prev = i
        return "".join(out)


def batches(seqs, vocab, bs, train):
    order = list(range(len(seqs)))
    if train:
        random.shuffle(order)
        mega = bs * 25                      # sort inside big chunks: similar lengths -> little padding
        order = [i for k in range(0, len(order), mega) for i in sorted(order[k:k + mega], key=lambda i: len(seqs[i][0]))]
        starts = list(range(0, len(order), bs)); random.shuffle(starts)
    else:
        order.sort(key=lambda i: len(seqs[i][0]))
        starts = list(range(0, len(order), bs))
    for k in starts:
        chunk = [seqs[i] for i in order[k:k + bs]]
        xs = [torch.from_numpy(x.astype(np.float32)) for x, _ in chunk]
        if train: xs = [augment(x) for x in xs]
        L = torch.tensor([len(x) for x in xs])
        X = torch.zeros(len(xs), int(L.max()), 63)
        for i, x in enumerate(xs): X[i, :len(x)] = x
        yield X, L, [t for _, t in chunk]


@torch.no_grad()
def evaluate(model, seqs, vocab, bs=32):
    model.eval(); edits = chars = exact = 0
    for X, L, texts in batches(seqs, vocab, bs, False):
        lp = model(X.to(next(model.parameters()).device), L).argmax(-1).cpu()
        for i, t in enumerate(texts):
            ol = int(Net.out_len(L[i]))
            pred = vocab.decode(lp[i, :ol].tolist()); ref = "".join(c for c in t if c in vocab.idx)
            edits += edit_distance(pred, ref); chars += max(1, len(ref)); exact += pred == ref
    return max(0.0, 1 - edits / chars), exact / max(1, len(seqs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--test", default="")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--minutes", type=float, default=0)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hid", type=int, default=192); ap.add_argument("--arch", default="gru"); ap.add_argument("--ema", type=float, default=0.0)
    ap.add_argument("--out", default="models/fingerspell.pt"); ap.add_argument("--log", default="logs/train.log")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(os.path.dirname(a.log) or ".", exist_ok=True); os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    logf = open(a.log, "a", buffering=1)
    def log(*m):
        s = " ".join(str(x) for x in m); print(s, flush=True); logf.write(s + "\n")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = load(a.train), load(a.val)
    if not tr or not va: raise SystemExit("No training/validation data found. Check the --train / --val patterns.")
    vocab = Vocab("".join(t for _, t in tr))
    log(f"train {len(tr)} clips, val {len(va)} clips, {len(vocab.chars)-1} symbols, device {dev}")
    model = Net(len(vocab.chars), a.hid, a.arch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps = a.epochs * math.ceil(len(tr) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps, 10))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    ema = copy.deepcopy(model).eval() if a.ema > 0 else None
    best, t0 = -1.0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train(); tot = n = 0
        for X, L, texts in batches(tr, vocab, a.batch, True):
            tg = [torch.tensor(vocab.enc(t)) for t in texts]
            lp = model(X.to(dev), L).log_softmax(-1).transpose(0, 1)
            loss = ctc(lp, torch.cat(tg).to(dev), Net.out_len(L).tolist(), [len(t) for t in tg])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            if sched is not None and sched.last_epoch < steps - 1: sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
            tot += float(loss.detach()); n += 1
        acc, exact = evaluate(model, va, vocab); src = model
        if ema is not None:
            acc_e, exact_e = evaluate(ema, va, vocab)
            if acc_e >= acc: acc, exact, src = acc_e, exact_e, ema
        log(f"epoch {ep}/{a.epochs} loss {tot/max(1,n):.3f}  val char-accuracy {acc*100:.1f}%  exact {exact*100:.1f}%  {"(ema) " if src is ema and ema is not None else ""}{time.time()-t0:.0f}s")
        if best > 0.3 and acc < 0.5 * best:      # training collapsed: go back to the best weights, halve the learning rate
            ck = torch.load(a.out, map_location=dev); model.load_state_dict(ck["model"])
            if ema is not None: ema.load_state_dict(ck["model"])
            for g in opt.param_groups: g["lr"] = g["lr"] * 0.5
            sched = None; log("collapse detected: reloaded best weights, learning rate halved")
        if acc > best:
            best = acc
            torch.save({"model": src.state_dict(), "chars": vocab.chars, "hid": a.hid, "arch": a.arch, "val_char_acc": acc}, a.out)
        if a.minutes and (time.time() - t0) / 60 > a.minutes:
            log("time budget reached"); break
    log(f"BEST validation character accuracy: {best*100:.1f}%  (unseen signers)")
    if a.test:
        te = load(a.test)
        if te:
            ck = torch.load(a.out, map_location=dev); model.load_state_dict(ck["model"])
            acc, exact = evaluate(model, te, vocab)
            log(f"TEST character accuracy: {acc*100:.1f}%   exact phrase match: {exact*100:.1f}%   ({len(te)} clips)  <- the number that matters")


if __name__ == "__main__":
    main()
