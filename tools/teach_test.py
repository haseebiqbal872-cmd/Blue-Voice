"""Record-your-own-words from VIDEOS (the in-app Record button does the same from the camera).

  python tools/teach_test.py add  VIDEO WORD  T1-T2 T3-T4 T5-T6     teach WORD from 3 stretches of the video (seconds)
  python tools/teach_test.py read VIDEO [--start S --end E]         read the video with YOUR words first, then the built-in 250
  python tools/teach_test.py list                                   show your words
  python tools/teach_test.py delete WORD

Your words are saved in data/user_words.json (only on this computer).
"""
import argparse, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np
from tools.word_test import load_frames, log

STORE = os.path.join(ROOT, "data", "user_words.json")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["add", "read", "list", "delete"])
    ap.add_argument("rest", nargs="*"); ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--models", default=os.path.join(ROOT, "assets", "models"))
    a = ap.parse_args()
    from core.gs.teach import TeachStore
    store = TeachStore(STORE)
    if a.cmd == "list":
        for r in store.summary(): log(f"  {r['word']}: {r['takes']} takes" + ("" if r["ready"] else "  (needs 3)"))
        if not store.words: log("  (no words yet)")
        return
    if a.cmd == "delete":
        log("deleted" if store.delete(" ".join(a.rest)) else "no such word"); return
    from core.sign_words import WordRecognizer
    rec = WordRecognizer(os.path.join(ROOT, "models", "gs"))
    if a.cmd == "add":
        video, word, spans = a.rest[0], a.rest[1], a.rest[2:]
        frames, times, fps = load_frames(video, 0.0, None, a.models)
        for sp in spans:
            t1, t2 = (float(x) for x in sp.split("-"))
            sel = (times >= t1 * 1000) & (times <= t2 * 1000)
            seg = frames[sel]
            if len(seg) < 8: log(f"  {sp}: too short / nothing there, skipped"); continue
            n = store.add_take(word, rec.embed(seg, fps))
            log(f"  take {n} for '{word}' from {sp}s ({len(seg)} frames)")
        log(f"'{word}' now has {len(store.words.get(word, []))} takes" + ("  -> ready" if word in store.ready_words() else "  (needs at least 3)"))
        return
    # read
    from core.gs.segmenter import Segmenter
    video = a.rest[0]
    frames, times, fps = load_frames(video, a.start, a.end, a.models)
    sg = Segmenter(); out = []
    def handle(ev):
        if ev and ev[0] == "end":
            seg = ev[1]; r = rec.read(seg.frames, seg.fps, store)
            tag = "YOUR WORD" if r["source"] == "yours" else "built-in"
            txt = "  ".join(f"{w} {p*100:.0f}%" for w, p in r["top"])
            log(f"  [at {seg.start_ms/1000:5.1f}s]  {txt}   ({tag}{', not sure' if r['kind']=='not-sure' else ''})")
    for f, t in zip(frames, times): handle(sg.push(f, t))
    handle(sg.flush())

if __name__ == "__main__":
    main()
