"""Read a video of single held letters (like an ASL alphabet lesson) in short windows.

    python tools/alphabet_test.py "Learn ASL Alphabet Video.mp4" [--win 2.0] [--hop 1.0]

Prints, for every short stretch of the video, what the models read, plus a summary sequence.
Output is also appended to logs/video_test.log.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tools.transcribe_video import log, landmarks_from_video


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video")
    ap.add_argument("--win", type=float, default=2.0); ap.add_argument("--hop", type=float, default=1.0)
    ap.add_argument("--min-hand", type=float, default=0.6, help="skip windows where the hand is visible less than this fraction")
    a = ap.parse_args()
    from core.fingerspell import Fingerspeller
    fs = Fingerspeller()
    log(f"---- {time.strftime('%Y-%m-%d %H:%M:%S')}  ALPHABET TEST  video: {a.video}  window {a.win}s hop {a.hop}s  models: {len(fs.models)}")
    frames, seen, fps = landmarks_from_video(a.video)
    log(f"frames: {len(frames)}  with a hand: {seen}  fps {fps:.0f}")
    n, w, h = len(frames), int(a.win * 30), int(a.hop * 30)       # landmarks_from_video resamples to ~30 fps
    seq, t = [], 0
    while t + w <= n:
        chunk = frames[t:t + w]
        vis = sum(c is not None for c in chunk) / len(chunk)
        if vis >= a.min_hand:
            text, conf, mir = fs.transcribe_landmarks(chunk)
            log(f"t={t/30:6.1f}s  hand {vis*100:3.0f}%  ->  '{text}'  ({conf:.2f})")
            if text: seq.append(text)
        else:
            log(f"t={t/30:6.1f}s  hand {vis*100:3.0f}%  ->  (no hand / too little)")
        t += h
    log("SEQUENCE: " + " ".join(seq))
    letters = "".join(s[0] for s in seq if s)
    log("FIRST LETTER OF EACH WINDOW: " + letters)


if __name__ == "__main__":
    main()
