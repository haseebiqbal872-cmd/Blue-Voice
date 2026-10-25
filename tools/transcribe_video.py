"""Run the fingerspelling models on a VIDEO of someone spelling (phone video, practice video, ...).

    python tools/transcribe_video.py <video> [--truth "hello"] [--start 0 --end 60]

Prints what the model thinks was spelled, a confidence, and (with --truth) whether it matched.
Everything is also appended to logs/video_test.log so Claude can read it.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import cv2, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def log(msg):
    print(msg, flush=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "video_test.log"), "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def landmarks_from_video(path, start=0.0, end=None):
    from tools._common import landmarker
    mp, det = landmarker()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit("Cannot open the video: " + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / 30.0)))            # the models were trained on ~30 frames per second
    frames, i, seen = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        i += 1
        t = i / fps
        if t < start: continue
        if end and t > end: break
        if i % step: continue
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(t * 1000))
        if res.hand_landmarks:
            frames.append([(l.x, l.y, l.z) for l in res.hand_landmarks[0]]); seen += 1
        else:
            frames.append(None)
    return frames, seen, fps


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--truth", default="")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--end", type=float, default=None)
    a = ap.parse_args()
    from core.fingerspell import Fingerspeller
    from tools.fsboard.train_ctc import edit_distance
    fs = Fingerspeller()
    log(f"---- {time.strftime('%Y-%m-%d %H:%M:%S')}  video: {a.video}  models: {len(fs.models)}")
    frames, seen, fps = landmarks_from_video(a.video, a.start, a.end)
    log(f"frames used: {len(frames)}  with a hand: {seen}  (video fps {fps:.0f})")
    if seen < 8:
        log("RESULT: no hand found in this video (the hand must be clearly visible, well lit, filling a good part of the picture).")
        return
    text, conf, mirrored = fs.transcribe_landmarks(frames)
    log(f"RESULT: '{text}'   confidence {conf:.2f}   (hand mirrored: {mirrored})")
    for w in text.split():
        s = fs.suggest(w)
        if s: log(f"  word '{w}' -> suggestions: {', '.join(s)}")
    if a.truth:
        t = a.truth.lower().strip(); p = text.lower().strip()
        cer = edit_distance(p, t) / max(1, len(t))
        log(f"EXPECTED: '{t}'  ->  {'CORRECT' if p == t else 'wrong'}   letter accuracy {max(0, 1 - cer) * 100:.0f}%")


if __name__ == "__main__":
    main()
