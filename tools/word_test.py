"""Read SIGN WORDS from a video (250 ASL words, upper-body tracking).

    python tools/word_test.py <video> [--truth "hello"] [--whole] [--windows] [--start 0 --end 30] [--speak]

Default: the video is cut into separate signs wherever the hands rest for a moment, and each sign gets its
top-3 guesses (or "not sure"). --whole: the whole video is ONE sign. --windows: no cutting, a guess every
half second over a 1.5 s window (for fast, flowing signing). Everything is also written to logs/word_test.log.
Model: GestureSign (MIT) trained on Google "Isolated Sign Language Recognition" (CC BY 4.0).
"""
import argparse, os, sys, time
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import numpy as np

LOG = os.path.join(ROOT, "logs", "word_test.log")


def log(msg):
    print(msg, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def track_video(path, start, end, model_dir):
    import cv2
    from core.gs.tracker import UpperBodyTracker, missing_models, try_download, URLS
    miss = try_download(model_dir) if missing_models(model_dir) else []
    if miss:
        log("MISSING tracker files in " + os.path.abspath(model_dir) + ":")
        for n in miss:
            log(f"   {n}   <- open this link in your browser, it downloads the file:\n   {URLS[n]}")
        log("Put the downloaded file(s) into that folder (check the name still ends in .task), then run again.")
        sys.exit(2)
    tr = UpperBodyTracker(model_dir)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("Cannot open the video: " + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / 30.0)))
    frames, times, i = [], [], 0
    t0 = time.time()
    while True:
        ok, fr = cap.read()
        if not ok: break
        i += 1
        t = (i - 1) / fps
        if t < start: continue
        if end and t > end: break
        if (i - 1) % step: continue
        frames.append(tr.track(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), int(t * 1000)))
        times.append(t * 1000)
        if len(frames) % 60 == 0:
            print(f"  tracked {len(frames)} frames ({time.time()-t0:.0f}s) ...", flush=True)
    return np.stack(frames) if frames else np.zeros((0, 543, 3), np.float32), np.array(times), fps / step


def load_frames(video, start, end, models):
    """Landmarks of a video (cached in cache/ so a second run is instant)."""
    cache = os.path.join(ROOT, "cache", "landmarks_" + os.path.basename(video) + f"_{start}_{end}.npz")
    if os.path.isfile(cache):
        z = np.load(cache); log("(landmarks loaded from cache)"); return z["f"], z["t"], float(z["fps"])
    frames, times, fps = track_video(video, start, end, models)
    os.makedirs(os.path.dirname(cache), exist_ok=True); np.savez(cache, f=frames, t=times, fps=fps)
    return frames, times, fps


def fmt(res):
    return "  ".join(f"{w} {p*100:.0f}%" for w, p in res["top"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--truth", default="")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--whole", action="store_true"); ap.add_argument("--windows", action="store_true")
    ap.add_argument("--speak", action="store_true"); ap.add_argument("--models", default=os.path.join(ROOT, "assets", "models"))
    a = ap.parse_args()
    from core.sign_words import WordRecognizer
    from core.gs.segmenter import Segmenter, hands_missing
    rec = WordRecognizer(os.path.join(ROOT, "models", "gs"))
    log(f"---- {time.strftime('%Y-%m-%d %H:%M:%S')}  video: {a.video}")
    frames, times, fps = load_frames(a.video, a.start, a.end, a.models)
    n = len(frames)
    nh = int(sum(not hands_missing(f) for f in frames))
    body = int(sum(np.isfinite(f[489:522, 0]).any() for f in frames))
    log(f"frames: {n}  with a hand: {nh}  with a body: {body}  ({fps:.0f} fps)")
    if nh < 8:
        log("RESULT: hardly any hand found. The hands must be clearly visible, well lit, and not too small in the picture.")
        return
    results = []
    if a.whole:
        r = rec.predict(frames, fps); results.append(("whole video", r))
    elif a.windows:
        win, hop = int(1.5 * fps), int(0.5 * fps)
        for s in range(0, max(1, n - win + 1), hop):
            seg = frames[s:s + win]
            if sum(not hands_missing(f) for f in seg) < 8: continue
            results.append((f"{times[s]/1000:5.1f}s", rec.predict(seg, fps)))
    else:
        sg = Segmenter()
        def handle(ev):
            if ev and ev[0] == "end":
                seg = ev[1]
                results.append((f"at {(seg.start_ms)/1000:5.1f}s, {seg.duration_ms/1000:.1f}s long", rec.predict(seg.frames, seg.fps)))
        for f, t in zip(frames, times):
            handle(sg.push(f, t))
        handle(sg.flush())
        if not results:
            log("No separate signs were found (the hands never rested). Try --windows or --whole.")
    spoken = []
    for label, r in results:
        tag = {"sign": "", "not-sure": "  (not sure)", "nothing": "  (nothing)"}[r["kind"]]
        log(f"  [{label}]  {fmt(r)}{tag}")
        if r["kind"] == "sign": spoken.append(r["top"][0][0])
    log("WORDS (confident ones): " + " ".join(spoken))
    if a.truth:
        t = a.truth.lower().strip()
        hit1 = any(r["top"] and r["top"][0][0].lower() == t for _, r in results)
        hit3 = any(any(w.lower() == t for w, _ in r["top"]) for _, r in results)
        log(f"EXPECTED '{a.truth}': in a first guess: {hit1}   in a top-3: {hit3}")
    if a.speak and spoken:
        try:
            import pyttsx3; e = pyttsx3.init(); e.say(" ".join(spoken)); e.runAndWait()
        except Exception as ex:
            log("(speech not available: " + str(ex) + ")")


if __name__ == "__main__":
    main()
