"""Turn phone/camera VIDEOS of you signing into training rows (no webcam needed).

    python tools/extract_from_video.py <video.mp4> <LABEL> [--session NAME] [--every 3]

Film yourself holding ONE letter steady for ~5 s (phone camera, hand fully visible, plain
background).  Run once per letter.  Use a different --session for each filming day, light,
or background: that is how train.py measures accuracy on situations it has not seen.
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import append_rows, landmarker
import cv2

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("label")
    ap.add_argument("--session", default=time.strftime("%Y%m%d")); ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    mp, det = landmarker()
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise SystemExit("Cannot open video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30; i = 0; rows = []; miss = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        i += 1
        if i % a.every: continue
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)), int(i / fps * 1000))
        if res.hand_landmarks: rows.append((a.label.upper(), a.session, [(l.x, l.y, l.z) for l in res.hand_landmarks[0]]))
        else: miss += 1
    append_rows(rows); print(a.label.upper(), len(rows), "frames saved,", miss, "without a hand, session", a.session)

if __name__ == "__main__":
    main()
