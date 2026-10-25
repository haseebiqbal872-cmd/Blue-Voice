"""Record your own labelled hand shapes from the webcam.

    python tools/collect.py

A small window shows your camera ONLY in this developer tool (the app never does).
Hold a hand shape, then press its key (a-z or 0-9): 60 frames are saved with that label.
Press ESC to quit. Use a different session (lighting, day, person) each time you run
it - the evaluation uses sessions to measure accuracy on situations it has not seen.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import append_rows, landmarker
import cv2

def main():
    mp, det = landmarker()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
    if not cap.isOpened():
        raise SystemExit("No camera.")
    session = time.strftime("%Y%m%d-%H%M%S"); rec = None; left = 0; rows = []; counts = {}
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        res = det.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), int(time.perf_counter() * 1000))
        pts = [(l.x, l.y, l.z) for l in res.hand_landmarks[0]] if res.hand_landmarks else None
        if left > 0 and pts:
            rows.append((rec, session, pts)); left -= 1; counts[rec] = counts.get(rec, 0) + 1
        msg = f"recording '{rec}' {left}" if left > 0 else "hold a shape, press its key (ESC quits)"
        cv2.putText(frame, msg + ("  hand OK" if pts else "  no hand"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 200, 0), 2)
        cv2.imshow("collect", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: break
        if chr(k).isalnum() if 0 < k < 128 else False:
            rec, left = chr(k).upper(), 60
    cap.release(); cv2.destroyAllWindows()
    append_rows(rows); print("saved", len(rows), "samples:", counts)

if __name__ == "__main__":
    main()
