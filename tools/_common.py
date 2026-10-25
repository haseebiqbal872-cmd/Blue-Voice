import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from core.letter_classifier import normalise

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "landmarks.csv")
HEADER = ["label", "session"] + [f"f{i}" for i in range(63)]


def append_rows(rows, path=DATA):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    new = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(HEADER)
        for label, session, pts in rows:
            w.writerow([label, session] + [f"{v:.5f}" for v in normalise(pts)])


def load(path=DATA):
    X, y, g = [], [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            y.append(row[0]); g.append(row[1]); X.append([float(v) for v in row[2:]])
    return np.array(X, dtype=np.float32), np.array(y), np.array(g)


def landmarker(image_mode=False):
    from core.hand_tracker import HandTrackerThread
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode as M
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
    path = HandTrackerThread.ensure_model()
    if not path:
        raise SystemExit("Hand model missing: put hand_landmarker.task in assets/models/ (needs internet once).")
    opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=path),
                                 running_mode=M.IMAGE if image_mode else M.VIDEO, num_hands=1)
    return mp, HandLandmarker.create_from_options(opts)
