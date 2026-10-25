"""Upper-body tracker: MediaPipe Hand + Pose + Face landmarkers combined into the 543-point
Holistic layout the word model was trained on (face 468 | left hand 21 | pose 33 | right hand 21).

    t = UpperBodyTracker("assets/models")
    frame543 = t.track(rgb_image, timestamp_ms)       # (543,3) float32, NaN = not seen

Only x and y are used downstream. Each hand is given to the left or right slot by which pose wrist it is
nearest to (the hand model's own Left/Right label is mirrored depending on the camera, so it is only a fallback).
"""
from __future__ import annotations

import os
import numpy as np

from core.gs import landmarks as lm

URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
}


def missing_models(model_dir: str) -> list[str]:
    return [n for n in URLS if not os.path.isfile(os.path.join(model_dir, n))]


def try_download(model_dir: str) -> list[str]:
    """Best effort download of missing model files; returns the ones still missing."""
    import urllib.request
    os.makedirs(model_dir, exist_ok=True)
    for n in missing_models(model_dir):
        try:
            urllib.request.urlretrieve(URLS[n], os.path.join(model_dir, n))
        except Exception:
            pass
    return missing_models(model_dir)


class UpperBodyTracker:
    def __init__(self, model_dir: str = "assets/models"):
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python import vision as V
        self.mp = mp
        M = V.RunningMode.VIDEO
        b = lambda n: BaseOptions(model_asset_path=os.path.join(model_dir, n))
        self.hand = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
            base_options=b("hand_landmarker.task"), running_mode=M, num_hands=2,
            min_hand_detection_confidence=0.4, min_hand_presence_confidence=0.4, min_tracking_confidence=0.4))
        self.pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(
            base_options=b("pose_landmarker_full.task"), running_mode=M, num_poses=2))
        self.face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
            base_options=b("face_landmarker.task"), running_mode=M, num_faces=1))
        self.last_ts = -1
        self.people = 0

    def track(self, rgb: np.ndarray, ts_ms: int) -> np.ndarray:
        ts = max(int(ts_ms), self.last_ts + 1); self.last_ts = ts
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        out = np.full((lm.N_POINTS, 3), np.nan, np.float32)
        pr = self.pose.detect_for_video(img, ts)
        self.people = len(pr.pose_landmarks)
        if pr.pose_landmarks:
            p = pr.pose_landmarks[0]
            out[lm.POSE_START:lm.POSE_START + lm.N_POSE] = [(q.x, q.y, q.z) for q in p[:lm.N_POSE]]
        fr = self.face.detect_for_video(img, ts)
        if fr.face_landmarks:
            out[:lm.N_FACE] = [(q.x, q.y, q.z) for q in fr.face_landmarks[0][:lm.N_FACE]]
        hr = self.hand.detect_for_video(img, ts)
        wl = out[lm.pose(lm.POSE_LEFT_WRIST), :2]; wr = out[lm.pose(lm.POSE_RIGHT_WRIST), :2]
        have_pose = np.isfinite(wl).all() and np.isfinite(wr).all()
        used = set()
        hands = []
        for pts, cat in zip(hr.hand_landmarks, hr.handedness):
            arr = np.array([(q.x, q.y, q.z) for q in pts], np.float32)
            hands.append((arr, cat[0].category_name))
        # assign by nearest pose wrist (closest pair first)
        if have_pose and hands:
            cand = sorted(((float(np.hypot(*(h[0][0, :2] - w))), i, side) for i, h in enumerate(hands)
                           for side, w in (("L", wl), ("R", wr))))
            slots = {}
            for _, i, side in cand:
                if i in used or side in slots: continue
                slots[side] = i; used.add(i)
        else:
            slots = {}
            for i, (arr, name) in enumerate(hands):
                side = "R" if name == "Left" else "L"      # hand model labels assume a mirrored (selfie) image
                if side not in slots: slots[side] = i
        if "L" in slots: out[lm.LEFT_HAND_START:lm.LEFT_HAND_START + lm.N_HAND] = hands[slots["L"]][0]
        if "R" in slots: out[lm.RIGHT_HAND_START:lm.RIGHT_HAND_START + lm.N_HAND] = hands[slots["R"]][0]
        return out
