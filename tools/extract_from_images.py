"""Turn a folder-per-label image dataset into landmark rows.

    python tools/extract_from_images.py <dataset_dir> [--per-label 400]

<dataset_dir>/<LABEL>/*.jpg|png  (e.g. A/, B/, ... 0/, 1/ ...).
Check the dataset's LICENCE allows your use before you train on it.
Images where no hand is found are skipped and counted.
"""
import sys, os, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import append_rows, landmarker
import cv2

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--per-label", type=int, default=400)
    a = ap.parse_args()
    mp, det = landmarker(image_mode=True)
    for lab in sorted(os.listdir(a.root)):
        d = os.path.join(a.root, lab)
        if not os.path.isdir(d): continue
        files = sorted(glob.glob(os.path.join(d, "*.*")))[: a.per_label]
        rows, miss = [], 0
        for i, fpath in enumerate(files):
            img = cv2.imread(fpath)
            if img is None: continue
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            if not res.hand_landmarks: miss += 1; continue
            # session = file group, so evaluation can hold whole groups out
            rows.append((lab.upper(), f"img{i % 5}", [(l.x, l.y, l.z) for l in res.hand_landmarks[0]]))
        append_rows(rows); print(lab, len(rows), "ok", miss, "no hand")

if __name__ == "__main__":
    main()
