"""Download two public GitHub landmark datasets and import them as separate sessions.

    python tools/fetch_public_data.py

Needs internet.  Both repos have no clear open data licence for the data itself
(one has none, one inherits an unlicensed image set): use them ONLY to test the
pipeline on your own PC, never ship a model trained on them without permission.
"""
import os, sys, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import import_csv
from _common import append_rows

SRC = {
 "srcA": "https://raw.githubusercontent.com/tukaitlin2/ASL-Sign-Language-Recognition/main/asl_data.csv",
 "srcB": "https://raw.githubusercontent.com/lmohammedaariff/Real-Time-ASL-Alphabet-Recognition-Using-Normalized-Hand-Landmarks/main/landmarks.csv",
}
here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(here, exist_ok=True)
for name, url in SRC.items():
    dst = os.path.join(here, name + ".csv")
    if not os.path.isfile(dst):
        print("downloading", url); urllib.request.urlretrieve(url, dst)
    rows = [(l, name, p) for l, p in import_csv.read(dst)]
    append_rows(rows); print(name, len(rows), "rows imported")
print("now run: python tools/train.py")
