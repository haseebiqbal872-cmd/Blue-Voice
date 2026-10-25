"""Import an existing landmark CSV into data/landmarks.csv as its own session.

    python tools/import_csv.py <file.csv> <session_name>

Understands: 'label,x0,y0,z0,...,z20'  and  'f0,...,f62,label'.
Each source gets its own session, so train.py can measure accuracy on a source it
was NOT trained on (the honest test).  Check each dataset's licence first.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import append_rows

def read(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f); head = next(r)
        lab_first = head[0].lower() == "label"
        for row in r:
            if len(row) < 64: continue
            lab, vals = (row[0], row[1:64]) if lab_first else (row[63], row[:63])
            v = [float(x) for x in vals]
            rows.append((lab.strip().upper(), [tuple(v[i:i + 3]) for i in range(0, 63, 3)]))
    return rows

def main():
    path, session = sys.argv[1], sys.argv[2]
    rows = [(l, session, p) for l, p in read(path)]
    append_rows(rows); print("imported", len(rows), "rows as session", session)

if __name__ == "__main__":
    main()
