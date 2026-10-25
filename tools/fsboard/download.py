"""Download a small slice of FSboard (CC BY 4.0) with the Kaggle CLI.

    python tools/fsboard/download.py [--train 3] [--val 1] [--test 1] [--study daun_v3]

The whole dataset is ~1.45 TB; the landmark shards you need are ~0.6 GB each, so the default
(3 train + 1 val + 1 test) is about 3 GB.  More train shards = better accuracy.
You need a free Kaggle account and an API token:  kaggle.com > Settings > API > Create token,
then save the downloaded kaggle.json to  C:\\Users\\<you>\\.kaggle\\kaggle.json
Licence: CC BY 4.0 - credit "FSboard (Georg et al. 2024, Google)".  Do not identify the signers.
"""
import argparse, os, shutil, subprocess, sys, zipfile

SHARDS = {"train": 100, "val": 20, "test": 20}      # daun_v3 (checked on the Kaggle page listing)


def have_credentials():
    home = os.path.expanduser("~")
    return (os.path.isfile(os.path.join(home, ".kaggle", "kaggle.json"))
            or os.path.isfile(os.path.join(home, ".kaggle", "access_token"))
            or os.environ.get("KAGGLE_API_TOKEN") or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")))


def kaggle_exe():
    exe = shutil.which("kaggle")
    if exe: return [exe]
    cand = os.path.join(os.path.dirname(sys.executable), "kaggle.exe" if os.name == "nt" else "kaggle")
    return [cand] if os.path.isfile(cand) else [sys.executable, "-m", "kaggle.cli"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=3); ap.add_argument("--val", type=int, default=1); ap.add_argument("--test", type=int, default=1)
    ap.add_argument("--study", default="daun_v3"); ap.add_argument("--out", default="data/fsboard")
    a = ap.parse_args()
    if not have_credentials():
        raise SystemExit("Kaggle token not found. Create one at kaggle.com > Settings > API > Create New Token and save "
                         "kaggle.json to " + os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json"))
    os.makedirs(a.out, exist_ok=True)
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        total = SHARDS[split]
        for k in range(n):
            fn = "validation" if split == "val" else split
            name = f"{a.study}/landmarks/{a.study}-{fn}.arrow-{k:05d}-of-{total:05d}"
            dst = os.path.join(a.out, name.replace("/", os.sep))
            if os.path.isfile(dst):
                print("have", name); continue
            print("downloading", name, flush=True)
            r = subprocess.run(kaggle_exe() + ["datasets", "download", "googleai/fsboard", "-f", name, "-p", a.out], capture_output=True, text=True)
            print(r.stdout.strip(), r.stderr.strip())
            # Kaggle may deliver a .zip (file stored flat or with its folders)
            for z in [os.path.join(a.out, f) for f in os.listdir(a.out) if f.endswith(".zip")]:
                with zipfile.ZipFile(z) as zf: zf.extractall(os.path.dirname(dst) if not any("/" in n for n in zf.namelist()) else a.out)
                os.remove(z)
            if not os.path.isfile(dst):
                found = [os.path.join(dp, f) for dp, _, fs in os.walk(a.out) for f in fs if f == os.path.basename(name)]
                if found:
                    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(found[0], dst)
            if not os.path.isfile(dst):
                raise SystemExit(f"Download failed for {name}. See the message above.")
    print("done ->", a.out)


if __name__ == "__main__":
    main()
