"""
Build the training folder layout for the expanded crop list.

Reads:
  --pv   PlantVillage "color" folder (contains one folder per class)
  --veg  Folder holding the Mendeley "Comprehensive Vegetable Leaf Disease
         Image Collection" (one .zip file or folder per class)
Writes:
  --out  <out>/<ClassName>/*.jpg  using the exact names in app/ml/labels.py,
         ready for:  python -m app.ml.dataset_utils --raw_dir <out> --processed_dir <processed>

Matching ignores case, spaces and punctuation, so small naming differences
between dataset versions are fine. Anything not matched is listed, never
silently dropped.

Datasets: PlantVillage (Hughes & Salathe) and the Mendeley Data collection
10.17632/rh8c5tgg5k.1 (CC BY 4.0) - credit the authors in your report.
"""
import argparse
import re
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


PV_MAP = {
    "tomatohealthy": "Tomato___Healthy",
    "tomatoearlyblight": "Tomato___Early_Blight",
    "tomatolateblight": "Tomato___Late_Blight",
    "tomatoleafmold": "Tomato___Leaf_Mold",
    "potatohealthy": "Potato___Healthy",
    "potatoearlyblight": "Potato___Early_Blight",
    "potatolateblight": "Potato___Late_Blight",
    "cornmaizehealthy": "Corn___Healthy",
    "cornmaizecommonrust": "Corn___Common_Rust",
    "cornmaizecercosporaleafspotgrayleafspot": "Corn___Gray_Leaf_Spot",
    "cornmaizegrayleafspot": "Corn___Gray_Leaf_Spot",
    "pepperbellhealthy": "Pepper___Healthy",
    "pepperbellbacterialspot": "Pepper___Bacterial_Spot",
    "squashpowderymildew": "Squash___Powdery_Mildew",
}

VEG_MAP = {
    "broccolihealthleaf": "Broccoli___Healthy",
    "broccolihealthyleaf": "Broccoli___Healthy",
    "broccolialternariablackspot": "Broccoli___Alternaria_Leaf_Spot",
    "broccolialternarialeafspot": "Broccoli___Alternaria_Leaf_Spot",
    "broccoliblackrot": "Broccoli___Black_Rot",
    "cabbagehealthyleaf": "Cabbage___Healthy",
    "cabbagealternarialeafspot": "Cabbage___Alternaria_Leaf_Spot",
    "cabbagegreymoulddisease": "Cabbage___Grey_Mould",
    "cabbagegreymould": "Cabbage___Grey_Mould",
    "cabbagetipburn": "Cabbage___Tip_Burn",
    "cabbagedamagedbywhitebutterfly": "Cabbage___Damaged_Leaf",
    "cabbagedamagedleaf": "Cabbage___Damaged_Leaf",
    "cauliflowerhealthyleaf": "Cauliflower___Healthy",
    "caulifloweralternarialeafspot": "Cauliflower___Alternaria_Leaf_Spot",
    "caulifloweralternariacurdspot": "Cauliflower___Alternaria_Leaf_Spot",
    "cauliflowerbacterialsoftrot": "Cauliflower___Bacterial_Soft_Rot",
    "caulifloweryellowvirus": "Cauliflower___Yellow_Virus",
}

EXPECTED = sorted(set(PV_MAP.values()) | set(VEG_MAP.values()))


def collect_sources(root: Path, mapping: dict, label: str):
    """Return {class_name: [source paths]} and a list of unmatched entries."""
    found = defaultdict(list)
    unmatched = []
    if not root.exists():
        print(f"[ERROR] {label} folder not found: {root}")
        return found, unmatched
    for entry in sorted(root.iterdir()):
        if entry.is_dir() or entry.suffix.lower() == ".zip":
            key = norm(entry.stem if entry.is_file() else entry.name)
            if key in mapping:
                found[mapping[key]].append(entry)
            else:
                unmatched.append(entry.name)
    return found, unmatched


def copy_source(src: Path, dst: Path, start: int) -> int:
    n = start
    if src.is_file():
        with zipfile.ZipFile(src) as zf:
            for member in zf.namelist():
                if "__MACOSX" in member or Path(member).suffix.lower() not in IMG_EXT:
                    continue
                n += 1
                with zf.open(member) as fin, open(dst / f"{n:05d}{Path(member).suffix.lower()}", "wb") as fout:
                    shutil.copyfileobj(fin, fout)
    else:
        for p in sorted(src.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                n += 1
                shutil.copy2(p, dst / f"{n:05d}{p.suffix.lower()}")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pv", required=True)
    ap.add_argument("--veg", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    sources = defaultdict(list)
    skipped = []
    for root, mapping, label in ((Path(args.pv), PV_MAP, "PlantVillage"), (Path(args.veg), VEG_MAP, "Vegetable collection")):
        found, unmatched = collect_sources(root, mapping, label)
        for cls, items in found.items():
            sources[cls].extend(items)
        skipped.extend(f"{label}: {name}" for name in unmatched)

    print(f"\nWriting class folders to {out}\n")
    summary = {}
    for cls in EXPECTED:
        if cls not in sources:
            continue
        dst = out / cls
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        n = 0
        for src in sources[cls]:
            n = copy_source(src, dst, n)
        summary[cls] = n

    print(f"{'Class':45s} images")
    for cls in EXPECTED:
        count = summary.get(cls)
        if count is None:
            print(f"{cls:45s} MISSING  <-- source folder not found")
        else:
            note = "  (small, expect weak results)" if count < 100 else ""
            print(f"{cls:45s} {count}{note}")

    if skipped:
        print("\nNot used (no matching class in labels.py):")
        for s in skipped:
            print("  -", s)
    missing = [c for c in EXPECTED if c not in summary]
    print(f"\nDone. {len(summary)}/{len(EXPECTED)} classes written." + (" Fix the MISSING ones before training." if missing else ""))


if __name__ == "__main__":
    main()