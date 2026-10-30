"""Turn a labels file into train/val/test JSONL.

Input: TSV with columns  image_path <TAB> text <TAB> group
`group` is the writer or document id. Splits are made by group so the
validation/test sets measure generalisation to unseen writers.
"""
import argparse
import csv
import json
import random
from pathlib import Path

from PIL import Image

from common import normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="TSV: image_path, text, group")
    ap.add_argument("--image-root", default=".", help="prefix for relative image paths")
    ap.add_argument("--out", default="data")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.image_root)
    by_group: dict[str, list[dict]] = {}
    skipped = 0
    with open(args.labels, encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) < 3:
                skipped += 1
                continue
            path, text, group = row[0], normalize(row[1]), row[2]
            full = (root / path).resolve()
            if not text or not full.exists():
                skipped += 1
                continue
            try:
                with Image.open(full) as im:
                    im.verify()
            except Exception:
                skipped += 1
                continue
            by_group.setdefault(group, []).append({"image": str(full), "text": text})

    groups = sorted(by_group)
    random.Random(args.seed).shuffle(groups)
    n_test = max(1, round(len(groups) * args.test_frac))
    n_val = max(1, round(len(groups) * args.val_frac))
    split = {
        "test": groups[:n_test],
        "val": groups[n_test:n_test + n_val],
        "train": groups[n_test + n_val:],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, gs in split.items():
        rows = [r for g in gs for r in by_group[g]]
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} samples from {len(gs)} groups")
    print(f"skipped {skipped} bad rows")


if __name__ == "__main__":
    main()
