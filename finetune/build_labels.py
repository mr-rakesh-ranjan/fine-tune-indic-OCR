"""Build labels.tsv (line crop <TAB> text <TAB> doc id) from the BN-HTR dataset.

BN-HTR layout used here:
  Segmentation_Images/Words/<doc>/<doc>_<page>/<doc>_<page>_<line>.jpg   one crop per handwritten line
  Segmentation_Images/Words/<doc>/<doc>_<page>/<doc>_<page>_<line>.txt   YOLO boxes, one per word
  Recognition_Ground_Truth_Texts/<doc>/<doc>.xlsx                        Sheet1: Id=<doc>_<page>_<line>_<word>, Word

The .txt next to each doc is the typed source text and does not line up with the handwritten
lines, so line labels are rebuilt by joining the per-word transcriptions from the xlsx.
Only the stdlib is needed to read the xlsx.

The sheets contain typos, so ids are parsed leniently ("78_6_1 _1", "8_5_!8_10") and a line is
dropped when its text can't be trusted: duplicate word ids, or a word count that disagrees with
the number of word boxes. A sheet that mostly names another doc (84 is a copy of 8) is skipped.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET
import zipfile

from common import normalize

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MAX_PREFIX_MISMATCH = 0.05


def read_words(xlsx: Path) -> list[tuple[str, str]]:
    """Return (Id, Word) pairs from Sheet1."""
    with zipfile.ZipFile(xlsx) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")) for si in root.findall("m:si", NS)]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    pairs = []
    for row in sheet.iter(f"{{{NS['m']}}}row"):
        vals = {}
        for c in row.findall("m:c", NS):
            v = c.find("m:v", NS)
            if v is None or v.text is None:
                continue
            col = re.match(r"[A-Z]+", c.get("r")).group()
            vals[col] = shared[int(v.text)] if c.get("t") == "s" else v.text
        if "A" in vals and "B" in vals:
            pairs.append((vals["A"], vals["B"]))
    return pairs


def parse_id(wid: str):
    nums = re.findall(r"\d+", wid)
    return tuple(map(int, nums)) if len(nums) == 4 else None


def box_count(txt: Path) -> int:
    return sum(1 for l in txt.read_text().splitlines() if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="path to BN-HTR_Dataset")
    ap.add_argument("--out", default="labels.tsv")
    ap.add_argument("--keep-count-mismatch", action="store_true", help="keep lines whose word count != box count")
    args = ap.parse_args()

    root = Path(args.dataset)
    gt_dir = root / "Recognition_Ground_Truth_Texts"
    words_dir = root / "Segmentation_Images" / "Words"

    stats = defaultdict(int)
    with open(args.out, "w", encoding="utf-8", newline="") as out:
        for doc_dir in sorted(gt_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
            doc = doc_dir.name
            xlsx = next(iter(doc_dir.glob("*.xlsx")), None)  # doc 60 is named 60.xl.xlsx
            if xlsx is None:
                print(f"doc {doc}: no xlsx, skipped")
                continue

            entries = []
            for wid, word in read_words(xlsx):
                ids = parse_id(wid)
                if ids:
                    entries.append((ids, word))
            if not entries:
                continue
            mismatch = sum(ids[0] != int(doc) for ids, _ in entries) / len(entries)
            if mismatch > MAX_PREFIX_MISMATCH:
                print(f"doc {doc}: {mismatch:.0%} of ids name another doc, skipped")
                stats["docs_skipped"] += 1
                continue

            lines = defaultdict(list)  # (page, line) -> [(word_idx, word)]
            for (_, page, line, w), word in entries:
                lines[(page, line)].append((w, word))

            for (page, line), ws in sorted(lines.items()):
                stem = words_dir / doc / f"{doc}_{page}" / f"{doc}_{page}_{line}"
                if not stem.with_suffix(".jpg").exists():
                    stats["no_crop"] += 1
                    continue
                idx = [i for i, _ in ws]
                if len(set(idx)) != len(idx):
                    stats["duplicate_ids"] += 1
                    continue
                text = normalize(" ".join(w for _, w in sorted(ws)))
                if not text:
                    stats["empty"] += 1
                    continue
                boxes = stem.with_suffix(".txt")
                if not args.keep_count_mismatch and boxes.exists() and box_count(boxes) != len(ws):
                    stats["count_mismatch"] += 1
                    continue
                rel = stem.with_suffix(".jpg").relative_to(root)
                out.write(f"{rel.as_posix()}\t{text}\t{doc}\n")
                stats["kept"] += 1

    print(dict(stats))
    print(f"wrote {stats['kept']} lines to {args.out}")


if __name__ == "__main__":
    main()
