from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def list_pairs(data_root: Path) -> List[Tuple[str, str]]:
    """
    Save paths relative to data_root:
      train/images/...
      train/labels/...
    """
    pairs: List[Tuple[str, str]] = []

    for split in ["train", "test"]:
        img_root = data_root / split / "images"
        lbl_root = data_root / split / "labels"

        img_files = []
        json_files = []

        for d in img_root.glob("*"):
            if d.is_dir():
                img_files.extend([p for p in d.iterdir()
                                  if p.is_file() and p.suffix.lower() in IMG_EXTS])

        for d in lbl_root.glob("*"):
            if d.is_dir():
                json_files.extend([p for p in d.iterdir()
                                   if p.is_file() and p.suffix.lower() == ".json"])

        def key(p: Path):
            return (split, p.parent.name, p.stem)

        img_map = {key(p): p for p in img_files}
        json_map = {key(p): p for p in json_files}

        for k, imgp in img_map.items():
            jp = json_map.get(k)
            if jp is None:
                continue

            # 🔑 relative path (train/... or test/...)
            img_rel = imgp.relative_to(data_root)
            json_rel = jp.relative_to(data_root)

            pairs.append((str(img_rel), str(json_rel)))

    # stable ordering
    pairs.sort(key=lambda x: x[0])
    return pairs


def chunk_indices(n: int, k: int) -> List[tuple[int, int]]:
    """
    Split n items into k chunks as evenly as possible.
    """
    base = n // k
    rem = n % k
    out = []
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        out.append((start, end))
        start = end
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, type=str)
    ap.add_argument("--out_dir", default="./review_lists", type=str)
    ap.add_argument("--workers", default=4, type=int)
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_pairs(data_root)
    n = len(pairs)
    print("Total pairs:", n)

    # save master
    master = out_dir / "all_pairs.tsv"
    with master.open("w", encoding="utf-8") as f:
        for imgp, jp in pairs:
            f.write(f"{imgp}\t{jp}\n")
    print("Saved:", master)

    # split
    ranges = chunk_indices(n, args.workers)
    for i, (a, b) in enumerate(ranges, start=1):
        p = out_dir / f"worker_{i}.tsv"
        with p.open("w", encoding="utf-8") as f:
            for imgp, jp in pairs[a:b]:
                f.write(f"{imgp}\t{jp}\n")
        print(f"Saved: {p}  ({b-a} items)  range=[{a},{b})")


if __name__ == "__main__":
    main()
