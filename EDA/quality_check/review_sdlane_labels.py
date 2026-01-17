from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np


IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class Sample:
    split: str
    img_path: Path
    json_path: Path


def read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def list_samples(data_root: Path, split: str) -> List[Sample]:
    """
    Expected structure:
      {root}/{split}/images/{hash_dir}/*.jpg
      {root}/{split}/labels/{hash_dir}/*.json
    Pairing by (hash_dir, stem)
    """
    img_root = data_root / split / "images"
    lbl_root = data_root / split / "labels"

    img_files: List[Path] = []
    json_files: List[Path] = []

    for d in img_root.glob("*"):
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() in IMG_EXTS:
                    img_files.append(p)

    for d in lbl_root.glob("*"):
        if d.is_dir():
            for p in d.iterdir():
                if p.is_file() and p.suffix.lower() == ".json":
                    json_files.append(p)

    def key_dir_stem(p: Path) -> Tuple[str, str]:
        return (p.parent.name, p.stem)

    img_map = {key_dir_stem(p): p for p in img_files}
    json_map = {key_dir_stem(p): p for p in json_files}

    samples: List[Sample] = []
    missing_json = 0
    for k, imgp in img_map.items():
        jp = json_map.get(k)
        if jp is None:
            missing_json += 1
            continue
        samples.append(Sample(split=split, img_path=imgp, json_path=jp))

    if missing_json > 0:
        print(f"[WARN] {split}: missing json for {missing_json} images (ignored).")

    samples.sort(key=lambda s: str(s.img_path))
    return samples


def load_samples_from_list(list_file: Path, data_root: Path) -> List[Sample]:
    """
    list_file TSV format (paths relative to data_root):
      train/images/.../xxx.jpg<TAB>train/labels/.../xxx.json
    """
    samples: List[Sample] = []
    lines = list_file.read_text(encoding="utf-8").splitlines()

    for ln, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        if "\t" not in line:
            raise ValueError(f"Invalid TSV format at line {ln}: tab not found")

        img_rel, json_rel = line.split("\t", 1)
        img_rel = img_rel.strip()
        json_rel = json_rel.strip()

        imgp = (data_root / img_rel).expanduser().resolve()
        jp = (data_root / json_rel).expanduser().resolve()

        split = "train" if img_rel.startswith("train/") else ("test" if img_rel.startswith("test/") else "unknown")

        if not imgp.exists():
            raise FileNotFoundError(f"Image not found (line {ln}): {imgp}")
        if not jp.exists():
            raise FileNotFoundError(f"JSON not found (line {ln}): {jp}")

        samples.append(Sample(split=split, img_path=imgp, json_path=jp))

    return samples


def extract_geometry_polylines(data: Any) -> List[np.ndarray]:
    """
    Based on your EDA:
      data["geometry"] is list of polylines
      each polyline is [[x,y], [x,y], ...]
    """
    out: List[np.ndarray] = []
    if not isinstance(data, dict) or "geometry" not in data:
        return out
    geom = data["geometry"]
    if not isinstance(geom, list):
        return out

    for g in geom:
        if isinstance(g, list) and len(g) >= 2 and isinstance(g[0], (list, tuple)) and len(g[0]) >= 2:
            try:
                pts = np.array([[float(pt[0]), float(pt[1])] for pt in g], dtype=np.float32)
                if pts.ndim == 2 and pts.shape[1] == 2:
                    out.append(pts)
            except Exception:
                continue
    return out


def detect_normalized(polylines: List[np.ndarray]) -> bool:
    if not polylines:
        return False
    pts = np.concatenate(polylines, axis=0)
    # heuristic: coordinates within 0~2 -> likely normalized
    return (pts[:, 0].max() <= 2.0) and (pts[:, 1].max() <= 2.0)


def draw_overlay(
    img_bgr: np.ndarray,
    polylines: List[np.ndarray],
    normalized: bool,
    max_lanes: int = 12,
    line_thickness: int = 2,
    point_radius: int = 2,
) -> np.ndarray:
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    for i, pts in enumerate(polylines[:max_lanes]):
        pts2 = pts.copy()
        if normalized:
            pts2[:, 0] *= w
            pts2[:, 1] *= h

        pts_int = np.round(pts2).astype(np.int32)

        for j in range(len(pts_int) - 1):
            x1, y1 = pts_int[j]
            x2, y2 = pts_int[j + 1]
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), line_thickness)

        for (x, y) in pts_int:
            cv2.circle(vis, (x, y), point_radius, (0, 0, 255), -1)

        x0, y0 = pts_int[0]
        cv2.putText(vis, f"{i}", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return vis


def load_progress(progress_path: Path) -> Dict[str, Any]:
    if progress_path.exists():
        try:
            return json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_progress(progress_path: Path, payload: Dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def relpath_str(p: Path, data_root: Path) -> str:
    try:
        return str(p.relative_to(data_root))
    except Exception:
        return str(p)


def append_jsonl_with_pos(path: Path, record: Dict[str, Any]) -> int:
    """
    Append a JSON line and return the byte offset *before* writing.
    This offset can be used to truncate the file on undo.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(record, ensure_ascii=False, default=str) + "\n").encode("utf-8")

    # binary mode to get exact byte offsets
    with path.open("ab") as f:
        pos = f.tell()
        f.write(line)
    return pos


def truncate_file_to_pos(path: Path, pos: int) -> None:
    """
    Truncate file to a previous byte offset.
    """
    if pos < 0:
        return
    if not path.exists():
        return
    with path.open("rb+") as f:
        f.seek(pos)
        f.truncate()


def copy_bad(sample: Sample, bad_dir: Path, data_root: Path) -> Tuple[Optional[str], Optional[str]]:
    """
    Copy bad samples preserving relative structure.
    Returns relative destination paths (img_rel_dst, json_rel_dst) for undo deletion.
    """
    img_rel = relpath_str(sample.img_path, data_root)
    json_rel = relpath_str(sample.json_path, data_root)

    dst_img = bad_dir / img_rel
    dst_json = bad_dir / json_rel

    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_json.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(sample.img_path, dst_img)
    shutil.copy2(sample.json_path, dst_json)

    return (str(dst_img), str(dst_json))


def try_remove_file(p: Optional[str]) -> None:
    if not p:
        return
    try:
        Path(p).unlink(missing_ok=True)  # py>=3.8
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="SDLane root path")
    ap.add_argument("--split", type=str, default="train", choices=["train", "test"])
    ap.add_argument(
        "--list_file",
        type=str,
        default=None,
        help="TSV list file: <img_rel>\\t<label_rel>, paths relative to data_root. If set, ignores --split scanning.",
    )
    ap.add_argument("--out_dir", type=str, default="./label_audit_out")
    ap.add_argument("--max_lanes", type=int, default=12)
    ap.add_argument("--line_thickness", type=int, default=2)
    ap.add_argument("--point_radius", type=int, default=2)
    ap.add_argument("--copy_bad", action="store_true", help="Copy bad samples into out_dir/bad_files (preserve structure)")
    ap.add_argument("--start", type=int, default=0, help="Start index (if not resuming)")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    good_jsonl = out_dir / "good.jsonl"
    bad_jsonl = out_dir / "bad.jsonl"
    skip_jsonl = out_dir / "skip.jsonl"
    progress_path = out_dir / "progress.json"
    bad_files_dir = out_dir / "bad_files"

    # load samples
    if args.list_file:
        list_file = Path(args.list_file).expanduser().resolve()
        samples = load_samples_from_list(list_file, data_root)
        source_desc = f"list_file={list_file}"
    else:
        samples = list_samples(data_root, args.split)
        source_desc = f"split={args.split}"

    if not samples:
        raise RuntimeError("No samples found. Check path/split/list_file structure.")

    # resume
    prog = load_progress(progress_path)
    idx = int(prog.get("index", args.start))
    # history items: {"index": int, "file": str, "pos": int, "copied": [img_dst, json_dst]}
    history: List[Dict[str, Any]] = prog.get("history", []) if isinstance(prog.get("history", []), list) else []

    print(f"Loaded {len(samples)} samples ({source_desc})")
    print("Keys: g=good, b=bad, s=skip, u=undo, q=quit (ESC also quits)")

    win = "SDLane Label Audit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while 0 <= idx < len(samples):
        s = samples[idx]

        img = cv2.imread(str(s.img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] failed to read image: {s.img_path}")
            idx += 1
            continue

        try:
            data = read_json(s.json_path)
        except Exception:
            data = {}

        polylines = extract_geometry_polylines(data)
        normalized = detect_normalized(polylines)

        vis = draw_overlay(
            img,
            polylines,
            normalized,
            max_lanes=args.max_lanes,
            line_thickness=args.line_thickness,
            point_radius=args.point_radius,
        )

        h, w = vis.shape[:2]
        rel_img = relpath_str(s.img_path, data_root)
        header = f"[{idx+1}/{len(samples)}] lanes={len(polylines)} norm={bool(normalized)} | {rel_img}"
        cv2.putText(vis, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        footer = "g:good  b:bad  s:skip  u:undo  q/ESC:quit"
        cv2.putText(vis, footer, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, vis)
        key = cv2.waitKey(0) & 0xFF

        def make_record(status: str) -> Dict[str, Any]:
            return {
                "status": str(status),
                "split": str(s.split),
                "image": relpath_str(s.img_path, data_root),
                "label": relpath_str(s.json_path, data_root),
                "num_polylines": int(len(polylines)),
                "normalized_detected": bool(normalized),
            }

        # quit
        if key in (ord("q"), 27):  # q or ESC
            print("Quit. Saving progress...")
            save_progress(progress_path, {"index": idx, "history": history})
            break

        # undo (REAL delete from jsonl by truncation)
        if key == ord("u"):
            if not history:
                print("[INFO] nothing to undo")
                continue

            last = history.pop()
            prev_idx = int(last["index"])
            file_path = Path(last["file"])
            pos = int(last["pos"])
            copied = last.get("copied", [])

            # truncate the jsonl back
            try:
                truncate_file_to_pos(file_path, pos)
                # remove copied bad files if any
                if isinstance(copied, list) and copied:
                    for p in copied:
                        try_remove_file(p)
                idx = prev_idx
                print(f"[UNDO] reverted last action. back to index {idx}")
            except Exception as e:
                print(f"[UNDO-ERROR] failed to undo properly: {e}")

            save_progress(progress_path, {"index": idx, "history": history})
            continue

        # handle decisions
        if key == ord("g"):
            rec = make_record("good")
            pos = append_jsonl_with_pos(good_jsonl, rec)
            history.append({"index": idx, "file": str(good_jsonl), "pos": pos, "copied": []})
            idx += 1

        elif key == ord("b"):
            rec = make_record("bad")
            pos = append_jsonl_with_pos(bad_jsonl, rec)
            copied_paths: List[str] = []
            if args.copy_bad:
                dst_img, dst_json = copy_bad(s, bad_files_dir, data_root)
                if dst_img:
                    copied_paths.append(dst_img)
                if dst_json:
                    copied_paths.append(dst_json)
            history.append({"index": idx, "file": str(bad_jsonl), "pos": pos, "copied": copied_paths})
            idx += 1

        elif key == ord("s"):
            rec = make_record("skip")
            pos = append_jsonl_with_pos(skip_jsonl, rec)
            history.append({"index": idx, "file": str(skip_jsonl), "pos": pos, "copied": []})
            idx += 1

        else:
            print("[INFO] unknown key. Use g/b/s/u/q.")
            continue

        # periodic save
        if idx % 20 == 0:
            save_progress(progress_path, {"index": idx, "history": history})

    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
