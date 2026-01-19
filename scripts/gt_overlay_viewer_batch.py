# from __future__ import annotations

# import argparse
# import json
# from pathlib import Path
# from typing import Any, Dict, List, Optional, Tuple, Union

# import cv2
# import numpy as np


# # -----------------------------
# # JSONL loader (robust)
# # -----------------------------
# def load_jsonl_paths(jsonl_path: Union[str, Path]) -> List[str]:
#     jsonl_path = Path(jsonl_path)
#     paths: List[str] = []
#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, 1):
#             line = line.strip()
#             if not line:
#                 continue

#             try:
#                 obj = json.loads(line)
#             except json.JSONDecodeError:
#                 # raw path
#                 paths.append(line)
#                 continue

#             if isinstance(obj, str):
#                 paths.append(obj)
#             elif isinstance(obj, dict):
#                 for k in ["path", "img", "image", "img_path", "image_path", "relpath", "rel_path"]:
#                     v = obj.get(k)
#                     if isinstance(v, str) and v.strip():
#                         paths.append(v.strip().replace("\\", "/"))
#                         break
#                 else:
#                     raise ValueError(f"[line {line_no}] Cannot find path-like field in: {obj}")
#             else:
#                 raise ValueError(f"[line {line_no}] Unsupported item type: {type(obj)}")

#     if not paths:
#         raise ValueError(f"No paths found in {jsonl_path}")
#     return paths


# # -----------------------------
# # path mapping: images -> labels
# # -----------------------------
# def img_rel_to_label_rel(img_rel: str) -> str:
#     """
#     train/images/<scene>/<frame>.jpg -> train/labels/<scene>/<frame>.json
#     test/images/...도 동일 규칙
#     """
#     p = Path(normalize_rel(img_rel))
#     parts = list(p.parts)
#     if "images" not in parts:
#         raise ValueError(f"'images' not in path: {img_rel}")
#     idx = parts.index("images")
#     parts[idx] = "labels"
#     parts[-1] = Path(parts[-1]).with_suffix(".json").name
#     return str(Path(*parts))



# def normalize_rel(p: str) -> str:
#     return str(Path(p.replace("\\", "/")))

# def resolve_img_path(root: Path, rel_or_abs: str, exts: Tuple[str, ...] = ("jpg", "png")) -> Path:
#     """
#     jsonl path가 다음 중 뭐든 올 수 있다고 가정하고 최대한 찾아줌.
#     - test/images/.../xxxx.jpg
#     - images/.../xxxx.jpg
#     - test\\images\\...\\xxxx.jpg (windows sep)
#     - 확장자 없는 케이스
#     - root가 .../SDLane 인지 .../SDLane/train 인지 등 미스매치
#     """
#     p_raw = Path(rel_or_abs)
#     if p_raw.is_absolute():
#         if p_raw.exists():
#             return p_raw
#         raise FileNotFoundError(f"Absolute image path not found: {p_raw}")

#     rel = normalize_rel(rel_or_abs)
#     rel_path = Path(rel)

#     candidates: List[Path] = []

#     # 1) root / rel 그대로
#     candidates.append(root / rel_path)

#     # 2) root가 split(train/test/val)일 수도 있으니, rel의 선두 split을 제거한 버전도 시도
#     if len(rel_path.parts) >= 2 and rel_path.parts[0] in ("train", "test", "val", "valid", "validation"):
#         candidates.append(root / Path(*rel_path.parts[1:]))

#     # 3) rel에 images가 이미 들어가 있는데, root가 .../SDLane/<split>라면 rel 앞에 split이 빠졌을 수 있음
#     #    -> root.parent / rel 도 시도
#     candidates.append(root.parent / rel_path)

#     # 4) rel이 "images/..." 형태인데, 실제는 "<split>/images/..."일 수 있음
#     if len(rel_path.parts) >= 1 and rel_path.parts[0] == "images":
#         for split in ("train", "test", "val"):
#             candidates.append(root / split / rel_path)

#     # 5) rel이 "<split>/images/..."인데 root가 ".../SDLane/<split>"일 수 있음
#     #    -> root / images/... 시도
#     if len(rel_path.parts) >= 2 and rel_path.parts[1] == "images":
#         candidates.append(root / Path(*rel_path.parts[1:]))

#     # 확장자 보정
#     final_candidates: List[Path] = []
#     for c in candidates:
#         if c.suffix:
#             final_candidates.append(c)
#         else:
#             for ext in exts:
#                 final_candidates.append(c.with_suffix(f".{ext}"))

#     for c in final_candidates:
#         if c.exists():
#             return c

#     # 디버깅용: 어떤 후보를 시도했는지 에러에 포함
#     tried = "\n".join(str(c) for c in final_candidates[:12])
#     raise FileNotFoundError(f"Image not found: {rel_or_abs}\nTried:\n{tried}")



# # -----------------------------
# # label json -> mask (SDLane geometry supported)
# # -----------------------------
# def label_json_to_mask(label_path: Path, h: int, w: int, thickness: int) -> np.ndarray:
#     with open(label_path, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     lanes = data.get("geometry")
#     if lanes is None:
#         lanes = data.get("lanes") or data.get("Lane") or data.get("lane") or []

#     mask = np.zeros((h, w), dtype=np.uint8)

#     if not isinstance(lanes, list):
#         return mask

#     for lane in lanes:
#         if not isinstance(lane, list) or len(lane) < 2:
#             continue
#         pts = np.asarray(lane, dtype=np.float32)
#         if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
#             continue

#         pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
#         pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
#         pts = pts.astype(np.int32)

#         cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)

#     return mask


# def overlay_red(img_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
#     """Overlay mask region with red color (BGR: (0,0,255))."""
#     out = img_bgr.copy()
#     m = mask > 0
#     if not np.any(m):
#         return out
#     red = np.zeros_like(out)
#     red[..., 2] = 255
#     out[m] = (alpha * red[m] + (1 - alpha) * out[m]).astype(np.uint8)
#     return out


# def put_hud(img_bgr: np.ndarray, text_lines: List[str]) -> np.ndarray:
#     out = img_bgr.copy()
#     y = 26
#     for t in text_lines:
#         cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
#         cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
#         y += 26
#     return out


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--sdlane_root", type=str, required=True)
#     ap.add_argument("--jsonl", type=str, required=True)
#     ap.add_argument("--thickness", type=int, default=6)
#     ap.add_argument("--alpha", type=float, default=0.45)
#     ap.add_argument("--start", type=int, default=0)
#     ap.add_argument("--save_dir", type=str, default="./_gt_overlay_saves")
#     args = ap.parse_args()

#     root = Path(args.sdlane_root)
#     jsonl_path = Path(args.jsonl)
#     save_dir = Path(args.save_dir)
#     save_dir.mkdir(parents=True, exist_ok=True)

#     paths = load_jsonl_paths(jsonl_path)
#     n = len(paths)
#     idx = max(0, min(args.start, n - 1))

#     win = "GT Overlay Viewer (a/d or <-/->, g jump, s save, q quit)"
#     cv2.namedWindow(win, cv2.WINDOW_NORMAL)

#     def render(i: int) -> np.ndarray:
#         rel_or_abs = paths[i]
#         img_path = resolve_img_path(root, rel_or_abs)
#         # label mapping: need relative path w.r.t root
#         try:
#             rel = img_path.relative_to(root).as_posix()
#         except Exception:
#             # if absolute path is outside root, we can't map automatically
#             raise ValueError(
#                 f"Image is outside sdlane_root, cannot infer label path.\n"
#                 f"img_path={img_path}\n"
#                 f"sdlane_root={root}\n"
#                 f"Fix: store relative paths in jsonl or choose correct sdlane_root."
#             )

#         lbl_rel = img_rel_to_label_rel(rel)
#         lbl_path = root / lbl_rel

#         img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
#         if img_bgr is None:
#             raise RuntimeError(f"Failed to read image: {img_path}")
#         h, w = img_bgr.shape[:2]

#         if not lbl_path.exists():
#             raise FileNotFoundError(f"Label not found: {lbl_path}")

#         mask = label_json_to_mask(lbl_path, h, w, thickness=args.thickness)
#         vis = overlay_red(img_bgr, mask, alpha=args.alpha)

#         hud = [
#             f"[{i+1}/{n}]  mask_sum={int(mask.sum())}",
#             f"IMG: {rel}",
#             f"LBL: {lbl_rel}",
#         ]
#         vis = put_hud(vis, hud)
#         return vis

#     while True:
#         try:
#             vis = render(idx)
#         except Exception as e:
#             # show error frame
#             blank = np.zeros((720, 1280, 3), dtype=np.uint8)
#             msg = f"ERROR at index {idx}: {e}"
#             blank = put_hud(blank, [msg, "Press d to skip, a to go back, q to quit"])
#             vis = blank

#         cv2.imshow(win, vis)
#         key = cv2.waitKey(0) & 0xFF

#         # quit
#         if key in (27, ord("q")):
#             break

#         # next
#         if key in (ord("d"), 83):  # 'd' or right arrow (some env return 83/81 after waitKey)
#             idx = min(n - 1, idx + 1)
#             continue

#         # prev
#         if key in (ord("a"), 81):
#             idx = max(0, idx - 1)
#             continue

#         # jump
#         if key == ord("g"):
#             try:
#                 val = input(f"Jump to index (1~{n}): ").strip()
#                 j = int(val) - 1
#                 if 0 <= j < n:
#                     idx = j
#             except Exception:
#                 pass
#             continue

#         # save
#         if key == ord("s"):
#             out_path = save_dir / f"overlay_{idx+1:06d}.jpg"
#             cv2.imwrite(str(out_path), vis)
#             print(f"[saved] {out_path}")
#             continue

#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()



from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np


# -----------------------------
# JSONL loader (robust)
# -----------------------------
def load_jsonl_paths(jsonl_path: Union[str, Path]) -> List[str]:
    jsonl_path = Path(jsonl_path)
    paths: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # raw path
                paths.append(line.replace("\\", "/"))
                continue

            if isinstance(obj, str):
                paths.append(obj.replace("\\", "/"))
            elif isinstance(obj, dict):
                for k in ["path", "img", "image", "img_path", "image_path", "relpath", "rel_path"]:
                    v = obj.get(k)
                    if isinstance(v, str) and v.strip():
                        paths.append(v.strip().replace("\\", "/"))
                        break
                else:
                    raise ValueError(f"[line {line_no}] Cannot find path-like field in: {obj}")
            else:
                raise ValueError(f"[line {line_no}] Unsupported item type: {type(obj)}")

    if not paths:
        raise ValueError(f"No paths found in {jsonl_path}")
    return paths


# -----------------------------
# path mapping: images -> labels
# -----------------------------
def normalize_rel(p: str) -> str:
    return str(Path(p.replace("\\", "/")))


def img_rel_to_label_rel(img_rel: str) -> str:
    """
    train/images/<scene>/<frame>.jpg -> train/labels/<scene>/<frame>.json
    test/images/...도 동일 규칙
    """
    p = Path(normalize_rel(img_rel))
    parts = list(p.parts)
    if "images" not in parts:
        raise ValueError(f"'images' not in path: {img_rel}")
    idx = parts.index("images")
    parts[idx] = "labels"
    parts[-1] = Path(parts[-1]).with_suffix(".json").name
    return str(Path(*parts)).replace("\\", "/")


def resolve_img_path(root: Path, rel_or_abs: str, exts: Tuple[str, ...] = ("jpg", "png")) -> Path:
    """
    jsonl path가 다음 중 뭐든 올 수 있다고 가정하고 최대한 찾아줌.
    - test/images/.../xxxx.jpg
    - images/.../xxxx.jpg
    - test\\images\\...\\xxxx.jpg (windows sep)
    - 확장자 없는 케이스
    - root가 .../SDLane 인지 .../SDLane/train 인지 등 미스매치
    """
    p_raw = Path(rel_or_abs)
    if p_raw.is_absolute():
        if p_raw.exists():
            return p_raw
        raise FileNotFoundError(f"Absolute image path not found: {p_raw}")

    rel = normalize_rel(rel_or_abs)
    rel_path = Path(rel)

    candidates: List[Path] = []

    # 1) root / rel 그대로
    candidates.append(root / rel_path)

    # 2) root가 split(train/test/val)일 수도 있으니, rel의 선두 split을 제거한 버전도 시도
    if len(rel_path.parts) >= 2 and rel_path.parts[0] in ("train", "test", "val", "valid", "validation"):
        candidates.append(root / Path(*rel_path.parts[1:]))

    # 3) root.parent / rel 도 시도
    candidates.append(root.parent / rel_path)

    # 4) rel이 "images/..." 형태인데, 실제는 "<split>/images/..."일 수 있음
    if len(rel_path.parts) >= 1 and rel_path.parts[0] == "images":
        for split in ("train", "test", "val"):
            candidates.append(root / split / rel_path)

    # 5) rel이 "<split>/images/..."인데 root가 ".../SDLane/<split>"일 수 있음
    #    -> root / images/... 시도
    if len(rel_path.parts) >= 2 and rel_path.parts[1] == "images":
        candidates.append(root / Path(*rel_path.parts[1:]))

    # 확장자 보정
    final_candidates: List[Path] = []
    for c in candidates:
        if c.suffix:
            final_candidates.append(c)
        else:
            for ext in exts:
                final_candidates.append(c.with_suffix(f".{ext}"))

    for c in final_candidates:
        if c.exists():
            return c

    tried = "\n".join(str(c) for c in final_candidates[:12])
    raise FileNotFoundError(f"Image not found: {rel_or_abs}\nTried:\n{tried}")


# -----------------------------
# label json -> mask (SDLane geometry supported)
# -----------------------------
def label_json_to_mask(label_path: Path, h: int, w: int, thickness: int) -> np.ndarray:
    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lanes = data.get("geometry")
    if lanes is None:
        lanes = data.get("lanes") or data.get("Lane") or data.get("lane") or []

    mask = np.zeros((h, w), dtype=np.uint8)

    if not isinstance(lanes, list):
        return mask

    for lane in lanes:
        if not isinstance(lane, list) or len(lane) < 2:
            continue
        pts = np.asarray(lane, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 2:
            continue

        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        pts = pts.astype(np.int32)

        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)

    return mask


def overlay_red(img_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = img_bgr.copy()
    m = mask > 0
    if not np.any(m):
        return out
    red = np.zeros_like(out)
    red[..., 2] = 255
    out[m] = (alpha * red[m] + (1 - alpha) * out[m]).astype(np.uint8)
    return out


def put_hud(img_bgr: np.ndarray, text_lines: List[str]) -> np.ndarray:
    out = img_bgr.copy()
    y = 30
    for t in text_lines:
        cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 26
    return out


# -----------------------------
# TSV utilities (append + undo)
# -----------------------------
def ensure_tsv_header(tsv_path: Path, header: str) -> None:
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    if not tsv_path.exists():
        tsv_path.write_text(header + "\n", encoding="utf-8")


def read_tsv_lines(tsv_path: Path) -> List[str]:
    if not tsv_path.exists():
        return []
    return tsv_path.read_text(encoding="utf-8").splitlines()


def append_tsv(tsv_path: Path, line: str) -> None:
    # 항상 newline 포함해서 append
    with open(tsv_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")


def undo_last_entry(tsv_path: Path, last_line_exact: str) -> bool:
    """
    TSV에서 '마지막에 추가된 동일 라인' 1개를 제거.
    - 정확히 같은 문자열(탭 포함)이 마지막에 존재해야 제거.
    - 성공하면 True, 실패하면 False.
    """
    lines = read_tsv_lines(tsv_path)
    if not lines:
        return False

    # header가 첫 줄일 수 있음
    # 마지막부터 탐색해서 처음 만나는 exact line 1개 제거
    for i in range(len(lines) - 1, -1, -1):
        if lines[i] == last_line_exact:
            del lines[i]
            tmp = tsv_path.with_suffix(tsv_path.suffix + ".tmp")
            tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            tmp.replace(tsv_path)
            return True
    return False


def already_marked(tsv_path: Path, img_rel: str) -> bool:
    """
    중복 방지: TSV에 img_relpath가 이미 존재하면 True
    (header 제외, 첫 컬럼 비교)
    """
    if not tsv_path.exists():
        return False
    for line in read_tsv_lines(tsv_path):
        if not line or line.startswith("img_relpath"):
            continue
        first = line.split("\t", 1)[0]
        if first == img_rel:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdlane_root", type=str, required=True)
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--thickness", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default="./_gt_overlay_saves")
    ap.add_argument("--out_tsv", type=str, default="./_delete_candidates.tsv", help="tsv file to save delete candidates")
    ap.add_argument("--no_dedup", action="store_true", help="allow duplicate entries in TSV")
    args = ap.parse_args()

    root = Path(args.sdlane_root)
    jsonl_path = Path(args.jsonl)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = Path(args.out_tsv)
    ensure_tsv_header(out_tsv, header="img_relpath\tlabel_relpath")

    paths = load_jsonl_paths(jsonl_path)
    n = len(paths)
    idx = max(0, min(args.start, n - 1))

    win = "GT Overlay Viewer (a/d or <-/->, x mark, u undo, g jump, s save, q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # undo 스택: TSV에 실제로 쓴 "정확한 라인 문자열"을 저장
    undo_stack: List[str] = []

    def render(i: int) -> Tuple[np.ndarray, str, str]:
        rel_or_abs = paths[i]
        img_path = resolve_img_path(root, rel_or_abs)

        try:
            rel = img_path.relative_to(root).as_posix()
        except Exception:
            raise ValueError(
                f"Image is outside sdlane_root, cannot infer label path.\n"
                f"img_path={img_path}\n"
                f"sdlane_root={root}\n"
                f"Fix: store relative paths in jsonl or choose correct sdlane_root."
            )

        rel = rel.replace("\\", "/")
        lbl_rel = img_rel_to_label_rel(rel)
        lbl_path = root / lbl_rel

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        h, w = img_bgr.shape[:2]

        if not lbl_path.exists():
            raise FileNotFoundError(f"Label not found: {lbl_path}")

        mask = label_json_to_mask(lbl_path, h, w, thickness=args.thickness)
        vis = overlay_red(img_bgr, mask, alpha=args.alpha)

        marked = already_marked(out_tsv, rel) if not args.no_dedup else False

        hud = [
            f"[{i+1}/{n}]  mask_sum={int(mask.sum())}",
            f"IMG: {rel}",
            f"LBL: {lbl_rel}",
            f"TSV: {out_tsv}  (marked={marked})",
            "Keys: a/d prev/next | x mark | u undo | g jump | s save | q quit",
        ]
        vis = put_hud(vis, hud)
        return vis, rel, lbl_rel

    while True:
        try:
            vis, cur_img_rel, cur_lbl_rel = render(idx)
        except Exception as e:
            blank = np.zeros((720, 1280, 3), dtype=np.uint8)
            msg = f"ERROR at index {idx}: {e}"
            blank = put_hud(blank, [msg, "Press d to skip, a to go back, q to quit"])
            vis = blank
            cur_img_rel, cur_lbl_rel = "", ""

        cv2.imshow(win, vis)
        key = cv2.waitKey(0) & 0xFF

        # quit
        if key in (27, ord("q")):
            break

        # next
        if key in (ord("d"), 83):
            idx = min(n - 1, idx + 1)
            continue

        # prev
        if key in (ord("a"), 81):
            idx = max(0, idx - 1)
            continue

        # jump
        if key == ord("g"):
            try:
                val = input(f"Jump to index (1~{n}): ").strip()
                j = int(val) - 1
                if 0 <= j < n:
                    idx = j
            except Exception:
                pass
            continue

        # save overlay image
        if key == ord("s"):
            out_path = save_dir / f"overlay_{idx+1:06d}.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"[saved] {out_path}")
            continue

        # mark delete candidate -> append TSV
        if key == ord("x"):
            if not cur_img_rel:
                print("[warn] cannot mark: current frame not resolved.")
                continue

            line = f"{cur_img_rel}\t{cur_lbl_rel}"
            if (not args.no_dedup) and already_marked(out_tsv, cur_img_rel):
                print(f"[skip] already marked: {cur_img_rel}")
            else:
                append_tsv(out_tsv, line)
                undo_stack.append(line)
                print(f"[marked] {line} -> {out_tsv}")

            # mark 후 자동으로 다음으로 넘기고 싶으면 아래 줄 유지
            idx = min(n - 1, idx + 1)
            continue

        # undo last mark -> remove from TSV
        if key == ord("u"):
            if not undo_stack:
                print("[undo] nothing to undo.")
                continue
            last = undo_stack.pop()
            ok = undo_last_entry(out_tsv, last)
            if ok:
                print(f"[undo] removed: {last}")
            else:
                print(f"[undo] failed to remove (not found): {last}")
            continue

    cv2.destroyAllWindows()
    print(f"\nDone. TSV saved at: {out_tsv}")


if __name__ == "__main__":
    main()
