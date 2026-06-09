import json
import os
import re
from pathlib import Path
from multiprocessing import get_context


# =========================
# User-configurable paths
# =========================
YES_JSON_GLOB = "data/data00/object365_patch*_yes.json"
ANN_JSON = "data/data00/object365/Objects365/data/train/zhiyuan_objv2_train.json"
OUT_DIR = "data/data00/obj365_yes_ann_train"
SPLIT_NAME = "train"

# Matching strategy: "auto" | "basename" | "file_name"
MATCH_MODE = "auto"

# CPU parallelism
N_WORKERS = max(1, (os.cpu_count() or 1))  # use all CPU cores by default


# -------------------------
# Helpers
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def load_json_list(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_glob(glob_str: str):
    """Expand an absolute glob path robustly."""
    if any(ch in glob_str for ch in "*?[]"):
        p = Path(glob_str)
        parent = p.parent if str(p.parent) != "" else Path(".")
        pattern = p.name
        return sorted(str(x) for x in parent.glob(pattern))
    return [glob_str]


def xyxy2xywhn_clip(x1, y1, x2, y2, img_w, img_h):
    """Convert pixel xyxy to normalized xywh (center-based) with clipping."""
    x1 = max(0.0, min(float(x1), float(img_w)))
    x2 = max(0.0, min(float(x2), float(img_w)))
    y1 = max(0.0, min(float(y1), float(img_h)))
    y2 = max(0.0, min(float(y2), float(img_h)))

    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    if img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0, 0.0
    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def extract_patch_hint(path_str: str):
    """Try to extract 'patchXX' token from a path for disambiguation."""
    m = re.search(r"(patch\d+)", path_str, flags=re.IGNORECASE)
    return m.group(1) if m else ""


def split_even(items, n_parts: int):
    """Split a list into n nearly-equal contiguous chunks."""
    n = len(items)
    if n_parts <= 1 or n == 0:
        return [items]
    parts = []
    for i in range(n_parts):
        a = (n * i) // n_parts
        b = (n * (i + 1)) // n_parts
        if a < b:
            parts.append(items[a:b])
    return parts


# -------------------------
# Global data for workers (fork-friendly)
# -------------------------
CAT_ID_TO_CID = None
CID_TO_NAME = None
ANN_MAP = None
LABELS_DIR = None
SPLIT = None


def init_worker(cat_id_to_cid, cid_to_name, ann_map, labels_dir, split_name):
    """Initializer to set global readonly references in each worker."""
    global CAT_ID_TO_CID, CID_TO_NAME, ANN_MAP, LABELS_DIR, SPLIT
    CAT_ID_TO_CID = cat_id_to_cid
    CID_TO_NAME = cid_to_name
    ANN_MAP = ann_map
    LABELS_DIR = labels_dir
    SPLIT = split_name


def worker_process(args):
    """
    Worker writes:
      - labels_yolo/<stem>.txt for each image
      - manifest_part_{worker_id}.jsonl
    """
    worker_id, records, out_dir_str = args
    out_dir = Path(out_dir_str)
    part_path = out_dir / f"manifest_part_{worker_id:03d}.jsonl"

    found = 0
    with open(part_path, "w", encoding="utf-8") as mf:
        for rec in records:
            # rec fields prepared by main: image_path, coco_file_name, image_id, width, height, match_how
            img_path = rec["image_path"]
            coco_file_name = rec["coco_file_name"]
            img_id = rec["image_id"]
            width = float(rec["width"])
            height = float(rec["height"])
            match_how = rec["match_how"]

            anns = ANN_MAP.get(img_id, [])
            yolo_lines = []
            labels = []

            for a in anns:
                cat_id = a.get("category_id", None)
                if cat_id is None or cat_id not in CAT_ID_TO_CID:
                    continue

                cid = CAT_ID_TO_CID[cat_id]
                x, y, w, h = a["bbox"]  # COCO bbox: xywh in pixels, top-left

                x1, y1, x2, y2 = x, y, x + w, y + h
                cxn, cyn, wn, hn = xyxy2xywhn_clip(x1, y1, x2, y2, width, height)

                yolo_lines.append(f"{cid} {cxn:.5f} {cyn:.5f} {wn:.5f} {hn:.5f}")
                labels.append({
                    "cid": int(cid),
                    "name": CID_TO_NAME.get(cid, str(cid)),
                    "xywhn": [cxn, cyn, wn, hn],
                    "area": float(wn * hn),
                })

            # Write YOLO label file (one file per image; safe without locks)
            stem = Path(coco_file_name if coco_file_name else img_path).stem
            txt_path = Path(LABELS_DIR) / f"{stem}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                if yolo_lines:
                    f.write("\n".join(yolo_lines) + "\n")

            out_rec = {
                "split": SPLIT,
                "image_path": img_path,
                "coco_file_name": coco_file_name,
                "image_id": img_id,
                "width": int(width),
                "height": int(height),
                "match_how": match_how,
                "labels": labels,
            }
            mf.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            found += 1

    return {"worker_id": worker_id, "count": found, "part_path": str(part_path)}


def main():
    ensure_dir(Path(OUT_DIR))
    labels_dir = Path(OUT_DIR) / "labels_bbox"
    ensure_dir(labels_dir)

    # 1) Load yes_json list
    yes_json_files = expand_glob(YES_JSON_GLOB)
    if not yes_json_files:
        raise FileNotFoundError(f"No yes_json matched: {YES_JSON_GLOB}")

    yes_img_paths = []
    for yp in yes_json_files:
        lst = load_json_list(yp)
        if not isinstance(lst, list):
            raise ValueError(f"{yp} is not a JSON list")
        yes_img_paths.extend(lst)

    yes_img_paths = sorted(set(yes_img_paths))
    print(f"[INFO] split={SPLIT_NAME}")
    print(f"[INFO] yes_json files: {len(yes_json_files)}")
    print(f"[INFO] unique yes images: {len(yes_img_paths)}")

    # 2) Load COCO annotation json as dict (fast + fork-friendly)
    print(f"[INFO] loading COCO json: {ANN_JSON}")
    with open(ANN_JSON, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = coco.get("categories", [])
    images_list = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # Ultralytics-style: cat_id -> contiguous cid (0..N-1), using categories order
    cat_id_to_cid = {c["id"]: i for i, c in enumerate(categories)}
    cid_to_name = {i: c.get("name", str(i)) for i, c in enumerate(categories)}

    # 3) Build partial image index only for needed basenames
    need_basenames = set(Path(p).name for p in yes_img_paths)
    basename_to_imgs = {}
    file_name_to_img = {}

    print(f"[INFO] COCO images: {len(images_list)}. Building partial image index ...")
    for im in images_list:
        fn = im.get("file_name", "")
        bn = Path(fn).name
        if bn in need_basenames:
            basename_to_imgs.setdefault(bn, []).append(im)
            file_name_to_img[fn] = im
    print(f"[INFO] indexed basenames: {len(basename_to_imgs)}")

    # 4) Resolve yes images to COCO image entries (collect needed image_ids)
    def resolve_image(img_path: str):
        p = Path(img_path)
        bn = p.name
        hint = extract_patch_hint(img_path)

        if MATCH_MODE == "file_name":
            # Try suffix match on indexed file_names
            sp = str(p).replace("\\", "/")
            for fn, im in file_name_to_img.items():
                if fn and sp.endswith(fn.replace("\\", "/")):
                    return im, "file_name_suffix"
            return None, "not_found"

        ims = basename_to_imgs.get(bn, [])

        if MATCH_MODE == "basename":
            if len(ims) == 1:
                return ims[0], "basename_unique"
            if len(ims) > 1:
                if hint:
                    cand = [im for im in ims if hint.lower() in im.get("file_name", "").lower()]
                    if len(cand) == 1:
                        return cand[0], "basename_patch_disambiguated"
                return ims[0], "basename_ambiguous"
            return None, "not_found"

        # auto
        if len(ims) == 1:
            return ims[0], "basename_unique"
        if len(ims) > 1:
            if hint:
                cand = [im for im in ims if hint.lower() in im.get("file_name", "").lower()]
                if len(cand) == 1:
                    return cand[0], "basename_patch_disambiguated"
            sp = str(p).replace("\\", "/")
            for im in ims:
                fn = im.get("file_name", "")
                if fn and sp.endswith(fn.replace("\\", "/")):
                    return im, "file_name_suffix_from_ambiguous"
            return ims[0], "ambiguous_fallback"

        # No basename hit: try suffix match
        sp = str(p).replace("\\", "/")
        for fn, im in file_name_to_img.items():
            if fn and sp.endswith(fn.replace("\\", "/")):
                return im, "file_name_suffix"
        return None, "not_found"

    resolved_records = []
    missing = 0
    ambiguous = 0
    needed_image_ids = set()

    for img_path in yes_img_paths:
        im, how = resolve_image(img_path)
        if im is None:
            missing += 1
            continue
        if "ambiguous" in how:
            ambiguous += 1

        img_id = im["id"]
        needed_image_ids.add(img_id)

        resolved_records.append({
            "image_path": img_path,
            "coco_file_name": im.get("file_name", ""),
            "image_id": img_id,
            "width": im.get("width", 0),
            "height": im.get("height", 0),
            "match_how": how,
        })

    print(f"[INFO] resolved: {len(resolved_records)}  missing: {missing}  ambiguous: {ambiguous}")
    print(f"[INFO] building ann_map for needed images: {len(needed_image_ids)}")

    # 5) Build ann_map only for needed image_ids (single pass over all annotations)
    ann_map = {}
    for a in annotations:
        img_id = a.get("image_id", None)
        if img_id in needed_image_ids:
            ann_map.setdefault(img_id, []).append(a)

    # 6) Parallel processing: evenly split records across CPU workers
    n_workers = min(N_WORKERS, max(1, len(resolved_records)))
    chunks = split_even(resolved_records, n_workers)
    print(f"[INFO] CPU workers: {n_workers}  chunks: {len(chunks)}")

    # Use "fork" to share large read-only dicts efficiently on Linux
    ctx = get_context("fork") if os.name != "nt" else get_context("spawn")

    with ctx.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(cat_id_to_cid, cid_to_name, ann_map, str(labels_dir), SPLIT_NAME),
    ) as pool:
        tasks = [(i, chunks[i], OUT_DIR) for i in range(len(chunks))]
        results = pool.map(worker_process, tasks)

    total = sum(r["count"] for r in results)
    print(f"[INFO] workers done. total processed: {total}")

    # 7) Merge manifest parts into a single manifest.jsonl
    merged_manifest = Path(OUT_DIR) / "manifest.jsonl"
    with open(merged_manifest, "w", encoding="utf-8") as out:
        for r in sorted(results, key=lambda x: x["worker_id"]):
            part_path = Path(r["part_path"])
            if not part_path.exists():
                continue
            with open(part_path, "r", encoding="utf-8") as pf:
                for line in pf:
                    out.write(line)

    print(f"[DONE] labels_yolo: {labels_dir}")
    print(f"[DONE] manifest: {merged_manifest}")


if __name__ == "__main__":
    main()