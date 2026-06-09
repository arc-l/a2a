import json
from pathlib import Path

# =========================
# Fixed config here (no argparse)
# =========================
OUT_ROOT = Path("data/data00")
MANIFEST_JSONL = Path("data/data00/obj365_yes_ann_train/manifest.jsonl")

# Naming rule for round-1 output files (the round-1 script you provided)
ROUND1_PATTERN = "object365_patch*_round1_aff_select.jsonl"

# Merged output, to avoid overwriting due to name collisions
MERGED_SUFFIX = "_merged_v1"  # Produces: object365_patchX_round1_aff_select_merged_v1.jsonl

# bbox injection format precision (determines the look of the "name + coordinates" string; the downstream VLM must reproduce it verbatim)
COORD_DECIMALS = 2

# =========================
# Load the manifest index (one-time)
# =========================
def load_manifest_index(manifest_jsonl: Path):
    idx = {}
    bad = 0
    total = 0
    with manifest_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
                ip = rec.get("image_path", "")
                if not ip:
                    bad += 1
                    continue
                # Here labels are what you wrote in the data-processing script: [{cid,name,xywhn,area}, ...]
                idx[ip] = rec
            except Exception:
                bad += 1
    print(f"[INFO] manifest loaded: lines={total}, indexed={len(idx)}, bad_lines={bad}")
    return idx

# =========================
# Geometry conversion
# =========================
def xywhn_to_xyxy(xywhn, W: int, H: int):
    cx, cy, w, h = [float(x) for x in xywhn]
    x1 = (cx - w / 2.0) * W
    y1 = (cy - h / 2.0) * H
    x2 = (cx + w / 2.0) * W
    y2 = (cy + h / 2.0) * H
    # clip
    x1 = max(0.0, min(x1, float(W)))
    x2 = max(0.0, min(x2, float(W)))
    y1 = max(0.0, min(y1, float(H)))
    y2 = max(0.0, min(y2, float(H)))
    return [x1, y1, x2, y2]

def fmt_xyxy(xyxy, nd=2):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    s = f"{{:.{nd}f}}"
    return f"[{s.format(x1)},{s.format(y1)},{s.format(x2)},{s.format(y2)}]"

# =========================
# For a given object_name, collect the bboxes of all same-named instances in the image, and generate the unique "name + coordinates" identifier string
# =========================
def collect_object_instances(manifest_rec: dict, object_name: str, nd=2):
    W = int(manifest_rec.get("width", 0) or 0)
    H = int(manifest_rec.get("height", 0) or 0)
    labels = manifest_rec.get("labels", [])
    if W <= 0 or H <= 0 or not isinstance(labels, list):
        return {
            "image_width": W,
            "image_height": H,
            "instances": [],             # ["cup [x1,y1,x2,y2]", ...]
            "instances_xyxy": [],        # [[x1,y1,x2,y2], ...]
            "instances_xywhn": [],       # [[cx,cy,w,h], ...]
            "instances_area": [],        # [area, ...]
        }

    matched = []
    for a in labels:
        name = str(a.get("name", "")).strip()
        if name != object_name:
            continue
        xywhn = a.get("xywhn", None)
        if not (isinstance(xywhn, list) and len(xywhn) == 4):
            continue
        area = float(a.get("area", float(xywhn[2]) * float(xywhn[3])))
        matched.append((area, [float(x) for x in xywhn]))

    # Sort by area from large to small; stable and more like "primary instance first"
    matched.sort(key=lambda t: t[0], reverse=True)

    instances_xywhn = [m[1] for m in matched]
    instances_area = [float(m[0]) for m in matched]
    instances_xyxy = [xywhn_to_xyxy(xywhn, W, H) for xywhn in instances_xywhn]
    instances = [f"{object_name} {fmt_xyxy(xyxy, nd=nd)}" for xyxy in instances_xyxy]

    return {
        "image_width": W,
        "image_height": H,
        "instances": instances,
        "instances_xyxy": instances_xyxy,
        "instances_xywhn": instances_xywhn,
        "instances_area": instances_area,
    }

# =========================
# Merge: round1 -> merged (batch)
# =========================
def merge_one_file(round1_path: Path, manifest_idx: dict):
    out_path = round1_path.with_name(round1_path.stem + MERGED_SUFFIX + round1_path.suffix)

    n_total = 0
    n_ok = 0
    n_manifest_miss = 0

    with round1_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            if not rec.get("ok", False):
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            ip = rec.get("image_path", "")
            if not ip or ip not in manifest_idx:
                n_manifest_miss += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            mrec = manifest_idx[ip]
            n_ok += 1

            # Add the image dimensions into the round-1 result (so round 2 can get them even when only reading round-1 results)
            result = rec.get("result", {})
            if not isinstance(result, dict):
                result = {}

            # The "name + coordinates" unique identifier you agreed on later; the core is right here: the instances string list
            result["image_width"] = int(mrec.get("width", 0) or 0)
            result["image_height"] = int(mrec.get("height", 0) or 0)

            selected = result.get("selected", [])
            if isinstance(selected, list):
                for item in selected:
                    if not isinstance(item, dict):
                        continue
                    obj = str(item.get("object", "")).strip()
                    if not obj:
                        continue

                    pack = collect_object_instances(mrec, obj, nd=COORD_DECIMALS)

                    # These three fields are what your round-2 prompt injection will use:
                    # item["instances"] lets the VLM quote line by line verbatim (including coordinates) to disambiguate
                    # item["image_width/height"] if you want to write the dimensions in the prompt, you can also use the ones in result
                    item["instances"] = pack["instances"]

                    # If in round 2 you also want stronger constraints on bbox quality/scale, you can use these "numeric fields" as a rubric
                    # But since you said you don't want the model to output bbox, these are only for internal prompt use
                    item["instances_xyxy"] = pack["instances_xyxy"]
                    item["instances_area"] = pack["instances_area"]

            rec["result"] = result
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] merged file: {out_path}")
    print(f"[INFO] total={n_total} ok={n_ok} manifest_miss={n_manifest_miss}")

def main():
    assert MANIFEST_JSONL.exists(), f"manifest not found: {MANIFEST_JSONL}"
    manifest_idx = load_manifest_index(MANIFEST_JSONL)

    round1_files = sorted(OUT_ROOT.glob(ROUND1_PATTERN))
    print(f"[INFO] found round1 files: {len(round1_files)}")
    if not round1_files:
        return

    for p in round1_files:
        merge_one_file(p, manifest_idx)

if __name__ == "__main__":
    main()