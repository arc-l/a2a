import json
from pathlib import Path

# =========================
# 固定配置写在这里（不使用 argparse）
# =========================
OUT_ROOT = Path("data/data00")
MANIFEST_JSONL = Path("data/data00/obj365_yes_ann_train/manifest.jsonl")

# 第一轮输出文件命名规则（你给的第一轮脚本）
ROUND1_PATTERN = "object365_patch*_round1_aff_select.jsonl"

# 合并后的输出，避免重名覆盖
MERGED_SUFFIX = "_merged_v1"  # 会生成：object365_patchX_round1_aff_select_merged_v1.jsonl

# bbox 注入格式精度（决定“name + coordinates”字符串长相；后续 VLM 必须逐字复刻）
COORD_DECIMALS = 2

# =========================
# 读取 manifest 索引（一次性）
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
                # 这里的 labels 是你在数据处理脚本里写入的：[{cid,name,xywhn,area}, ...]
                idx[ip] = rec
            except Exception:
                bad += 1
    print(f"[INFO] manifest loaded: lines={total}, indexed={len(idx)}, bad_lines={bad}")
    return idx

# =========================
# 几何转换
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
# 为某个 object_name 收集该图中所有同名实例的 bbox，并生成“name + coordinates”唯一标识字符串
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

    # 面积从大到小，稳定且更像“主实例优先”
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
# 合并：round1 -> merged（批量）
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

            # 把图像尺寸补进第一轮 result（第二轮只读第一轮结果时就能拿到）
            result = rec.get("result", {})
            if not isinstance(result, dict):
                result = {}

            # 你后续同意的“name + coordinates”唯一标识，核心就在这里：instances 字符串列表
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

                    # 这三个字段就是你第二轮 prompt 注入要用的：
                    # item["instances"] 让 VLM 按行逐字引用（含坐标）来 disambiguate
                    # item["image_width/height"] 如果你想在 prompt 里写尺寸，也可用 result 里的
                    item["instances"] = pack["instances"]

                    # 如果你第二轮还想做 bbox 质量/尺度的更强约束，可用这些“数值字段”做 rubic
                    # 但你说不想让模型输出 bbox，这里只是给 prompt 内部使用
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