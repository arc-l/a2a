import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def extract_eval_objects(rec: dict) -> Set[str]:
    out: Set[str] = set()
    res = rec.get("result", {})
    if not isinstance(res, dict):
        return out
    evals = res.get("evaluations", [])
    if not isinstance(evals, list):
        return out
    for e in evals:
        if not isinstance(e, dict):
            continue
        obj = e.get("object", "")
        if isinstance(obj, str) and obj.strip():
            out.add(norm_name(obj))
    return out


def collect_target_image_paths(files: List[Path]) -> Set[str]:
    paths = set()
    for fp in files:
        for rec in iter_jsonl(fp):
            ip = rec.get("image_path", "")
            if isinstance(ip, str) and ip:
                paths.add(ip)
    return paths


def build_manifest_index(manifest_jsonl: Path, target_paths: Set[str]) -> Dict[str, Counter]:
    index: Dict[str, Counter] = {}
    with manifest_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            ip = rec.get("image_path", "")
            if ip not in target_paths:
                continue
            labels = rec.get("labels", [])
            c = Counter()
            if isinstance(labels, list):
                for lb in labels:
                    if not isinstance(lb, dict):
                        continue
                    n = lb.get("name", "")
                    if isinstance(n, str) and n.strip():
                        c[norm_name(n)] += 1
            index[ip] = c
    return index


OBJECT_ALIASES = {
    "bakset": "basket",
    "dinning table": "dining table",
    "moniter/tv": "monitor/tv",
    "table teniis paddle": "table tennis paddle",
}


def match_one_object_to_labels(
    obj_norm: str,
    label_counter: Counter,
    fuzzy_match: bool,
    fuzzy_cutoff: float,
) -> Tuple[str, bool]:
    # returns (matched_label_name, is_fuzzy)
    if obj_norm in label_counter:
        return obj_norm, False

    alias = OBJECT_ALIASES.get(obj_norm)
    if alias and alias in label_counter:
        return alias, False

    if fuzzy_match and label_counter:
        label_names = list(label_counter.keys())
        hit = get_close_matches(obj_norm, label_names, n=1, cutoff=fuzzy_cutoff)
        if hit:
            return hit[0], True

    return "", False


def parse_args():
    p = argparse.ArgumentParser(
        description="Quality-only filter + object bbox count<=5 bucketing into keep/1..5"
    )
    p.add_argument(
        "--source_root",
        type=str,
        default="data/data",
        help="where object365_patch*_round1_aff_select_merged_v1_round2_affordance_verify_v1.jsonl are located",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="object365_patch*_round1_aff_select_merged_v1_round2_affordance_verify_v1.jsonl",
    )
    p.add_argument(
        "--manifest_jsonl",
        type=str,
        default="data/data00/obj365_yes_ann_train/manifest.jsonl",
    )
    p.add_argument(
        "--object_summary_json",
        type=str,
        default="data/select/object_image_count_summary.json",
    )
    p.add_argument(
        "--out_root",
        type=str,
        default="data/keep",
    )
    p.add_argument("--min_image_quality", type=int, default=2)
    p.add_argument("--max_clutter", type=int, default=4)
    p.add_argument("--require_bw_yellow_veto_zero", action="store_true", default=True)
    p.add_argument("--fuzzy_match", action="store_true", default=True)
    p.add_argument("--fuzzy_cutoff", type=float, default=0.90)
    return p.parse_args()


def load_allowed_objects(summary_json: Path) -> Set[str]:
    with summary_json.open("r", encoding="utf-8") as f:
        d = json.load(f)
    rows = d.get("objects", [])
    out = set()
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict):
                o = r.get("object", "")
                if isinstance(o, str) and o.strip():
                    out.add(norm_name(o))
    return out


def main():
    args = parse_args()

    source_root = Path(args.source_root)
    files = sorted(source_root.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No source files: {source_root}/{args.pattern}")

    manifest_jsonl = Path(args.manifest_jsonl)
    if not manifest_jsonl.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_jsonl}")

    object_summary_json = Path(args.object_summary_json)
    if not object_summary_json.exists():
        raise FileNotFoundError(f"object summary not found: {object_summary_json}")

    allowed_objects = load_allowed_objects(object_summary_json)
    if not allowed_objects:
        raise RuntimeError("No allowed objects loaded from summary json.")

    out_root = Path(args.out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    for k in range(1, 6):
        (out_root / str(k)).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] source_files={len(files)}")
    print(f"[INFO] allowed_objects={len(allowed_objects)} from {object_summary_json}")

    target_paths = collect_target_image_paths(files)
    manifest_index = build_manifest_index(manifest_jsonl, target_paths)
    print(f"[INFO] manifest_index_images={len(manifest_index)}")

    global_bucket = Counter()
    global_rows_in = 0
    global_rows_imgpass = 0
    global_rows_written = 0
    per_file_stats = {}
    bucket_unique_images: Dict[int, Set[str]] = {k: set() for k in range(1, 6)}

    for fp in files:
        file_stem = fp.stem
        writers = {k: (out_root / str(k) / f"{file_stem}.jsonl").open("w", encoding="utf-8") for k in range(1, 6)}

        stats = {
            "source_file": fp.name,
            "rows_in": 0,
            "rows_ok": 0,
            "rows_imgpass": 0,
            "rows_no_manifest": 0,
            "rows_written": 0,
            "candidates_total": 0,
            "candidates_allowed_obj": 0,
            "candidates_matched_obj": 0,
            "candidates_bucketed": 0,
        }
        per_bucket = Counter()

        for rec in iter_jsonl(fp):
            stats["rows_in"] += 1
            global_rows_in += 1

            if not rec.get("ok", False):
                continue
            stats["rows_ok"] += 1

            res = rec.get("result", {})
            if not isinstance(res, dict):
                continue

            bw = int(res.get("bw_yellow_veto", 0) or 0)
            iq = int(res.get("image_quality", 0) or 0)
            cl = int(res.get("clutter", 0) or 0)

            if args.require_bw_yellow_veto_zero and bw != 0:
                continue
            if iq < args.min_image_quality:
                continue
            if cl > args.max_clutter:
                continue

            stats["rows_imgpass"] += 1
            global_rows_imgpass += 1

            image_path = rec.get("image_path", "")
            if not isinstance(image_path, str) or not image_path:
                continue

            label_counter = manifest_index.get(image_path)
            if label_counter is None:
                stats["rows_no_manifest"] += 1
                continue

            eval_obj_names = extract_eval_objects(rec)
            stats["candidates_total"] += len(eval_obj_names)

            for obj_norm in sorted(eval_obj_names):
                if obj_norm not in allowed_objects:
                    continue
                stats["candidates_allowed_obj"] += 1

                matched_name, is_fuzzy = match_one_object_to_labels(
                    obj_norm=obj_norm,
                    label_counter=label_counter,
                    fuzzy_match=args.fuzzy_match,
                    fuzzy_cutoff=args.fuzzy_cutoff,
                )
                if not matched_name:
                    continue

                stats["candidates_matched_obj"] += 1
                bbox_count = int(label_counter[matched_name])
                if not (1 <= bbox_count <= 5):
                    continue

                out = dict(rec)
                out["keep_quality_only"] = True
                out["keep_target_object"] = obj_norm
                out["keep_matched_label_name"] = matched_name
                out["keep_target_bbox_count"] = bbox_count
                out["keep_is_fuzzy_match"] = bool(is_fuzzy)
                out["keep_bbox_count_source"] = str(manifest_jsonl)
                out["keep_allowed_object_source"] = str(object_summary_json)

                writers[bbox_count].write(json.dumps(out, ensure_ascii=False) + "\n")
                stats["candidates_bucketed"] += 1
                stats["rows_written"] += 1
                global_rows_written += 1
                per_bucket[bbox_count] += 1
                global_bucket[bbox_count] += 1
                bucket_unique_images[bbox_count].add(image_path)

        for k in writers:
            writers[k].close()

        stats.update({f"bucket_{k}": int(per_bucket[k]) for k in range(1, 6)})
        per_file_stats[file_stem] = stats

    summary = {
        "source_root": str(source_root),
        "pattern": args.pattern,
        "manifest_jsonl": str(manifest_jsonl),
        "object_summary_json": str(object_summary_json),
        "out_root": str(out_root),
        "quality_filter": {
            "require_bw_yellow_veto_zero": bool(args.require_bw_yellow_veto_zero),
            "min_image_quality": int(args.min_image_quality),
            "max_clutter": int(args.max_clutter),
        },
        "match": {
            "fuzzy_match": bool(args.fuzzy_match),
            "fuzzy_cutoff": float(args.fuzzy_cutoff),
        },
        "global": {
            "rows_in": int(global_rows_in),
            "rows_imgpass": int(global_rows_imgpass),
            "rows_written": int(global_rows_written),
            "bucket_1": int(global_bucket[1]),
            "bucket_2": int(global_bucket[2]),
            "bucket_3": int(global_bucket[3]),
            "bucket_4": int(global_bucket[4]),
            "bucket_5": int(global_bucket[5]),
            "bucket_1_unique_images": len(bucket_unique_images[1]),
            "bucket_2_unique_images": len(bucket_unique_images[2]),
            "bucket_3_unique_images": len(bucket_unique_images[3]),
            "bucket_4_unique_images": len(bucket_unique_images[4]),
            "bucket_5_unique_images": len(bucket_unique_images[5]),
        },
        "per_file": per_file_stats,
    }

    summary_path = out_root / "keep_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] keep built:", out_root)
    print("[DONE] summary:", summary_path)
    print("[DONE] bucket counts:", {k: int(global_bucket[k]) for k in range(1, 6)})


if __name__ == "__main__":
    main()
