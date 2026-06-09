import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


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


def safe_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def norm_score(v: float, hi: float) -> float:
    if hi <= 0:
        return 0.0
    v = max(0.0, min(float(v), float(hi)))
    return v / hi


def clean_object_name(x) -> str:
    if not isinstance(x, str):
        return ""
    return x.strip()


def canon_name(x) -> str:
    return clean_object_name(x).lower()


def count_confidence_score(conf: str) -> float:
    conf = clean_object_name(conf).lower()
    if conf == "high":
        return 1.0
    if conf == "medium":
        return 0.65
    if conf == "low":
        return 0.35
    return 0.0


def eval_item_score(e: dict) -> float:
    aff = norm_score(safe_int(e.get("affordance_quality", 0)), 5.0)
    obj = norm_score(safe_int(e.get("object_visible", 0)), 3.0)
    bbox = norm_score(safe_int(e.get("bbox_quality", 0)), 3.0)
    parts = norm_score(safe_int(e.get("parts_visibility", 0)), 3.0)
    return 0.45 * aff + 0.20 * obj + 0.20 * bbox + 0.15 * parts


def load_count_lookup(count_root: Path) -> Dict[Tuple[str, str], dict]:
    lookup: Dict[Tuple[str, str], dict] = {}
    for path in sorted(count_root.glob("[1-5]/*_count_object_only_v2.jsonl")):
        for rec in iter_jsonl(path):
            image_path = rec.get("image_path", "")
            obj = canon_name(rec.get("object", ""))
            if not image_path or not obj:
                continue
            key = (image_path, obj)
            old = lookup.get(key)
            if old is None:
                lookup[key] = rec
                continue
            old_conf = count_confidence_score((old.get("result") or {}).get("confidence", ""))
            new_conf = count_confidence_score((rec.get("result") or {}).get("confidence", ""))
            if new_conf > old_conf:
                lookup[key] = rec
    return lookup


def match_target_eval(rec: dict) -> Optional[Tuple[dict, float]]:
    result = rec.get("result", {})
    if not isinstance(result, dict):
        return None

    evals = result.get("evaluations", [])
    if not isinstance(evals, list):
        return None

    target_names = {
        canon_name(rec.get("keep_target_object", "")),
        canon_name(rec.get("keep_matched_label_name", "")),
    }
    target_names.discard("")
    if not target_names:
        return None

    best_item = None
    best_score = -1.0
    for e in evals:
        if not isinstance(e, dict):
            continue
        obj_name = canon_name(e.get("object", ""))
        if obj_name not in target_names:
            continue
        s = eval_item_score(e)
        if s > best_score:
            best_score = s
            best_item = e

    if best_item is None:
        return None
    return best_item, best_score


def build_candidate(keep_rec: dict, count_rec: dict, bucket_id: int, src_path: Path) -> Optional[dict]:
    if not keep_rec.get("ok", False):
        return None
    if count_rec.get("ok") is not True:
        return None

    image_path = keep_rec.get("image_path", "")
    if not isinstance(image_path, str) or not image_path:
        return None

    source_count = safe_int(keep_rec.get("keep_target_bbox_count", bucket_id), bucket_id)
    count_result = count_rec.get("result", {})
    if not isinstance(count_result, dict):
        return None

    estimated_count = clean_object_name(count_result.get("estimated_count", ""))
    if estimated_count != str(source_count):
        return None

    matched = match_target_eval(keep_rec)
    if matched is None:
        return None
    best_eval, target_eval_score = matched

    keep_result = keep_rec.get("result", {})
    if not isinstance(keep_result, dict):
        return None

    image_quality = safe_int(keep_result.get("image_quality", -1), -1)
    clutter = safe_int(keep_result.get("clutter", -1), -1)
    bw_yellow_veto = safe_int(keep_result.get("bw_yellow_veto", 0), 0)
    affordance_quality = safe_int(best_eval.get("affordance_quality", 0), 0)
    object_visible = safe_int(best_eval.get("object_visible", 0), 0)
    bbox_quality = safe_int(best_eval.get("bbox_quality", 0), 0)
    parts_visibility = safe_int(best_eval.get("parts_visibility", 0), 0)
    target_affordance = clean_object_name(best_eval.get("affordance", "")).lower()
    conf = clean_object_name(count_result.get("confidence", "")).lower()
    conf_score = count_confidence_score(conf)

    img_norm = norm_score(image_quality, 5.0)
    clutter_norm = 1.0 - norm_score(clutter, 5.0)
    final_score = (
        0.30 * img_norm
        + 0.40 * target_eval_score
        + 0.15 * clutter_norm
        + 0.15 * conf_score
    )

    return {
        "image_path": image_path,
        "bucket_id": int(bucket_id),
        "source_file": str(src_path),
        "target_object": clean_object_name(keep_rec.get("keep_target_object", "")),
        "matched_label_name": clean_object_name(keep_rec.get("keep_matched_label_name", "")),
        "source_count": source_count,
        "estimated_count": estimated_count,
        "count_confidence": conf,
        "count_confidence_score": round(conf_score, 6),
        "image_quality": image_quality,
        "clutter": clutter,
        "bw_yellow_veto": bw_yellow_veto,
        "target_affordance_quality": affordance_quality,
        "target_affordance": target_affordance,
        "target_object_visible": object_visible,
        "target_bbox_quality": bbox_quality,
        "target_parts_visibility": parts_visibility,
        "target_eval_score": round(target_eval_score, 6),
        "final_score": round(final_score, 6),
        "target_eval": best_eval,
        "keep_record": keep_rec,
        "count_record": count_rec,
    }


def adaptive_thresholds(obj_count: int, max_count: int) -> dict:
    if obj_count <= 0 or max_count <= 1:
        strictness = 0.0
    else:
        strictness = math.log1p(obj_count) / math.log1p(max_count)

    return {
        "strictness": round(strictness, 6),
        "min_image_quality": 3 if strictness < 0.35 else 4,
        "max_clutter": 4 if strictness < 0.20 else (3 if strictness < 0.65 else 2),
        "min_affordance_quality": 3 if strictness < 0.40 else 4,
        "min_object_visible": 1 if strictness < 0.25 else 2,
        "min_bbox_quality": 1 if strictness < 0.25 else 2,
        "min_target_eval_score": round(0.45 + 0.25 * strictness, 6),
        "min_final_score": round(0.52 + 0.23 * strictness, 6),
    }


def keep_by_thresholds(cand: dict, thr: dict, allow_bw_veto: bool) -> Tuple[bool, str]:
    if not allow_bw_veto and safe_int(cand.get("bw_yellow_veto", 0), 0) == 1:
        return False, "bw_yellow_veto"
    if safe_int(cand.get("image_quality", -1), -1) < thr["min_image_quality"]:
        return False, "image_quality"
    if safe_int(cand.get("clutter", 99), 99) > thr["max_clutter"]:
        return False, "clutter"
    if safe_int(cand.get("target_affordance_quality", 0), 0) < thr["min_affordance_quality"]:
        return False, "target_affordance_quality"
    if safe_int(cand.get("target_object_visible", 0), 0) < thr["min_object_visible"]:
        return False, "target_object_visible"
    if safe_int(cand.get("target_bbox_quality", 0), 0) < thr["min_bbox_quality"]:
        return False, "target_bbox_quality"
    if float(cand.get("target_eval_score", 0.0)) < thr["min_target_eval_score"]:
        return False, "target_eval_score"
    if float(cand.get("final_score", 0.0)) < thr["min_final_score"]:
        return False, "final_score"
    return True, "pass"


def compute_object_cap(
    obj_name: str,
    affordance: str,
    available_count: int,
    rare_keep_all_threshold: int,
    cap_sqrt_scale: float,
    wear_cap_scale: float,
    chair_hard_cap: int,
    helmet_hard_cap: int,
) -> int:
    available_count = max(0, int(available_count))
    if available_count <= rare_keep_all_threshold:
        cap = available_count
    else:
        cap = int(
            round(
                rare_keep_all_threshold
                + cap_sqrt_scale * math.sqrt(max(0, available_count - rare_keep_all_threshold))
            )
        )
        cap = min(cap, available_count)

    if affordance == "wear":
        cap = int(math.floor(cap * max(0.0, wear_cap_scale)))

    if obj_name == "chair" and chair_hard_cap >= 0:
        cap = min(cap, chair_hard_cap)
    if obj_name == "helmet" and helmet_hard_cap >= 0:
        cap = min(cap, helmet_hard_cap)

    return max(1, min(cap, available_count)) if available_count > 0 else 0


def parse_args():
    ap = argparse.ArgumentParser(
        description="Adaptive high-quality selection: relaxed for rare objects, strict for frequent objects."
    )
    ap.add_argument("--keep_root", type=str, default="data/keep")
    ap.add_argument("--count_root", type=str, default="data/count")
    ap.add_argument(
        "--out_root",
        type=str,
        default="data/final_adaptive_quality_rebalanced",
    )
    ap.add_argument(
        "--pattern",
        type=str,
        default="object365_patch*_round1_aff_select_merged_v1_round2_affordance_verify_v1.jsonl",
    )
    ap.add_argument("--allow_bw_veto", action="store_true")
    ap.add_argument("--rare_keep_all_threshold", type=int, default=200)
    ap.add_argument("--cap_sqrt_scale", type=float, default=12.0)
    ap.add_argument("--wear_cap_scale", type=float, default=0.55)
    ap.add_argument("--chair_hard_cap", type=int, default=1200)
    ap.add_argument("--helmet_hard_cap", type=int, default=300)
    return ap.parse_args()


def main():
    args = parse_args()
    keep_root = Path(args.keep_root)
    count_root = Path(args.count_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    count_lookup = load_count_lookup(count_root)

    total_keep_rows = 0
    total_count_matched_rows = 0
    missing_count_rows = 0
    missing_target_eval_rows = 0
    candidates: List[dict] = []

    keep_files = sorted(keep_root.glob(f"[1-5]/{args.pattern}"))
    if not keep_files:
        raise FileNotFoundError(f"No keep files matched: {keep_root}/[1-5]/{args.pattern}")

    for src_path in keep_files:
        bucket_id = safe_int(src_path.parent.name, 0)
        for rec in iter_jsonl(src_path):
            total_keep_rows += 1
            image_path = rec.get("image_path", "")
            target_obj = canon_name(rec.get("keep_target_object", ""))
            if not image_path or not target_obj:
                continue

            count_rec = count_lookup.get((image_path, target_obj))
            if count_rec is None:
                missing_count_rows += 1
                continue

            cand = build_candidate(rec, count_rec, bucket_id=bucket_id, src_path=src_path)
            if cand is None:
                if match_target_eval(rec) is None:
                    missing_target_eval_rows += 1
                continue

            total_count_matched_rows += 1
            candidates.append(cand)

    object_counter = Counter(c["target_object"] for c in candidates)
    max_object_count = max(object_counter.values()) if object_counter else 1

    reject_reason_counter = Counter()

    per_object_thresholds = {}
    for obj, cnt in object_counter.items():
        per_object_thresholds[obj] = adaptive_thresholds(cnt, max_object_count)

    passed_candidates: List[dict] = []
    for cand in candidates:
        obj = cand["target_object"]
        thr = per_object_thresholds[obj]
        keep_it, reason = keep_by_thresholds(cand, thr, allow_bw_veto=args.allow_bw_veto)
        if not keep_it:
            reject_reason_counter[reason] += 1
            continue

        passed_candidates.append(cand)

    passed_counter = Counter(c["target_object"] for c in passed_candidates)
    affordance_by_object: Dict[str, str] = {}
    for c in passed_candidates:
        affordance_by_object.setdefault(c["target_object"], c["target_affordance"])

    per_object_caps = {}
    for obj, cnt in passed_counter.items():
        per_object_caps[obj] = compute_object_cap(
            obj_name=obj,
            affordance=affordance_by_object.get(obj, ""),
            available_count=cnt,
            rare_keep_all_threshold=args.rare_keep_all_threshold,
            cap_sqrt_scale=args.cap_sqrt_scale,
            wear_cap_scale=args.wear_cap_scale,
            chair_hard_cap=args.chair_hard_cap,
            helmet_hard_cap=args.helmet_hard_cap,
        )

    passed_candidates.sort(
        key=lambda x: (
            x["final_score"],
            x["target_eval_score"],
            x["count_confidence_score"],
            x["image_quality"],
            -x["clutter"],
        ),
        reverse=True,
    )

    selected_by_output: Dict[Path, List[dict]] = defaultdict(list)
    kept_counter = Counter()
    rebalance_reject_counter = Counter()
    for cand in passed_candidates:
        obj = cand["target_object"]
        if kept_counter[obj] >= per_object_caps.get(obj, 0):
            rebalance_reject_counter[obj] += 1
            continue

        src_path = Path(cand["source_file"])
        out_path = out_root / str(cand["bucket_id"]) / src_path.name
        rec = dict(cand["keep_record"])
        rec["count_result_v2"] = cand["count_record"]
        rec["adaptive_quality_filter"] = {
            "target_object": cand["target_object"],
            "target_affordance": cand["target_affordance"],
            "source_count": cand["source_count"],
            "estimated_count": cand["estimated_count"],
            "count_confidence": cand["count_confidence"],
            "target_eval_score": cand["target_eval_score"],
            "target_affordance_quality": cand["target_affordance_quality"],
            "target_object_visible": cand["target_object_visible"],
            "target_bbox_quality": cand["target_bbox_quality"],
            "target_parts_visibility": cand["target_parts_visibility"],
            "image_quality": cand["image_quality"],
            "clutter": cand["clutter"],
            "final_score": cand["final_score"],
            "object_candidate_count": object_counter[obj],
            "object_passed_quality_count": passed_counter[obj],
            "object_cap_after_rebalance": per_object_caps.get(obj, 0),
            "thresholds": thr,
        }
        selected_by_output[out_path].append(rec)
        kept_counter[obj] += 1

    for out_path, rows in selected_by_output.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    per_bucket_summary = {}
    for bucket_id in range(1, 6):
        kept = 0
        for out_path, rows in selected_by_output.items():
            if out_path.parent.name == str(bucket_id):
                kept += len(rows)
        per_bucket_summary[str(bucket_id)] = {"kept_rows": kept}

    object_summary = []
    for obj, cnt in object_counter.most_common():
        object_summary.append(
            {
                "object": obj,
                "matched_count_candidates": int(cnt),
                "passed_quality_count": int(passed_counter.get(obj, 0)),
                "rebalance_cap": int(per_object_caps.get(obj, 0)),
                "kept_count": int(kept_counter.get(obj, 0)),
                "keep_ratio": round(kept_counter.get(obj, 0) / cnt, 6) if cnt > 0 else 0.0,
                "target_affordance": affordance_by_object.get(obj, ""),
                "thresholds": per_object_thresholds[obj],
            }
        )

    summary = {
        "keep_root": str(keep_root),
        "count_root": str(count_root),
        "out_root": str(out_root),
        "total_keep_rows": total_keep_rows,
        "count_lookup_rows": len(count_lookup),
        "missing_count_rows": missing_count_rows,
        "missing_target_eval_rows": missing_target_eval_rows,
        "count_matched_rows_before_adaptive_quality": total_count_matched_rows,
        "passed_adaptive_quality_before_rebalance": len(passed_candidates),
        "final_kept_rows": sum(len(v) for v in selected_by_output.values()),
        "unique_output_files": len(selected_by_output),
        "allow_bw_veto": bool(args.allow_bw_veto),
        "max_object_candidate_count": int(max_object_count),
        "rebalance_config": {
            "rare_keep_all_threshold": int(args.rare_keep_all_threshold),
            "cap_sqrt_scale": float(args.cap_sqrt_scale),
            "wear_cap_scale": float(args.wear_cap_scale),
            "chair_hard_cap": int(args.chair_hard_cap),
            "helmet_hard_cap": int(args.helmet_hard_cap),
        },
        "per_bucket_summary": per_bucket_summary,
        "reject_reason_top20": [
            {"reason": k, "count": int(v)} for k, v in reject_reason_counter.most_common(20)
        ],
        "rebalance_reject_top20": [
            {"object": k, "count": int(v)} for k, v in rebalance_reject_counter.most_common(20)
        ],
        "object_summary_top200": object_summary[:200],
    }

    summary_path = out_root / "adaptive_quality_summary.json"
    object_path = out_root / "adaptive_quality_object_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with object_path.open("w", encoding="utf-8") as f:
        json.dump(object_summary, f, ensure_ascii=False, indent=2)

    print("[DONE] adaptive quality filtering finished")
    print(f"[DONE] total_keep_rows={total_keep_rows}")
    print(f"[DONE] count_matched_rows_before_adaptive_quality={total_count_matched_rows}")
    print(f"[DONE] final_kept_rows={sum(len(v) for v in selected_by_output.values())}")
    print(f"[DONE] summary={summary_path}")
    print(f"[DONE] object_summary={object_path}")


if __name__ == "__main__":
    main()
