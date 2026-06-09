import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Split ORPS/TRPS final annotations into count buckets 1..5")
    p.add_argument("--count_root", type=str, default="data/count_filtered_loose_vehicle")
    p.add_argument("--merged_input", type=str, default="data/final_annotation_input/final_annotation_input_merged.jsonl")
    p.add_argument("--orps", type=str, default="data/final_annotations/orps_affordance_annotated.jsonl")
    p.add_argument("--trps", type=str, default="data/final_annotations/trps_affordance_annotated.jsonl")
    p.add_argument("--out_root", type=str, default="data/final_annotations_by_count")
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                x = json.loads(line)
            except Exception:
                continue
            if isinstance(x, dict):
                yield x


def clean(x) -> str:
    if not isinstance(x, str):
        return ""
    return x.strip()


def build_image_obj_to_bucket(count_root: Path) -> Tuple[Dict[Tuple[str, str], str], int]:
    mapping: Dict[Tuple[str, str], str] = {}
    conflicts = 0

    for b in ["1", "2", "3", "4", "5"]:
        for fp in sorted((count_root / b).glob("*.jsonl")):
            for rec in iter_jsonl(fp):
                image_path = clean(rec.get("image_path", ""))
                if not image_path:
                    continue
                evals = rec.get("prev_criteria_pass_evaluations", [])
                if not isinstance(evals, list):
                    evals = []
                for e in evals:
                    if not isinstance(e, dict):
                        continue
                    obj = clean(e.get("object", ""))
                    if not obj:
                        continue
                    k = (image_path, obj)
                    old = mapping.get(k)
                    if old is None:
                        mapping[k] = b
                    elif old != b:
                        conflicts += 1
    return mapping, conflicts


def build_line_bucket_map(merged_input: Path, image_obj_to_bucket: Dict[Tuple[str, str], str]) -> Dict[int, str]:
    line_to_bucket: Dict[int, str] = {}
    with merged_input.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            image_path = clean(rec.get("image_path", ""))
            obj = clean(rec.get("object", ""))
            b = image_obj_to_bucket.get((image_path, obj))
            if b in {"1", "2", "3", "4", "5"}:
                line_to_bucket[ln] = b
    return line_to_bucket


def parse_source_line(source: str) -> Optional[int]:
    # expected: final_annotation_input_merged.jsonl:<ln>:a<ai>:p<pi>
    m = re.search(r":(\d+):a\d+:p\d+$", source)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def split_file(
    in_path: Path,
    out_root: Path,
    line_to_bucket: Dict[int, str],
    image_obj_to_bucket: Dict[Tuple[str, str], str],
) -> Dict[str, int]:
    name = in_path.name
    writers = {}
    for b in ["1", "2", "3", "4", "5"]:
        d = out_root / b
        d.mkdir(parents=True, exist_ok=True)
        writers[b] = (d / name).open("w", encoding="utf-8")

    stat = Counter()

    for rec in iter_jsonl(in_path):
        stat["total"] += 1

        b = None
        source = clean(rec.get("source", ""))
        ln = parse_source_line(source) if source else None
        if ln is not None:
            b = line_to_bucket.get(ln)

        if b is None:
            image_path = clean(rec.get("image_path", ""))
            obj = clean(rec.get("object", ""))
            if image_path and obj:
                b = image_obj_to_bucket.get((image_path, obj))

        if b not in {"1", "2", "3", "4", "5"}:
            stat["unmatched"] += 1
            continue

        writers[b].write(json.dumps(rec, ensure_ascii=False) + "\n")
        stat[f"bucket_{b}"] += 1

    for w in writers.values():
        w.close()

    return {k: int(v) for k, v in stat.items()}


def main():
    args = parse_args()
    count_root = Path(args.count_root)
    merged_input = Path(args.merged_input)
    orps = Path(args.orps)
    trps = Path(args.trps)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    image_obj_to_bucket, conflicts = build_image_obj_to_bucket(count_root)
    line_to_bucket = build_line_bucket_map(merged_input, image_obj_to_bucket)

    summary = {
        "count_root": str(count_root),
        "merged_input": str(merged_input),
        "out_root": str(out_root),
        "image_obj_bucket_map_size": len(image_obj_to_bucket),
        "image_obj_bucket_conflicts": conflicts,
        "line_bucket_map_size": len(line_to_bucket),
    }

    if orps.exists():
        summary["orps"] = split_file(orps, out_root, line_to_bucket, image_obj_to_bucket)
    else:
        summary["orps"] = {"missing": 1}

    if trps.exists():
        summary["trps"] = split_file(trps, out_root, line_to_bucket, image_obj_to_bucket)
    else:
        summary["trps"] = {"missing": 1}

    with (out_root / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] split final annotations by count buckets")
    print(f"[DONE] summary: {out_root / 'split_summary.json'}")


if __name__ == "__main__":
    main()
