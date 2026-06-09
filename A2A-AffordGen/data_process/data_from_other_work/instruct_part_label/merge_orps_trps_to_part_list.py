import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

PATH_PREFIX_TO_REMOVE = "data/data00/"

AFFORDANCE_TASK_FALLBACK = {
    "Open": "Open the {obj}",
    "Press": "Turn on the {obj}",
    "Push": "Move the {obj} forward",
    "Pull": "Pull the {obj}",
    "Pour": "Pour with the {obj}",
    "Contain": "Put something in the {obj}",
    "Display": "Check the {obj}",
    "Listen": "Listen with the {obj}",
    "Wear": "Wear the {obj}",
    "Grasp": "Pick up the {obj}",
    "Lift": "Lift the {obj}",
    "Move": "Move the {obj}",
    "Sit": "Sit on the {obj}",
    "Support": "Use the {obj} for support",
    "Lay": "Place something on the {obj}",
    "Wrap": "Wrap with the {obj}",
    "Cut": "Cut with the {obj}",
    "Stab": "Pierce with the {obj}",
}


def iter_jsonl(path: Path):
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


def clean_str(x) -> str:
    return x.strip() if isinstance(x, str) else ""


def norm(x) -> str:
    return clean_str(x).lower()


def normalize_image_path(path: str) -> str:
    path = clean_str(path)
    if path.startswith(PATH_PREFIX_TO_REMOVE):
        return path[len(PATH_PREFIX_TO_REMOVE) :]
    return path


def make_key(rec: dict) -> Tuple[str, str, str, str, str]:
    return (
        clean_str(rec.get("bucket_id", "")),
        normalize_image_path(rec.get("image_path", "")),
        norm(rec.get("object", "")),
        norm(rec.get("part", "")),
        norm(rec.get("affordance", "")),
    )


def get_orps_text(orps_rec: dict) -> str:
    q1 = clean_str(orps_rec.get("query_direct", ""))
    if q1:
        return q1
    part = clean_str(orps_rec.get("part", ""))
    obj = clean_str(orps_rec.get("object", ""))
    if part and obj:
        return f"{part} of the {obj}"
    return part or obj


def fallback_trps(obj: str, affordance: str) -> str:
    tmpl = AFFORDANCE_TASK_FALLBACK.get(clean_str(affordance), "Use the {obj}")
    return tmpl.format(obj=clean_str(obj))


def should_keep_part_item(part: str, affordance: str, action: str) -> bool:
    part_n = norm(part)
    affordance_n = norm(affordance)
    action_n = norm(action)
    if affordance_n == "sit_on" or action_n == "sit on":
        return part_n == "seat"
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Merge ORPS and TRPS outputs into final part_list JSON files by bucket.")
    p.add_argument("--test_root", default="data/test")
    return p.parse_args()


def main():
    args = parse_args()
    test_root = Path(args.test_root)

    orps_by_key: Dict[Tuple[str, str, str, str, str], dict] = {}
    trps_by_key: Dict[Tuple[str, str, str, str, str], dict] = {}

    for fp in sorted(test_root.glob("orps_affordance_annotated_part*.jsonl")):
        for rec in iter_jsonl(fp):
            if rec.get("ok") is not True:
                continue
            orps_by_key[make_key(rec)] = rec

    for fp in sorted(test_root.glob("trps_affordance_annotated_part*.jsonl")):
        for rec in iter_jsonl(fp):
            if rec.get("ok") is not True:
                continue
            trps_by_key[make_key(rec)] = rec

    merged_by_bucket_image = defaultdict(lambda: defaultdict(list))
    merge_stat = {
        "orps_ok": len(orps_by_key),
        "trps_ok": len(trps_by_key),
        "merged_parts": 0,
        "missing_orps": 0,
        "missing_trps": 0,
        "fallback_trps_used": 0,
    }

    all_keys = sorted(set(orps_by_key) | set(trps_by_key))
    for key in all_keys:
        o = orps_by_key.get(key)
        t = trps_by_key.get(key)
        if o is None:
            merge_stat["missing_orps"] += 1
            continue

        bucket_id, image_path, _, _, _ = key
        if t is None:
            merge_stat["missing_trps"] += 1
            trps_instruction = fallback_trps(
                clean_str(o.get("object", "")),
                clean_str(o.get("affordance", "")),
            )
            merge_stat["fallback_trps_used"] += 1
        else:
            trps_instruction = clean_str(t.get("instruction", ""))

        part_item = {
            "object": clean_str((t or {}).get("object", o.get("object", ""))),
            "part": clean_str((t or {}).get("part", o.get("part", ""))),
            "affordance": clean_str((t or {}).get("affordance", o.get("affordance", ""))),
            "action": clean_str((t or {}).get("action", o.get("action", ""))),
            "instruction": {
                "trps": trps_instruction,
                "orps": get_orps_text(o),
            },
        }
        if not should_keep_part_item(
            part_item["part"],
            part_item["affordance"],
            part_item["action"],
        ):
            continue
        merged_by_bucket_image[bucket_id][image_path].append(part_item)
        merge_stat["merged_parts"] += 1

    bucket_summary = {}
    for bucket_id in ["1", "2", "3", "4", "5"]:
        records = []
        image_map = merged_by_bucket_image.get(bucket_id, {})
        for image_path in sorted(image_map):
            records.append({"image_path": image_path, "part_list": image_map[image_path]})
        out_dir = test_root / bucket_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "merged_annotations.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        bucket_summary[bucket_id] = {
            "images": len(records),
            "parts": sum(len(r["part_list"]) for r in records),
            "output_json": str(out_path),
        }

    summary = {
        "test_root": str(test_root),
        "merge_stat": merge_stat,
        "bucket_summary": bucket_summary,
    }
    summary_path = test_root / "merged_annotations_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
