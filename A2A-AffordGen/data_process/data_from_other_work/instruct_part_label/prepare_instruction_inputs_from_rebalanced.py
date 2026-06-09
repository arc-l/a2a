import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ACTION_MAP = {
    "open": "open",
    "pull": "pull",
    "press": "press",
    "push": "push",
    "grasp": "hold",
    "carry": "carry",
    "pour": "pour",
    "scoop": "scoop",
    "cut": "cut",
    "wipe": "wipe",
    "sit": "sit on",
    "sit_on": "sit on",
    "wear": "wear",
    "type": "type on",
    "type_on": "type on",
    "support": "support",
    "support_with": "support",
    "lay": "place on",
    "lie_on": "lie on",
    "drink_with": "drink from",
    "eat": "eat with",
    "hold": "hold",
    "move": "move",
    "look_out": "look through",
    "talk_on": "talk on",
    "write": "write with",
    "stir": "stir with",
    "cut_with": "cut with",
    "pack": "pack",
    "text_on": "text on",
    "wash": "wash with",
}


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


def clean_str(x) -> str:
    return x.strip() if isinstance(x, str) else ""


def norm(x) -> str:
    return clean_str(x).lower()


def dedupe_keep_order(parts: List[str]) -> List[str]:
    seen = set()
    out = []
    for p in parts:
        key = norm(p)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(clean_str(p))
    return out


def action_from_affordance(aff: str) -> str:
    a = norm(aff)
    return ACTION_MAP.get(a, a.replace("_", " "))


def collect_target_parts(rec: dict) -> List[Tuple[str, List[str]]]:
    target_obj = norm(rec.get("keep_target_object", ""))
    target_aff = norm(((rec.get("adaptive_quality_filter") or {}).get("target_affordance", "")))
    evals = ((rec.get("result") or {}).get("evaluations") or [])

    grouped: Dict[Tuple[str, str], List[str]] = {}
    for e in evals:
        if not isinstance(e, dict):
            continue
        obj = norm(e.get("object", ""))
        aff = norm(e.get("affordance", ""))
        if obj != target_obj:
            continue
        if target_aff and aff != target_aff:
            continue
        if int(e.get("affordance_quality", 0) or 0) <= 0:
            continue
        key = (clean_str(e.get("object", "")), clean_str(e.get("affordance", "")))
        grouped.setdefault(key, [])
        parts = e.get("parts", [])
        if isinstance(parts, str):
            parts = [parts]
        if isinstance(parts, list):
            grouped[key].extend(clean_str(p) for p in parts if clean_str(p))

    out = []
    for (obj, aff), parts in grouped.items():
        parts = dedupe_keep_order(parts)
        if not obj or not aff or not parts:
            continue
        out.append((aff, parts))
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Prepare flat instruction annotation inputs from rebalanced dataset.")
    p.add_argument(
        "--input_root",
        default="data/final_adaptive_quality_rebalanced",
    )
    p.add_argument(
        "--out_root",
        default="data/test",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    items = []
    per_bucket = Counter()
    per_aff = Counter()
    per_obj = Counter()

    for bucket_id in ["1", "2", "3", "4", "5"]:
        for fp in sorted((input_root / bucket_id).glob("*.jsonl")):
            for ln, rec in enumerate(iter_jsonl(fp), 1):
                image_path = clean_str(rec.get("image_path", ""))
                obj = clean_str(rec.get("keep_target_object", ""))
                if not image_path or not obj:
                    continue
                aff_parts = collect_target_parts(rec)
                for aff, parts in aff_parts:
                    action = action_from_affordance(aff)
                    for part_idx, part in enumerate(parts):
                        item = {
                            "source": f"{bucket_id}/{fp.name}:{ln}:p{part_idx}",
                            "bucket_id": bucket_id,
                            "image_path": image_path,
                            "object": obj,
                            "part": part,
                            "affordance": aff,
                            "action": action,
                        }
                        items.append(item)
                        per_bucket[bucket_id] += 1
                        per_aff[norm(aff)] += 1
                        per_obj[norm(obj)] += 1

    parts = [[], [], [], []]
    load = [0, 0, 0, 0]
    for item in sorted(items, key=lambda x: (x["bucket_id"], x["image_path"], x["object"], x["part"], x["affordance"])):
        idx = min(range(4), key=lambda i: load[i])
        parts[idx].append(item)
        load[idx] += 1

    out_files = []
    for i in range(4):
        out_path = out_root / f"instruction_input_part{i}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in parts[i]:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        out_files.append(str(out_path))

    summary = {
        "input_root": str(input_root),
        "out_root": str(out_root),
        "total_items": len(items),
        "part_sizes": load,
        "per_bucket": {k: int(v) for k, v in sorted(per_bucket.items())},
        "per_affordance_top50": [{"affordance": k, "count": int(v)} for k, v in per_aff.most_common(50)],
        "per_object_top50": [{"object": k, "count": int(v)} for k, v in per_obj.most_common(50)],
        "out_files": out_files,
    }
    summary_path = out_root / "instruction_input_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()
