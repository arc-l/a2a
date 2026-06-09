import argparse
import csv
import json
from collections import Counter, defaultdict
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


ERROR_MATCHES = [
    {"object": "barrel/bucket", "part": "handle", "wrong_affordances": ["Stir"]},
    {"object": "barrel/bucket", "part": "whole_object", "wrong_affordances": ["Eat"]},
    {"object": "bottle", "part": "cap", "wrong_affordances": ["Drink_with"]},
    {"object": "bottle", "part": "label", "wrong_affordances": ["Drink_with"]},
    {"object": "bottle", "part": "lid", "wrong_affordances": ["Drink_with"]},
    {"object": "canned", "part": "cap", "wrong_affordances": ["Drink_with"]},
    {"object": "canned", "part": "lid", "wrong_affordances": ["Drink_with", "Eat"]},
    {"object": "canned", "part": "pull_tab", "wrong_affordances": ["Drink_with"]},
    {"object": "canned", "part": "tab", "wrong_affordances": ["Drink_with"]},
    {"object": "flask", "part": "cap", "wrong_affordances": ["Drink_with"]},
    {"object": "flask", "part": "lid", "wrong_affordances": ["Drink_with", "Sip"]},
    {"object": "bowl/basin", "part": "base", "wrong_affordances": ["Eat"]},
    {"object": "bowl/basin", "part": "whole_object", "wrong_affordances": ["Take_photo"]},
    {"object": "cup", "part": "lid", "wrong_affordances": ["Eat"]},
    {"object": "wine glass", "part": "stem", "wrong_affordances": ["Drink_with", "Eat", "Sip"]},
    {"object": "wine glass", "part": "whole_object", "wrong_affordances": ["Eat"]},
    {"object": "cabinet/shelf", "part": "whole_object", "wrong_affordances": ["Take_photo"]},
    {"object": "desk", "part": "base", "wrong_affordances": ["Eat", "Type_on", "Write"]},
    {"object": "desk", "part": "drawer", "wrong_affordances": ["Write"]},
    {"object": "desk", "part": "drawers", "wrong_affordances": ["Write"]},
    {"object": "desk", "part": "edge", "wrong_affordances": ["Eat", "Type_on", "Write"]},
    {"object": "desk", "part": "leg", "wrong_affordances": ["Eat", "Type_on", "Write"]},
    {"object": "desk", "part": "legs", "wrong_affordances": ["Eat", "Type_on", "Write"]},
    {"object": "nightstand", "part": "surface", "wrong_affordances": ["Talk_on"]},
    {"object": "nightstand", "part": "whole_object", "wrong_affordances": ["Take_photo"]},
    {"object": "fan", "part": "blade", "wrong_affordances": ["Open", "Swing", "Turn_on"]},
    {"object": "fan", "part": "motor", "wrong_affordances": ["Open", "Swing", "Turn_on"]},
    {"object": "lamp", "part": "arm", "wrong_affordances": ["Type_on"]},
    {"object": "lamp", "part": "base", "wrong_affordances": ["Take_photo", "Talk_on", "Type_on"]},
    {"object": "lamp", "part": "head", "wrong_affordances": ["Take_photo", "Type_on"]},
    {"object": "lamp", "part": "lamp_head", "wrong_affordances": ["Take_photo"]},
    {"object": "lamp", "part": "lampshade", "wrong_affordances": ["Talk_on"]},
    {"object": "lamp", "part": "light bulb", "wrong_affordances": ["Take_photo"]},
    {"object": "lamp", "part": "mount", "wrong_affordances": ["Take_photo"]},
    {"object": "lamp", "part": "shade", "wrong_affordances": ["Take_photo", "Talk_on", "Type_on"]},
    {"object": "lamp", "part": "stand", "wrong_affordances": ["Take_photo"]},
    {"object": "lamp", "part": "switch", "wrong_affordances": ["Talk_on"]},
    {"object": "lamp", "part": "whole_object", "wrong_affordances": ["Take_photo"]},
    {"object": "pot", "part": "base", "wrong_affordances": ["Eat"]},
    {"object": "pot", "part": "handle", "wrong_affordances": ["Brush_with"]},
    {"object": "pot", "part": "interior", "wrong_affordances": ["Brush_with"]},
    {"object": "pot", "part": "lid", "wrong_affordances": ["Eat"]},
    {"object": "pot", "part": "whole_object", "wrong_affordances": ["Beat"]},
    {"object": "toilet", "part": "bowl", "wrong_affordances": ["Drink_with"]},
    {"object": "projector", "part": "body", "wrong_affordances": ["Take_photo"]},
    {"object": "projector", "part": "lens", "wrong_affordances": ["Take_photo"]},
    {"object": "boots", "part": "heel", "wrong_affordances": ["Ride"]},
    {"object": "boots", "part": "sole", "wrong_affordances": ["Ride"]},
    {"object": "boots", "part": "toe", "wrong_affordances": ["Ride"]},
    {"object": "boots", "part": "whole_object", "wrong_affordances": ["Ride", "Write"]},
    {"object": "gloves", "part": "whole_object", "wrong_affordances": ["Ride"]},
    {"object": "hat", "part": "whole_object", "wrong_affordances": ["Take_photo"]},
    {"object": "helmet", "part": "shell", "wrong_affordances": ["Ride"]},
    {"object": "helmet", "part": "strap", "wrong_affordances": ["Ride"]},
    {"object": "helmet", "part": "whole_object", "wrong_affordances": ["Ride"]},
    {"object": "high heels", "part": "whole_object", "wrong_affordances": ["Ride", "Take_photo"]},
    {"object": "leather shoes", "part": "sole", "wrong_affordances": ["Ride"]},
    {"object": "leather shoes", "part": "whole_object", "wrong_affordances": ["Ride"]},
    {"object": "sandals", "part": "whole_object", "wrong_affordances": ["Ride"]},
    {"object": "slippers", "part": "sole", "wrong_affordances": ["Ride"]},
    {"object": "slippers", "part": "upper", "wrong_affordances": ["Ride"]},
    {"object": "slippers", "part": "whole_object", "wrong_affordances": ["Ride"]},
    {"object": "sneakers", "part": "heel", "wrong_affordances": ["Ride"]},
    {"object": "sneakers", "part": "sole", "wrong_affordances": ["Ride"]},
    {"object": "sneakers", "part": "surface", "wrong_affordances": ["Write"]},
    {"object": "sneakers", "part": "toe", "wrong_affordances": ["Ride"]},
    {"object": "sneakers", "part": "upper", "wrong_affordances": ["Ride"]},
    {"object": "sneakers", "part": "whole_object", "wrong_affordances": ["Ride"]},
    {"object": "storage box", "part": "whole_object", "wrong_affordances": ["Eat", "Ride", "Take_photo"]},
    {"object": "toilet paper", "part": "whole_object", "wrong_affordances": ["Take_photo"]},
    {"object": "towel", "part": "whole_object", "wrong_affordances": ["Take_photo"]},
]


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
    return " ".join(clean_str(x).replace("_", " ").lower().split())


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
    if affordance_n == "sit on" or action_n == "sit on":
        return part_n == "seat"
    return True


def parse_args():
    p = argparse.ArgumentParser(description="Remap error_matches using dominant remaining affordances and regenerate merged outputs.")
    p.add_argument("--test_root", default="data/test")
    return p.parse_args()


def build_bad_triples():
    bad = set()
    for m in ERROR_MATCHES:
        for aff in m["wrong_affordances"]:
            bad.add((norm(m["object"]), norm(m["part"]), norm(aff)))
    return bad


def load_original_items(test_root: Path):
    orps_by_key: Dict[Tuple[str, str, str, str, str], dict] = {}
    trps_by_key: Dict[Tuple[str, str, str, str, str], dict] = {}

    for fp in sorted(test_root.glob("orps_affordance_annotated_part*.jsonl")):
        for rec in iter_jsonl(fp):
            if rec.get("ok") is True:
                orps_by_key[make_key(rec)] = rec

    for fp in sorted(test_root.glob("trps_affordance_annotated_part*.jsonl")):
        for rec in iter_jsonl(fp):
            if rec.get("ok") is True:
                trps_by_key[make_key(rec)] = rec

    items = []
    missing_orps = 0
    missing_trps = 0
    fallback_trps_used = 0

    for key in sorted(set(orps_by_key) | set(trps_by_key)):
        o = orps_by_key.get(key)
        t = trps_by_key.get(key)
        if o is None:
            missing_orps += 1
            continue

        bucket_id, image_path, _, _, _ = key
        if t is None:
            missing_trps += 1
            trps_instruction = fallback_trps(clean_str(o.get("object", "")), clean_str(o.get("affordance", "")))
            fallback_trps_used += 1
        else:
            trps_instruction = clean_str(t.get("instruction", ""))

        item = {
            "bucket_id": bucket_id,
            "image_path": image_path,
            "object": clean_str((t or {}).get("object", o.get("object", ""))),
            "part": clean_str((t or {}).get("part", o.get("part", ""))),
            "affordance": clean_str((t or {}).get("affordance", o.get("affordance", ""))),
            "action": clean_str((t or {}).get("action", o.get("action", ""))),
            "instruction": {
                "trps": trps_instruction,
                "orps": get_orps_text(o),
            },
            "source": clean_str(o.get("source", (t or {}).get("source", ""))),
        }
        items.append(item)

    return items, {
        "orps_ok": len(orps_by_key),
        "trps_ok": len(trps_by_key),
        "missing_orps": missing_orps,
        "missing_trps": missing_trps,
        "fallback_trps_used": fallback_trps_used,
    }


def build_cleaned_prototypes(test_root: Path):
    aff_counts = defaultdict(Counter)
    surface_counts = defaultdict(Counter)
    combo_counts = defaultdict(Counter)

    for fp in sorted(test_root.glob("[1-5]/merged_annotations.json")):
        with fp.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        for rec in rows:
            for item in rec.get("part_list", []):
                obj = clean_str(item.get("object", ""))
                part = clean_str(item.get("part", ""))
                aff = clean_str(item.get("affordance", ""))
                action = clean_str(item.get("action", ""))
                trps = clean_str(item.get("instruction", {}).get("trps", ""))
                orps = clean_str(item.get("instruction", {}).get("orps", ""))
                op_key = (norm(obj), norm(part))
                aff_n = norm(aff)
                aff_counts[op_key][aff_n] += 1
                surface_counts[(op_key, aff_n)][aff] += 1
                combo_counts[(op_key, aff_n)][(action, trps, orps)] += 1

    dominant = {}
    prototype = {}
    for op_key, counter in aff_counts.items():
        if counter:
            aff_n, _ = counter.most_common(1)[0]
            dominant[op_key] = aff_n
        for aff_n in counter:
            aff_surface = surface_counts[(op_key, aff_n)].most_common(1)[0][0]
            action, trps, orps = combo_counts[(op_key, aff_n)].most_common(1)[0][0]
            prototype[(op_key, aff_n)] = {
                "affordance": aff_surface,
                "action": action,
                "trps": trps,
                "orps": orps,
                "count": counter[aff_n],
            }
    return aff_counts, dominant, prototype


def dedupe_signature(item: dict):
    return (
        norm(item.get("object", "")),
        norm(item.get("part", "")),
        norm(item.get("affordance", "")),
        norm(item.get("action", "")),
        clean_str(item.get("instruction", {}).get("trps", "")),
        clean_str(item.get("instruction", {}).get("orps", "")),
    )


def regenerate_stats(test_root: Path):
    obj_part = defaultdict(lambda: {"affordances": set(), "actions": set(), "image_paths": set(), "item_count": 0, "buckets": set()})
    aff_to_obj_parts = defaultdict(lambda: defaultdict(lambda: {"parts": set(), "item_count": 0, "image_paths": set()}))
    aff_image_paths = defaultdict(set)
    aff_item_counts = Counter()
    aff_bucket_counts = defaultdict(Counter)
    all_parts = set()
    all_objects = set()
    all_aff = set()

    for fp in sorted(test_root.glob("[1-5]/merged_annotations.json")):
        bucket = fp.parent.name
        with fp.open("r", encoding="utf-8") as f:
            rows = json.load(f)
        for rec in rows:
            image_path = rec.get("image_path", "")
            for item in rec.get("part_list", []):
                obj = item.get("object", "")
                part = item.get("part", "")
                aff = item.get("affordance", "")
                action = item.get("action", "")
                if not obj or not part or not aff:
                    continue
                op = obj_part[(obj, part)]
                op["affordances"].add(aff)
                if action:
                    op["actions"].add(action)
                op["image_paths"].add(image_path)
                op["item_count"] += 1
                op["buckets"].add(bucket)

                all_aff.add(aff)
                all_objects.add(obj)
                all_parts.add(part)
                aff_item_counts[aff] += 1
                aff_image_paths[aff].add(image_path)
                aff_bucket_counts[aff][bucket] += 1
                aop = aff_to_obj_parts[aff][obj]
                aop["parts"].add(part)
                aop["item_count"] += 1
                aop["image_paths"].add(image_path)

    rows_out = []
    for (obj, part), d in sorted(obj_part.items(), key=lambda x: (x[0][0].lower(), x[0][1].lower())):
        rows_out.append(
            {
                "object": obj,
                "part": part,
                "affordance_count": len(d["affordances"]),
                "affordances": sorted(d["affordances"]),
                "actions": sorted(d["actions"]),
                "image_count": len(d["image_paths"]),
                "item_count": d["item_count"],
                "buckets": sorted(d["buckets"], key=int),
            }
        )

    payload = {"row_count": len(rows_out), "rows": rows_out}
    for base_name in ["object_part_to_affordances", "object_part_to_affordances_cleaned"]:
        with (test_root / f"{base_name}.json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        with (test_root / f"{base_name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        oneline_name = f"{base_name}_oneline.txt" if base_name.endswith("cleaned") else f"{base_name}_oneline.txt"
        with (test_root / oneline_name).open("w", encoding="utf-8") as f:
            for r in rows_out:
                f.write(
                    f"object={r['object']} | part={r['part']} | affordances={', '.join(r['affordances'])} | "
                    f"actions={', '.join(r['actions'])} | image_count={r['image_count']} | item_count={r['item_count']}\n"
                )
        with (test_root / f"{base_name}.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["object", "part", "affordance_count", "affordances", "actions", "image_count", "item_count", "buckets"])
            writer.writeheader()
            for r in rows_out:
                writer.writerow(
                    {
                        "object": r["object"],
                        "part": r["part"],
                        "affordance_count": r["affordance_count"],
                        "affordances": " | ".join(r["affordances"]),
                        "actions": " | ".join(r["actions"]),
                        "image_count": r["image_count"],
                        "item_count": r["item_count"],
                        "buckets": " | ".join(r["buckets"]),
                    }
                )

    summary = {
        "source": [str(p) for p in sorted(test_root.glob("[1-5]/merged_annotations.json"))],
        "total_affordances": len(all_aff),
        "total_objects": len(all_objects),
        "total_parts": len(all_parts),
        "affordances": {},
    }
    for aff in sorted(all_aff):
        objects = aff_to_obj_parts[aff]
        obj_summary = {}
        part_union = set()
        for obj in sorted(objects):
            rec = objects[obj]
            parts = sorted(rec["parts"])
            part_union.update(parts)
            obj_summary[obj] = {
                "part_count": len(parts),
                "parts": parts,
                "item_count": rec["item_count"],
                "image_count": len(rec["image_paths"]),
            }
        summary["affordances"][aff] = {
            "object_count": len(objects),
            "objects": obj_summary,
            "part_count_union": len(part_union),
            "parts_union": sorted(part_union),
            "item_count": aff_item_counts[aff],
            "image_count": len(aff_image_paths[aff]),
            "bucket_item_counts": {k: aff_bucket_counts[aff][k] for k in ["1", "2", "3", "4", "5"] if aff_bucket_counts[aff][k] > 0},
        }

    with (test_root / "affordance_object_part_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (test_root / "object_part_to_affordances_cleaned_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "row_count": len(rows_out),
                "total_affordances": len(all_aff),
                "total_objects": len(all_objects),
                "total_parts": len(all_parts),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return {
        "object_part_row_count": len(rows_out),
        "total_affordances": len(all_aff),
        "total_objects": len(all_objects),
        "total_parts": len(all_parts),
    }


def main():
    args = parse_args()
    test_root = Path(args.test_root)
    bad = build_bad_triples()

    aff_counts, dominant_aff, prototype = build_cleaned_prototypes(test_root)
    original_items, merge_stat = load_original_items(test_root)

    merged_by_bucket_image = defaultdict(lambda: defaultdict(list))
    seen = defaultdict(set)
    remap_counter = Counter()
    remap_examples = {}
    remap_no_target = Counter()

    for item in original_items:
        if not should_keep_part_item(item["part"], item["affordance"], item["action"]):
            continue

        obj_n = norm(item["object"])
        part_n = norm(item["part"])
        aff_n = norm(item["affordance"])
        bad_key = (obj_n, part_n, aff_n)

        if bad_key in bad:
            op_key = (obj_n, part_n)
            new_aff_n = dominant_aff.get(op_key)
            proto = prototype.get((op_key, new_aff_n)) if new_aff_n else None
            if not proto:
                remap_no_target[(item["object"], item["part"], item["affordance"])] += 1
                continue
            old_aff = item["affordance"]
            item["affordance"] = proto["affordance"]
            item["action"] = proto["action"] or item["action"]
            item["instruction"]["trps"] = proto["trps"] or fallback_trps(item["object"], item["affordance"])
            item["instruction"]["orps"] = proto["orps"] or item["instruction"]["orps"]
            remap_counter[(item["object"], item["part"], old_aff, item["affordance"])] += 1
            remap_examples.setdefault((item["object"], item["part"], old_aff, item["affordance"]), item["source"])

        sig = dedupe_signature(item)
        bucket_id = item["bucket_id"]
        image_path = item["image_path"]
        image_key = (bucket_id, image_path)
        if sig in seen[image_key]:
            continue
        seen[image_key].add(sig)

        merged_by_bucket_image[bucket_id][image_path].append(
            {
                "object": item["object"],
                "part": item["part"],
                "affordance": item["affordance"],
                "action": item["action"],
                "instruction": item["instruction"],
            }
        )

    bucket_summary = {}
    total_parts = 0
    for bucket_id in ["1", "2", "3", "4", "5"]:
        records = []
        image_map = merged_by_bucket_image.get(bucket_id, {})
        for image_path in sorted(image_map):
            part_list = sorted(image_map[image_path], key=lambda x: (norm(x["object"]), norm(x["part"]), norm(x["affordance"])))
            records.append({"image_path": image_path, "part_list": part_list})
        out_dir = test_root / bucket_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "merged_annotations.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        part_count = sum(len(r["part_list"]) for r in records)
        total_parts += part_count
        bucket_summary[bucket_id] = {"images": len(records), "parts": part_count, "output_json": str(out_path)}

    summary = {
        "test_root": str(test_root),
        "merge_stat": {**merge_stat, "merged_parts": total_parts},
        "bucket_summary": bucket_summary,
        "remap": {
            "triples_remapped": sum(remap_counter.values()),
            "unique_remap_rules_used": len(remap_counter),
            "remapped_pairs": [
                {
                    "object": obj,
                    "part": part,
                    "from_affordance": old_aff,
                    "to_affordance": new_aff,
                    "count": count,
                    "example_source": remap_examples[(obj, part, old_aff, new_aff)],
                }
                for (obj, part, old_aff, new_aff), count in remap_counter.most_common()
            ],
            "dropped_without_target": [
                {"object": obj, "part": part, "affordance": aff, "count": count}
                for (obj, part, aff), count in remap_no_target.most_common()
            ],
        },
    }
    with (test_root / "merged_annotations_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    stats = regenerate_stats(test_root)
    remap_summary = {
        "remapped_total": sum(remap_counter.values()),
        "dropped_without_target_total": sum(remap_no_target.values()),
        "stats": stats,
        "top_remaps": summary["remap"]["remapped_pairs"][:50],
    }
    with (test_root / "error_match_remap_summary.json").open("w", encoding="utf-8") as f:
        json.dump(remap_summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({"bucket_summary": bucket_summary, "remap": remap_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
