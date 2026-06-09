import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


def parse_args():
    p = argparse.ArgumentParser(description="Build ORPS/TRPS-compatible input from final count-filtered data")
    p.add_argument("--input_root", type=str, default="data/count_filtered_loose_vehicle")
    p.add_argument("--input_glob", type=str, default="[1-5]/object365_patch*_round1_aff_select_merged_v1_round2_affordance_verify_v1.jsonl")
    p.add_argument("--output_dir", type=str, default="data/final_annotation_input")
    p.add_argument("--part_size", type=int, default=20000)
    return p.parse_args()


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
    if not isinstance(x, str):
        return ""
    return x.strip()


def clean_parts(parts) -> List[str]:
    if isinstance(parts, str):
        parts = [parts]
    if not isinstance(parts, list):
        return []
    out = []
    for p in parts:
        s = clean_str(p)
        if s:
            out.append(s)
    return out


def main():
    args = parse_args()
    in_root = Path(args.input_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(p) for p in sorted(glob.glob(str(in_root / args.input_glob)))]
    if not files:
        raise FileNotFoundError(f"no files matched: {in_root / args.input_glob}")

    # Aggregate to dedup repeated (image, object, affordance, part) across all files.
    agg: Dict[Tuple[str, str], Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    n_in = 0
    n_eval = 0
    n_used_eval = 0

    for fp in files:
        for rec in iter_jsonl(fp):
            n_in += 1
            image_path = clean_str(rec.get("image_path", ""))
            if not image_path:
                continue

            evals = rec.get("prev_criteria_pass_evaluations", None)
            if not isinstance(evals, list):
                # fallback to filtered evaluations field if available
                evals = rec.get("adaptive_pass_evaluations", None)
            if not isinstance(evals, list):
                # final fallback
                res = rec.get("result", {})
                evals = res.get("evaluations", []) if isinstance(res, dict) else []
                if not isinstance(evals, list):
                    evals = []

            for e in evals:
                if not isinstance(e, dict):
                    continue
                n_eval += 1
                obj = clean_str(e.get("object", ""))
                aff = clean_str(e.get("affordance", ""))
                if not obj or not aff:
                    continue
                parts = clean_parts(e.get("parts", []))
                if not parts:
                    parts = ["whole_object"]

                key = (image_path, obj)
                for p in parts:
                    agg[key][aff].add(p)
                    n_used_eval += 1

    # Build annotation records compatible with annotate_orps/trps scripts.
    records = []
    for (image_path, obj), aff_map in agg.items():
        selected = []
        for aff in sorted(aff_map.keys()):
            parts = sorted(aff_map[aff])
            selected.append({"affordance": aff, "parts": parts, "exists": True})
        records.append(
            {
                "ok": True,
                "image_path": image_path,
                "object": obj,
                "result": {"selected_affordances": selected},
            }
        )

    records.sort(key=lambda x: (x.get("image_path", ""), x.get("object", "")))

    merged_path = out_dir / "final_annotation_input_merged.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    part_paths = []
    if args.part_size > 0:
        for i in range(0, len(records), args.part_size):
            part = records[i : i + args.part_size]
            p = out_dir / f"final_annotation_input_part{i // args.part_size:03d}.jsonl"
            with p.open("w", encoding="utf-8") as f:
                for r in part:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            part_paths.append(str(p))

    summary = {
        "input_root": str(in_root),
        "files": [str(x) for x in files],
        "input_records": n_in,
        "input_evals": n_eval,
        "used_eval_entries": n_used_eval,
        "unique_image_object_records": len(records),
        "merged_output": str(merged_path),
        "part_size": args.part_size,
        "parts": part_paths,
    }
    with (out_dir / "final_annotation_input_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] built annotation input")
    print(f"[DONE] merged: {merged_path}")
    print(f"[DONE] records: {len(records)}")


if __name__ == "__main__":
    main()
