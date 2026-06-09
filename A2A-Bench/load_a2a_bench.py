#!/usr/bin/env python3
"""
Reference loader for A2A-Bench.

Loads the InstructPart-style annotations or the evaluation manifest, resolves
image / mask paths relative to the dataset root, and (with --stats) prints a
quick summary. Dependency-light: only the standard library is required; Pillow
is used only if you ask it to open masks.

Layout it expects (see README.md "Dataset structure"):

    <root>/
      InstructPart/{data_all.json, images/, masks/}
      eval/{val_manifest.jsonl, images/, gt_mask/}
      train/annotations.json

Examples:
    python load_a2a_bench.py --root ./A2A-Bench-data --split instructpart --stats
    python load_a2a_bench.py --root ./A2A-Bench-data --split eval --stats
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_instructpart(root: Path, split: str = "instructpart"):
    """Yield (image_path, part_dict) records from an InstructPart-style file.

    `split` is one of: instructpart (-> InstructPart/data_all.json) or
    train (-> train/annotations.json).
    """
    if split == "train":
        ann_file = root / "train" / "annotations.json"
        img_dir = root / "train" / "images"
        mask_dir = root / "train" / "masks"
    else:
        ann_file = root / "InstructPart" / "data_all.json"
        img_dir = root / "InstructPart" / "images"
        mask_dir = root / "InstructPart" / "masks"

    records = json.loads(ann_file.read_text())
    for rec in records:
        image_path = img_dir / rec["image_path"]
        for part in rec.get("part_list", []):
            yield {
                "image": image_path,
                "object": part.get("object"),
                "part": part.get("part"),
                "affordance": part.get("affordance"),
                "action": part.get("action"),
                "instructions": part.get("instruction", []),
                # mask naming may vary by release; resolve_mask() tries a few.
                "mask": resolve_mask(mask_dir, rec, part),
            }


def resolve_mask(mask_dir: Path, rec: dict, part: dict):
    """Best-effort mask path resolution. Adjust to the hosted release's naming."""
    stem = Path(rec["image_path"]).stem
    candidates = [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}-{part.get('object')}-{part.get('part')}.png",
        mask_dir / f"{stem}_{part.get('part')}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # default guess; may not exist until data is downloaded


def load_eval_manifest(root: Path):
    """Yield evaluation queries from eval/val_manifest.jsonl."""
    man = root / "eval" / "val_manifest.jsonl"
    img_dir = root / "eval"
    with man.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield {
                "image": img_dir / rec["image_path"],
                "description": rec.get("description"),
                "gt_mask": img_dir / rec["gt_mask_path"],
                "gt_bbox": rec.get("gt_bbox"),
                "height": rec.get("height"),
                "width": rec.get("width"),
            }


def main():
    ap = argparse.ArgumentParser(description="A2A-Bench reference loader")
    ap.add_argument("--root", type=Path, required=True, help="dataset root dir")
    ap.add_argument("--split", choices=["instructpart", "train", "eval"],
                    default="instructpart")
    ap.add_argument("--stats", action="store_true", help="print a summary")
    ap.add_argument("--limit", type=int, default=0, help="print first N records")
    args = ap.parse_args()

    if args.split == "eval":
        records = list(load_eval_manifest(args.root))
        if args.stats:
            print(f"[eval] {len(records)} part queries")
        for r in records[: args.limit or 0]:
            print(r["description"], "->", r["gt_mask"].name)
        return

    records = list(load_instructpart(args.root, args.split))
    if args.stats:
        affs = Counter(r["affordance"] for r in records)
        objs = Counter(r["object"] for r in records)
        imgs = {str(r["image"]) for r in records}
        print(f"[{args.split}] {len(records)} part instances over {len(imgs)} images")
        print(f"  unique objects:     {len(objs)}")
        print(f"  unique affordances: {len(affs)}")
        print("  top affordances:   ", dict(affs.most_common(10)))
    for r in records[: args.limit or 0]:
        print(f"{r['object']}/{r['part']} [{r['affordance']}/{r['action']}] "
              f"<- {r['image'].name}  ({len(r['instructions'])} instructions)")


if __name__ == "__main__":
    main()
