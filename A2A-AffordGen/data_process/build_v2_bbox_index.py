"""
Stream through DSDL train_samples.json and extract a compact bbox index
for Objects365 v2 images only.

Output: v2_bbox_index.json
{
  "objects365_v2_00952335": {
    "categories": ["mouse", "keyboard", ...],   # category names per annotation
    "annotations": [
      {"category": "mouse", "bbox": [x1,y1,x2,y2], "cat_index": 0},
      ...
    ]
  },
  ...
}

cat_index = index of this annotation among all annotations of the same category
in this image (matches data4sam3's object_index field).

Usage:
    python build_v2_bbox_index.py \
        --input /path/to/train_samples.json \
        --output /path/to/v2_bbox_index.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def build_index(input_path: str, output_path: str):
    print(f"Streaming {input_path} ...")

    # Load category id -> name from class-domain.yaml is complex;
    # instead we collect category_id from annotations and resolve later.
    # For our purpose we need: image_stem -> list of (category_id, bbox_xyxy)

    import ijson

    # First pass: collect category mapping from the DSDL categories field if present
    # The DSDL format embeds category info differently - we'll do two approaches:
    # 1. Try to read class-domain.yaml for category names
    # 2. Fall back to using category_id numbers directly

    # Load class-domain if available alongside the JSON
    input_dir = Path(input_path).parent.parent / "defs"
    cat_map = {}  # id -> name
    class_domain_path = input_dir / "class-domain.yaml"
    if class_domain_path.exists():
        import yaml
        with open(class_domain_path) as f:
            dom = yaml.safe_load(f)
        # DSDL class-domain format: ClassDom: {classes: [...]}
        # Each class entry has name field
        classes = dom.get("ClassDom", {}).get("classes", [])
        for i, cls in enumerate(classes, start=1):  # 1-indexed
            if isinstance(cls, dict):
                name = cls.get("name", cls.get("__value__", f"cat_{i}"))
            else:
                name = str(cls)
            cat_map[i] = name
        print(f"Loaded {len(cat_map)} categories from class-domain.yaml")

    # Stream through samples, collect v2 only
    v2_data = {}  # stem -> list of {category_id, bbox [x1,y1,x2,y2]}

    with open(input_path, "rb") as f:
        items = ijson.items(f, "samples.item")
        for i, sample in enumerate(items):
            media = sample.get("media", {})
            media_path = media.get("media_path", "")

            # Only process v2 images
            if "objects365_v2_" not in media_path:
                continue

            stem = Path(media_path).stem  # e.g. objects365_v2_00952335
            anns = sample.get("annotations", [])
            entries = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                entries.append({
                    "category_id": int(ann["category_id"]),
                    "category": cat_map.get(int(ann["category_id"]), f"cat_{ann['category_id']}"),
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                })
            v2_data[stem] = entries

            if (i + 1) % 100000 == 0:
                print(f"  Processed {i+1} samples, v2 collected: {len(v2_data)}")

    print(f"Total v2 images found: {len(v2_data)}")

    # Build cat_index: for each image, rank annotations within the same category
    # (matches data4sam3's object_index = index among same-category objects)
    result = {}
    for stem, entries in v2_data.items():
        cat_counts = defaultdict(int)
        indexed = []
        for e in entries:
            cat = e["category"]
            idx = cat_counts[cat]
            cat_counts[cat] += 1
            indexed.append({
                "category": cat,
                "category_id": e["category_id"],
                "cat_index": idx,
                "bbox": [round(v, 2) for v in e["bbox"]],
            })
        result[stem] = indexed

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f)
    print(f"Saved index to {output_path}")
    print(f"Index size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/Objects365_v2/OpenDataLab___Objects365/dsdl/dsdl_Det_full/set-train/train_samples.json",
    )
    parser.add_argument(
        "--output",
        default="data/Objects365_v2/v2_bbox_index.json",
    )
    args = parser.parse_args()
    build_index(args.input, args.output)


if __name__ == "__main__":
    main()
