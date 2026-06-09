"""
Validate SegAgent Swift JSONL training data.

Checks per sample:
  - Valid JSON, correct schema (messages + images)
  - Image file exists on disk
  - User content has <image> tag and <ref>description</ref>
  - Assistant content matches expected pattern (IOU + click type + coords)
  - Coordinates in [0, 1000]
  - IOU in [0, 1]
  - Description non-empty and consistent between user/assistant

Reports stats + any errors found.

Usage:
    python data_process/validate_train_data.py \
        --jsonl data/affordance/segagent_train/train.jsonl \
        --max_errors 20
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ── Patterns ──────────────────────────────────────────────────────────────────
RE_ASSISTANT = re.compile(
    r"<ref>(.+?)</ref>\s+Current IOU:\s*([0-9.]+),\s*(Positive|Negative)\s+point:\s*\((\d+),\s*(\d+)\),\s*Predicted next IOU:\s*([0-9.]+)"
)
RE_USER_DESC = re.compile(r"<ref>(.+?)</ref>")


def validate_sample(sample: dict, idx: int) -> list[str]:
    errors = []

    # Schema
    if "messages" not in sample or "images" not in sample:
        errors.append(f"[{idx}] Missing 'messages' or 'images' key")
        return errors

    msgs = sample["messages"]
    imgs = sample["images"]

    if len(msgs) != 2:
        errors.append(f"[{idx}] Expected 2 messages, got {len(msgs)}")
        return errors

    if len(imgs) != 1:
        errors.append(f"[{idx}] Expected 1 image, got {len(imgs)}")

    # Image exists
    img_path = Path(imgs[0])
    if not img_path.exists():
        errors.append(f"[{idx}] Image not found: {imgs[0]}")

    # User message
    user_content = msgs[0].get("content", "")
    if "<image>" not in user_content:
        errors.append(f"[{idx}] User content missing <image> tag")

    user_match = RE_USER_DESC.search(user_content)
    if not user_match:
        errors.append(f"[{idx}] User content missing <ref>description</ref>")
        user_desc = ""
    else:
        user_desc = user_match.group(1).strip()
        if not user_desc:
            errors.append(f"[{idx}] Empty description in user content")

    # Assistant message
    asst_content = msgs[1].get("content", "")
    m = RE_ASSISTANT.match(asst_content)
    if not m:
        errors.append(f"[{idx}] Assistant content doesn't match expected pattern: {asst_content!r}")
        return errors

    asst_desc, iou_str, click_type, x_str, y_str, next_iou_str = m.groups()

    # Description consistency
    if user_desc and asst_desc.strip() != user_desc:
        errors.append(f"[{idx}] Description mismatch: user={user_desc!r} asst={asst_desc!r}")

    # IOU range
    try:
        iou = float(iou_str)
        if not (0.0 <= iou <= 1.0):
            errors.append(f"[{idx}] Current IOU out of range: {iou}")
    except ValueError:
        errors.append(f"[{idx}] Invalid current IOU value: {iou_str}")

    try:
        next_iou = float(next_iou_str)
        if not (0.0 <= next_iou <= 1.0):
            errors.append(f"[{idx}] Predicted next IOU out of range: {next_iou}")
    except ValueError:
        errors.append(f"[{idx}] Invalid predicted next IOU value: {next_iou_str}")

    # Coordinate range
    try:
        x, y = int(x_str), int(y_str)
        if not (0 <= x <= 1000 and 0 <= y <= 1000):
            errors.append(f"[{idx}] Coordinates out of range: ({x}, {y})")
    except ValueError:
        errors.append(f"[{idx}] Invalid coordinates: ({x_str}, {y_str})")

    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",       default="data/affordance/segagent_train/train.jsonl")
    ap.add_argument("--max_errors",  type=int, default=20, help="stop reporting after N errors")
    ap.add_argument("--sample_rate", type=int, default=1,  help="check every Nth line (1=all)")
    args = ap.parse_args()

    total = 0
    error_count = 0
    img_missing = 0
    click_types = Counter()
    iou_buckets = Counter()   # 0-0.1, 0.1-0.2, ...
    coord_x, coord_y = [], []

    print(f"Validating: {args.jsonl}")
    print(f"Sample rate: every {args.sample_rate} line(s)\n")

    with open(args.jsonl) as f:
        for i, line in enumerate(f):
            if i % args.sample_rate != 0:
                continue
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[{i}] JSON parse error: {e}")
                error_count += 1
                continue

            errors = validate_sample(sample, i)
            for e in errors:
                if "not found" in e:
                    img_missing += 1
                if error_count < args.max_errors:
                    print(e)
                error_count += len(errors)

            # Collect stats from valid samples
            if not errors:
                asst = sample["messages"][1]["content"]
                m = RE_ASSISTANT.match(asst)
                if m:
                    _, iou_str, click_type, x_str, y_str, _ = m.groups()
                    click_types[click_type] += 1
                    iou = float(iou_str)
                    bucket = int(iou * 10) / 10
                    iou_buckets[f"{bucket:.1f}"] += 1
                    coord_x.append(int(x_str))
                    coord_y.append(int(y_str))

            if (total) % 10000 == 0:
                print(f"  Checked {total} samples, errors so far: {error_count}", flush=True)

    print(f"\n{'='*50}")
    print(f"Total lines checked : {total}")
    print(f"Total errors        : {error_count}")
    print(f"  - image missing   : {img_missing}")
    print(f"\nClick type distribution:")
    for k, v in sorted(click_types.items()):
        print(f"  {k}: {v} ({100*v/sum(click_types.values()):.1f}%)")
    print(f"\nIOU distribution (step 0 is 0.00):")
    for k in sorted(iou_buckets):
        print(f"  {k}: {iou_buckets[k]}")
    if coord_x:
        import statistics
        print(f"\nCoordinate stats (x): mean={statistics.mean(coord_x):.0f}, "
              f"min={min(coord_x)}, max={max(coord_x)}")
        print(f"Coordinate stats (y): mean={statistics.mean(coord_y):.0f}, "
              f"min={min(coord_y)}, max={max(coord_y)}")
    print(f"\n{'PASS' if error_count == 0 else 'FAIL'} — {error_count} error(s) found")


if __name__ == "__main__":
    main()
