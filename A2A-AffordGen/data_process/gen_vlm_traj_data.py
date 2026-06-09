"""
Generate VLM fine-tuning data from affordance labeled data (data4sam3).

Prompt/output format (Qwen3.5 compatible, Swift JSONL):

  Input image : cropped original image (bbox + padding), NO mask overlay for step-0
  User prompt : <image>
                Please optimize the semi-transparent green mask in a point-wise
                manner based on the image and description content, so that it
                covers the target object as accurately as possible.
                The object description is as follows: <ref>{description}</ref>
  Answer      : <ref>{description}</ref> Current IOU: {iou}  {Positive/Negative} point: ({x}, {y})

Coordinates are in ABSOLUTE pixels of the CROPPED image.
Step-0 always has IOU=0 (no mask yet).  Only the first click is generated here;
multi-step data (with mask overlays) can be added later.

Usage:
    python gen_vlm_traj_data.py \
        --output_dir /path/to/cropped_images \
        --output_jsonl /path/to/train.jsonl \
        --pad_ratio 0.2
"""

import argparse
import json
import re
from pathlib import Path

from PIL import Image


# ── Path constants ──────────────────────────────────────────────────────────
DATA4SAM3_ROOT = Path("data/affordance/labeled_data_clean/data4sam3")
OBJ365_V1_TRAIN = Path(
    "data/Objects365_v1/OpenDataLab___Objects365_v1"
    "/raw/Objects365_v1/2019-08-02/train"
)
OBJ365_V2_DIR = Path(
    "data/affordance/assign_task/data_need_label"
)

# ── Prompt templates (Qwen3.5 / Swift format) ───────────────────────────────
USER_PROMPT = (
    "<image>\n"
    "Please optimize the semi-transparent green mask in a point-wise manner "
    "based on the image and description content, so that it covers the target "
    "object as accurately as possible. "
    "The object description is as follows: <ref>{description}</ref>"
)

# <ref>{desc}</ref> Current IOU: {iou}  {Positive/Negative} point: ({x}, {y})
ANSWER_TEMPLATE = "<ref>{description}</ref> Current IOU: {iou}  {click_type} point: ({x}, {y})"


# ── Image path resolution ────────────────────────────────────────────────────
def load_v1_id_to_filename(json_path: Path) -> dict:
    with open(json_path) as f:
        data = json.load(f)
    return {img["id"]: img["file_name"] for img in data["images"]}


def find_image_path(image_path_str: str, v1_id_map: dict) -> Path | None:
    filename = Path(image_path_str).name
    stem = Path(filename).stem

    if stem.startswith("objects365_v1_"):
        img_id = int(stem.replace("objects365_v1_", ""))
        v1_fname = v1_id_map.get(img_id)
        if v1_fname is None:
            return None
        return OBJ365_V1_TRAIN / v1_fname

    elif stem.startswith("objects365_v2_"):
        patch_match = re.search(r"patch(\d+)", image_path_str)
        if patch_match is None:
            return None
        patch_dir = f"patch{patch_match.group(1)}"
        return OBJ365_V2_DIR / patch_dir / filename

    return None


# ── Crop helpers ─────────────────────────────────────────────────────────────
def crop_with_padding(image: Image.Image, bbox: list, pad_ratio: float):
    """
    Crop to bbox [x1, y1, x2, y2] with proportional padding.
    Returns (cropped_image, crop_box) where crop_box = (cx1, cy1, cx2, cy2)
    in original image pixel coords.
    """
    W, H = image.size
    x1, y1, x2, y2 = bbox
    pad_x = max(1, int((x2 - x1) * pad_ratio))
    pad_y = max(1, int((y2 - y1) * pad_ratio))
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(W, x2 + pad_x)
    cy2 = min(H, y2 + pad_y)
    return image.crop((cx1, cy1, cx2, cy2)), (cx1, cy1, cx2, cy2)


def map_click_to_crop(click: dict, crop_box: tuple) -> tuple[int, int] | None:
    """
    Map a click from original image coords to cropped image coords (absolute px).
    Returns None if the click falls outside the crop region.
    """
    cx1, cy1, cx2, cy2 = crop_box
    px = click["x"] - cx1
    py = click["y"] - cy1
    cW, cH = cx2 - cx1, cy2 - cy1
    if px < 0 or py < 0 or px >= cW or py >= cH:
        return None
    return int(px), int(py)


# ── Sample builders ───────────────────────────────────────────────────────────
def build_step0_sample(
    cropped_img_path: str,
    description: str,
    first_click: dict,
    crop_box: tuple,
) -> dict | None:
    """
    Build a single-step training sample for step-0 (no mask, IOU=0).
    Predicts the first click in absolute pixels of the cropped image.
    """
    px, py = map_click_to_crop(first_click, crop_box) or (None, None)
    if px is None:
        return None

    click_type = "Positive" if first_click["type"] == "positive" else "Negative"
    answer = ANSWER_TEMPLATE.format(
        description=description,
        iou=0,
        click_type=click_type,
        x=px,
        y=py,
    )
    return {
        "messages": [
            {
                "role": "user",
                "content": USER_PROMPT.format(description=description),
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "images": [cropped_img_path],
    }


# ── Main processing ───────────────────────────────────────────────────────────
def process_annotation(
    ann_path: Path,
    v1_id_map: dict,
    output_img_dir: Path,
    pad_ratio: float,
) -> list[dict]:
    with open(ann_path) as f:
        ann = json.load(f)

    if ann.get("skipped") or not ann.get("completed"):
        return []

    img_path = find_image_path(ann["image_path"], v1_id_map)
    if img_path is None or not img_path.exists():
        return []

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception:
        return []

    samples = []
    img_stem = Path(ann["image_path"]).stem

    for part_key, part_data in ann.get("parts", {}).items():
        for inst_key, inst in part_data.get("instances", {}).items():
            bbox = inst.get("bbox")
            click_seq = inst.get("click_sequence", [])
            instruction = inst.get("instruction", {})
            description = instruction.get("orps", "the target object")

            if not bbox or not click_seq:
                continue

            # Crop image around bbox
            cropped, crop_box = crop_with_padding(image, bbox, pad_ratio)

            # Save cropped image (unique name per instance)
            save_name = f"{img_stem}__{inst_key}.jpg"
            save_path = output_img_dir / save_name
            if not save_path.exists():
                cropped.save(save_path, quality=95)

            # Step-0: predict first click (IOU=0, no mask)
            sample = build_step0_sample(
                str(save_path), description, click_seq[0], crop_box
            )
            if sample:
                samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="data/affordance/vlm_traj_data/cropped_images",
    )
    parser.add_argument(
        "--output_jsonl",
        default="data/affordance/vlm_traj_data/train.jsonl",
    )
    parser.add_argument("--pad_ratio", type=float, default=0.2)
    parser.add_argument(
        "--object_dirs", nargs="+", default=["object_1", "object_2"]
    )
    args = parser.parse_args()

    output_img_dir = Path(args.output_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Objects365 v1 image ID map...")
    v1_json = (
        Path("data/Objects365_v1/OpenDataLab___Objects365_v1")
        / "raw/Objects365_v1/2019-08-02/objects365_train.json"
    )
    v1_id_map = load_v1_id_to_filename(v1_json)
    print(f"  Loaded {len(v1_id_map)} v1 image entries")

    ann_paths = []
    for obj_dir in args.object_dirs:
        ann_paths.extend(sorted((DATA4SAM3_ROOT / obj_dir).rglob("annotation.json")))
    print(f"Found {len(ann_paths)} annotation files")

    total, skipped = 0, 0
    with open(output_jsonl, "w") as out_f:
        for i, ann_path in enumerate(ann_paths):
            samples = process_annotation(ann_path, v1_id_map, output_img_dir, args.pad_ratio)
            if not samples:
                skipped += 1
            for s in samples:
                out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
                total += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(ann_paths)}] samples: {total}, skipped: {skipped}")

    print(f"\nDone. Total samples: {total}, skipped: {skipped}")
    print(f"JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
