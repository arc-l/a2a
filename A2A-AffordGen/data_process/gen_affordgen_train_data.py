"""
Generate AffordGen VLM fine-tuning data from filtered trajectory JSONs.

For each click step k in each trajectory:
  - Step 0 : original image (no mask overlay)
  - Step k>0 : original image + green semi-transparent overlay of step k-1 mask

Prompt / answer format (Swift JSONL, Qwen3.5-compatible):
  User    : <image>
             Please optimize the semi-transparent green mask in a point-wise
             manner based on the image and description content, so that it
             covers the target object as accurately as possible.
             The object description is as follows: <ref>{description}</ref>
  Answer  : <ref>{description}</ref> Current IOU: {current_iou:.2f},
             {Positive/Negative} point: ({x}, {y}),
             Predicted next IOU: {next_iou:.2f}

  where x = int(col/W*1000), y = int(row/H*1000)  (both in [0, 1000])
  current_iou = IoU of previous mask (0.0 for step 0)
  next_iou    = IoU achieved by this click

Usage:
    python data_process/gen_affordgen_train_data.py \\
        --input_dir  dataset/trajs_dataset_after_filter \\
        --output_dir data/affordance/affordgen_train \\
        --output_jsonl data/affordance/affordgen_train/train.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
from PIL import Image


# ── Prompt templates ─────────────────────────────────────────────────────────
USER_PROMPT = (
    "<image>\n"
    "Please optimize the semi-transparent green mask in a point-wise manner "
    "based on the image and description content, so that it covers the target "
    "object as accurately as possible. "
    "The object description is as follows: <ref>{description}</ref>"
)

ANSWER_TEMPLATE = (
    "<ref>{description}</ref> Current IOU: {current_iou:.2f}, "
    "{click_type} point: ({x}, {y}), "
    "Predicted next IOU: {next_iou:.2f}"
)

# Green overlay color and alpha
OVERLAY_COLOR = (0, 255, 0)   # RGB green
OVERLAY_ALPHA = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_rle(rle: dict) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode()
    return mask_util.decode({"counts": counts, "size": rle["size"]})


def apply_green_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img = np.array(image, dtype=np.float32)
    green = np.zeros_like(img)
    green[..., 0] = OVERLAY_COLOR[0]
    green[..., 1] = OVERLAY_COLOR[1]
    green[..., 2] = OVERLAY_COLOR[2]
    m = mask.astype(bool)[..., None]
    blended = np.where(m, img * (1 - OVERLAY_ALPHA) + green * OVERLAY_ALPHA, img)
    return Image.fromarray(blended.astype(np.uint8))


def build_sample(img_path: str, description: str, current_iou: float,
                 next_iou: float, click: dict, H: int, W: int) -> dict:
    row, col = click["coor"]
    x = int(col / W * 1000)
    y = int(row / H * 1000)
    click_type = "Positive" if click["is_positive"] else "Negative"
    answer = ANSWER_TEMPLATE.format(
        description=description,
        current_iou=current_iou,
        click_type=click_type,
        x=x,
        y=y,
        next_iou=next_iou,
    )
    return {
        "messages": [
            {"role": "user",    "content": USER_PROMPT.format(description=description)},
            {"role": "assistant", "content": answer},
        ],
        "images": [img_path],
    }


# ── Per-item processing ───────────────────────────────────────────────────────
def get_caption(gt_ann: dict) -> str:
    """Build caption: use existing caption, or compose from part+object fields."""
    caption = gt_ann.get("caption", "")
    if caption:
        return caption
    part = gt_ann.get("part", "")
    obj  = gt_ann.get("object", "")
    if part and obj:
        return f"{part} of the {obj}"
    return obj or part or "the target object"


def dataset_name_from_path(json_path: Path) -> str:
    stem = json_path.stem
    if "_testA" in stem:
        return stem.split("_testA")[0]
    return stem


def process_item(item: dict, output_img_dir: Path, img_base: Path = None) -> list[dict]:
    img_path = Path(item["image_name"])
    if not img_path.exists():
        return []

    H, W = item["height"], item["width"]
    description = get_caption(item["gt_ann"])
    clicks = item["clicks_list"]
    if not clicks:
        return []

    try:
        orig_img = Image.open(img_path).convert("RGB")
    except Exception:
        return []

    # If img_base is given, mirror source dir structure under output_img_dir
    if img_base is not None:
        try:
            rel = img_path.parent.relative_to(img_base)
            item_overlay_dir = output_img_dir / rel
        except ValueError:
            item_overlay_dir = output_img_dir
        item_overlay_dir.mkdir(parents=True, exist_ok=True)
    else:
        item_overlay_dir = output_img_dir

    # Use inst_key (if present) for unique overlay naming across shared crop images
    inst_key = item["gt_ann"].get("inst_key", "")
    img_stem = img_path.stem + (f"__{inst_key}" if inst_key else "")
    samples = []

    for k, click in enumerate(clicks):
        if k == 0:
            step_img_path = str(img_path)
        else:
            prev_mask = decode_rle(clicks[k - 1]["mask"])
            overlay_img = apply_green_overlay(orig_img, prev_mask)
            save_name = f"{img_stem}__step{k}.jpg"
            save_path = item_overlay_dir / save_name
            if not save_path.exists():
                overlay_img.save(save_path, quality=95)
            step_img_path = str(save_path)

        current_iou = 0.0 if k == 0 else clicks[k - 1]["iou"]
        next_iou = click["iou"]
        sample = build_sample(step_img_path, description, current_iou, next_iou, click, H, W)
        samples.append(sample)

    return samples


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",    default="data/affordance/trajs_with_captions",
                    help="directory of single-object JSON files (trajs_with_captions)")
    ap.add_argument("--extra_jsons",  nargs="*", default=[],
                    help="additional single JSON files (e.g. filtered multi-object traj)")
    ap.add_argument("--extra_img_base", default=None,
                    help="image root for extra_jsons; overlays mirror its subdir structure")
    ap.add_argument("--extra_overlay_prefix", default=None,
                    help="prefix subdir under overlay_images for extra_jsons (e.g. affordance2act)")
    ap.add_argument("--output_dir",   default="data/affordance/affordgen_train")
    ap.add_argument("--output_jsonl", default="data/affordance/affordgen_train/train.jsonl")
    args = ap.parse_args()

    overlay_root = Path(args.output_dir) / "overlay_images"
    overlay_root.mkdir(parents=True, exist_ok=True)
    output_jsonl = Path(args.output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Collect all JSON sources
    json_files = sorted(Path(args.input_dir).glob("*.json"))
    extra_files = [Path(p) for p in args.extra_jsons]
    all_files = json_files + extra_files
    print(f"Input: {len(json_files)} files from dir + {len(extra_files)} extra JSONs")

    extra_img_base = Path(args.extra_img_base) if args.extra_img_base else None
    extra_overlay_prefix = args.extra_overlay_prefix  # e.g. "affordance2act"

    total, skipped = 0, 0
    with open(output_jsonl, "w") as out_f:
        for json_file in all_files:
            is_extra = json_file in extra_files

            if is_extra and extra_overlay_prefix:
                output_img_dir = overlay_root / extra_overlay_prefix
                img_base = extra_img_base
                label = extra_overlay_prefix
            else:
                dataset_name = dataset_name_from_path(json_file)
                output_img_dir = overlay_root / dataset_name
                img_base = None
                label = dataset_name
            output_img_dir.mkdir(parents=True, exist_ok=True)

            with open(json_file) as f:
                data = json.load(f)
            items = data["data"]
            file_total = 0
            for item in items:
                samples = process_item(item, output_img_dir, img_base=img_base)
                if not samples:
                    skipped += 1
                for s in samples:
                    out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
                    file_total += 1
            total += file_total
            print(f"  {json_file.name} [{label}]: {file_total} samples")

    print(f"\nDone. Total samples: {total}, skipped items: {skipped}")
    print(f"Overlay images: {overlay_root}")
    print(f"JSONL: {output_jsonl}")


if __name__ == "__main__":
    main()
