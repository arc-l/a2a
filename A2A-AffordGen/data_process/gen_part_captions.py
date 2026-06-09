"""
Generate part-level captions for trajectory datasets using Qwen3-VL-32B.

For each data item (except instructpart which already has part captions):
  1. Render the GT mask as green overlay on the original image
  2. Send to Qwen3-VL-32B: "What part of the {object} is highlighted?"
  3. Write back the new caption into the JSON

Runs with multiple worker processes for speed.

Usage:
    python data_process/gen_part_captions.py \
        --input_dir  dataset/trajs_dataset_after_filter \
        --output_dir data/affordance/trajs_with_captions \
        --workers 4 \
        --api_url http://localhost:8000/v1
"""

import argparse
import base64
import json
import re
import sys
import time
from copy import deepcopy
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import requests
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
API_URL      = "http://localhost:8000/v1/chat/completions"
MODEL_NAME   = "qwen3-vl-32b"
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_ALPHA = 0.5
MAX_RETRIES   = 3
TIMEOUT       = 30  # seconds per request

SKIP_DATASETS = {"instructpart"}  # already have part-level captions

SYSTEM_PROMPT = (
    "You are a precise visual assistant. "
    "Answer with a short noun phrase only. No explanation, no punctuation at the end."
)

USER_PROMPT_TEMPLATE = (
    "The green highlighted region in this image marks a specific part of a {object}. "
    "Describe ONLY that highlighted part using the format: '<part> of the <object>'. "
    "Example answers: 'handle of the scissors', 'blade of the knife', 'rim of the cup'. "
    "Answer with the phrase only."
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def decode_rle(rle: dict) -> np.ndarray:
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode()
    return mask_util.decode({"counts": counts, "size": rle["size"]})


def apply_green_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img = np.array(image, dtype=np.float32)
    green = np.zeros_like(img)
    green[..., 1] = 255
    m = mask.astype(bool)[..., None]
    blended = np.where(m, img * (1 - OVERLAY_ALPHA) + green * OVERLAY_ALPHA, img)
    return Image.fromarray(blended.astype(np.uint8))


def image_to_b64(img: Image.Image, max_size: int = 768) -> str:
    # Resize long edge to max_size to keep prompt tokens manageable
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def call_qwen(b64_image: str, object_name: str) -> str | None:
    payload = {
        "model": MODEL_NAME,
        "max_tokens": 32,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                    {
                        "type": "text",
                        "text": USER_PROMPT_TEMPLATE.format(object=object_name),
                    },
                ],
            },
        ],
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            # Strip thinking tags if Qwen3 thinking mode is on
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            return text or None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2)
            else:
                return None


# ── Per-item processing ───────────────────────────────────────────────────────
def process_item(item: dict) -> str:
    """Returns the new caption, or original if generation fails."""
    original_caption = item["gt_ann"].get("caption", "")
    img_path = Path(item["image_name"])
    if not img_path.exists():
        return original_caption

    # Extract object name: last word(s) of original caption or full caption
    object_name = original_caption.strip()

    # Decode GT mask
    rle = item["gt_ann"]["segmentation"]
    try:
        mask = decode_rle(rle)
    except Exception:
        return original_caption

    try:
        orig = Image.open(img_path).convert("RGB")
    except Exception:
        return original_caption

    overlay = apply_green_overlay(orig, mask)
    b64 = image_to_b64(overlay)

    result = call_qwen(b64, object_name)
    if result is None:
        return original_caption

    # Sanity check: must contain "of the"
    if "of the" not in result.lower():
        # Wrap it ourselves: "handle" → "handle of the mug"
        result = f"{result} of the {object_name}"

    return result


# ── Worker (runs in subprocess) ───────────────────────────────────────────────
def worker_process_file(args_tuple):
    json_path, output_dir, worker_id = args_tuple
    json_path = Path(json_path)
    output_dir = Path(output_dir)

    with open(json_path) as f:
        data = json.load(f)

    items = data["data"]
    updated = 0

    for i, item in enumerate(items):
        new_caption = process_item(item)
        old_caption = item["gt_ann"].get("caption", "")
        if new_caption != old_caption:
            item["gt_ann"]["caption"] = new_caption
            item["gt_ann"]["original_caption"] = old_caption
            updated += 1

        if (i + 1) % 100 == 0:
            print(f"[worker {worker_id}] {json_path.name}: {i+1}/{len(items)} done, {updated} updated",
                  flush=True)

    out_path = output_dir / json_path.name
    with open(out_path, "w") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"[worker {worker_id}] DONE {json_path.name}: {updated}/{len(items)} captions updated → {out_path}",
          flush=True)
    return str(json_path.name), updated, len(items)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir",  default="dataset/trajs_dataset_after_filter")
    ap.add_argument("--output_dir", default="data/affordance/trajs_with_captions")
    ap.add_argument("--workers",    type=int, default=4)
    ap.add_argument("--api_url",    default="http://localhost:8000/v1/chat/completions")
    args = ap.parse_args()

    global API_URL
    API_URL = args.api_url

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy instructpart directly without re-captioning
    all_files = sorted(Path(args.input_dir).glob("*.json"))
    skip_files, process_files = [], []
    for f in all_files:
        if any(s in f.name for s in SKIP_DATASETS):
            skip_files.append(f)
        else:
            process_files.append(f)

    for f in skip_files:
        import shutil
        shutil.copy(f, output_dir / f.name)
        print(f"[skip] copied {f.name} as-is")

    print(f"\nWill re-caption {len(process_files)} files with {args.workers} workers")
    print(f"API: {API_URL}\n")

    # Check API is alive
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        r.raise_for_status()
        print("API health check: OK\n")
    except Exception as e:
        print(f"ERROR: API not reachable: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = [(str(f), str(output_dir), i % args.workers)
             for i, f in enumerate(process_files)]

    with Pool(processes=args.workers) as pool:
        results = pool.map(worker_process_file, tasks)

    print("\n===== SUMMARY =====")
    total_updated = 0
    for name, updated, total in results:
        print(f"  {name}: {updated}/{total} captions updated")
        total_updated += updated
    print(f"Total updated: {total_updated}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
