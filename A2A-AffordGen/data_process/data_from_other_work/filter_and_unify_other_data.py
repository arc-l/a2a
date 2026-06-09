#!/usr/bin/env python3
"""
filter_and_unify_other_data.py

For each RAGNet-style (image, mask, object_class) sample:
  1. Filter and unify masks to 0/1 uint8:
       - target_mask_value samples use m == target_mask_value
       - binary masks use m > 0
       - other grayscale masks use m >= 128
       - strict datasets such as GraspNet only keep clean 0/255 masks
  2. Call local Qwen3-VL via vLLM (OpenAI-compatible API) to identify:
       - which specific part of the object the mask covers
       - up to 5 affordance task descriptions
  3. Save binary mask PNG + structured JSON records

Output layout:
  <output_dir>/
    masks/
      <stem>_<idx>.png     <- binary 0/1 uint8 PNG
    results.json           <- list of all records

Start vLLM server first:
  python -m vllm.entrypoints.openai.api_server \\
      --model models/Qwen3-VL-32B-Instruct \\
      --served-model-name qwen3-vl-32b \\
      --tensor-parallel-size 1 \\
      --gpu-memory-utilization 0.85 \\
      --max-model-len 4096 \\
      --limit-mm-per-prompt image=1 \\
      --port 8000

Usage:
  python filter_and_unify_other_data.py \\
      --pkl /path/to/egoobjects_train.pkl \\
      --base-dir data/affordance/RAGNet/data \\
      --output-dir ./output_egoobjects \\
      [--dataset egoobjects] \\
      [--start 0] [--end 500] \\
      [--server-url http://localhost:8000] \\
      [--model-name qwen3-vl-32b]
"""

import argparse
import base64
import json
import os
import pickle
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from PIL import Image
from tqdm import tqdm


# ── constants ─────────────────────────────────────────────────────────────────

MASK_OVERLAY_ALPHA = 0.5
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_MODEL_NAME = "qwen3-vl-32b"

SYSTEM_PROMPT = (
    "You are a precise robotic vision assistant. "
    "Given an image with a semi-transparent green mask highlighting part of an object, "
    "you identify the exact object part and relevant manipulation tasks."
)

USER_PROMPT_TEMPLATE = """\
The image shows a **{object_class}**. A semi-transparent green mask highlights one specific part of this object.

Please analyze the masked region carefully and respond with a JSON object (no markdown, raw JSON only):

{{
  "object_name": "<the object category, e.g. scissors>",
  "part_name": "<specific part highlighted by the mask, e.g. blade of the scissors>",
  "affordance": "<single verb describing the primary action, e.g. cut>",
  "tasks": [
    "<concrete task 1, e.g. cut the apple>",
    "<concrete task 2>",
    "<concrete task 3>",
    "<concrete task 4>",
    "<concrete task 5>"
  ]
}}

Rules:
- part_name must be specific and descriptive (not just the object name).
- tasks must be concrete action sentences (verb + object), max 5, at least 1.
- affordance is a single verb (cut, grasp, pour, press, etc.).
- Respond with raw JSON only, no extra text.
"""


# -- mask utilities ------------------------------------------------------------

def unify_binary_mask(
    mask_arr: np.ndarray,
    *,
    dataset_name: str = "",
    target_mask_value: int | None = None,
    strict_binary_datasets: set[str] | None = None,
    threshold: int = 128,
) -> tuple[np.ndarray | None, str | None, list[int], str]:
    """
    Return a uint8 HxW mask in {0, 1}, plus error/method metadata.

    This mirrors the trajectory-generation handling:
    - If target_mask_value is provided, select exactly that class id.
    - For strict datasets (GraspNet variants by default), only keep 0/255 masks.
    - Otherwise keep binary masks as m > 0, and grayscale/soft masks as m >= 128.
    """
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    if mask_arr.ndim != 2:
        return None, "bad_mask_ndim", [], ""

    unique_values = np.unique(mask_arr)
    unique_list = [int(x) for x in unique_values.tolist()]

    if len(unique_values) == 0 or int(unique_values.max()) == 0:
        return None, "empty_mask", unique_list, ""

    if target_mask_value is not None:
        binary = (mask_arr == int(target_mask_value)).astype(np.uint8)
        method = f"class_id={int(target_mask_value)}"
        if int(binary.sum()) == 0:
            return None, "class_id_not_found", unique_list, method
        return binary, None, unique_list, method

    u = set(unique_list)
    strict_binary_datasets = strict_binary_datasets or set()
    if dataset_name in strict_binary_datasets and not u.issubset({0, 255}):
        return None, "non_0_255_mask_values_for_strict_dataset", unique_list, ""

    if u.issubset({0, 1}) or u.issubset({0, 255}) or u.issubset({0, 1, 255}):
        binary = (mask_arr > 0).astype(np.uint8)
        method = "binary_nonzero"
    else:
        if int(unique_values.max()) < int(threshold):
            return None, "all_values_below_threshold", unique_list, f"threshold>={threshold}"
        binary = (mask_arr >= int(threshold)).astype(np.uint8)
        method = f"threshold>={threshold}"

    if int(binary.sum()) == 0:
        return None, "empty_mask_after_unify", unique_list, method

    return binary, None, unique_list, method


def overlay_mask_on_image(image_rgb: np.ndarray, binary_mask: np.ndarray,
                          alpha: float = MASK_OVERLAY_ALPHA) -> np.ndarray:
    """Return RGB image with semi-transparent green overlay where mask==1."""
    out = image_rgb.astype(np.float32)
    where = binary_mask > 0
    out[where, 0] = out[where, 0] * (1 - alpha)
    out[where, 1] = out[where, 1] * (1 - alpha) + 255 * alpha
    out[where, 2] = out[where, 2] * (1 - alpha)
    return out.astype(np.uint8)


def encode_image_rgb_to_b64(image_rgb: np.ndarray, max_size: int = 768) -> str:
    """Encode an RGB numpy image to base64 JPEG string, capped at max_size on longest side."""
    h, w = image_rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image")
    return base64.standard_b64encode(buf.tobytes()).decode("ascii")


# ── data loading ───────────────────────────────────────────────────────────────

def load_samples(pkl_path: str, base_dir: str) -> list[dict]:
    """
    Load a RAGNet-style pkl and return a flat list of dicts:
      { frame_path: str, mask_path: str, task_object_class: str }
    Both paths are resolved to absolute paths.
    """
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    base = Path(base_dir).resolve()

    def resolve(p: str) -> str:
        raw = Path(str(p).strip())
        candidates = []
        if raw.is_absolute():
            candidates.append(raw)
            parts = raw.parts
            if "RAGNet" in parts:
                candidates.append(base / Path(*parts[parts.index("RAGNet") + 1:]))
            elif "data" in parts:
                candidates.append(base / Path(*parts[parts.index("data"):]))
        else:
            # strip leading "./data/" or "data/" prefix used in RAGNet pkls
            p_rel = re.sub(r"^\./data/", "", str(p))
            p_rel = re.sub(r"^data/", "", p_rel)
            candidates.append(base / Path(p_rel))

        # HANDAL: strip "without_depth" directory level
        expanded = list(candidates)
        for c in candidates:
            parts = c.parts
            if "HANDAL" in parts and "without_depth" in parts:
                expanded.append(Path(*[pt for pt in parts if pt != "without_depth"]))

        for c in expanded:
            if c.exists():
                return str(c)
        return str(expanded[-1])

    samples = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) in {3, 4}:
                fp, mp, cls = item[:3]
                target_mask_value = int(item[3]) if len(item) == 4 and item[3] is not None else None
            elif isinstance(item, dict):
                fp = item.get("frame_path") or item.get("image_path", "")
                mp = item.get("mask_path", "")
                cls = item.get("task_object_class") or item.get("object_class", "object")
                target_mask_value = item.get("target_mask_value", None)
                target_mask_value = int(target_mask_value) if target_mask_value is not None else None
            else:
                continue
            if fp and mp:
                samples.append({
                    "frame_path": resolve(fp),
                    "mask_path":  resolve(mp),
                    "task_object_class": str(cls),
                    "target_mask_value": target_mask_value,
                })
    elif isinstance(raw, dict):
        images      = raw.get("images", {})
        labels      = raw.get("labels", {})
        class_names = raw.get("class_names", {})

        if isinstance(images, dict):
            # GraspNet style: images/labels/class_names are dict[class_name → list[path]]
            for cls, img_list in images.items():
                lbl_list = labels.get(cls, [])
                cls_list = class_names.get(cls, [])
                for i, (fp, mp) in enumerate(zip(img_list, lbl_list)):
                    fp = str(fp); mp = str(mp)
                    name = cls_list[i] if i < len(cls_list) else cls
                    samples.append({
                        "frame_path":        resolve(fp),
                        "mask_path":         resolve(mp),
                        "task_object_class": str(name),
                        "target_mask_value": None,
                    })
        else:
            # HANDAL style: images/labels are flat lists
            for img_entry, lbl_entry in zip(images, labels):
                fp  = img_entry if isinstance(img_entry, str) else str(img_entry)
                mp  = lbl_entry if isinstance(lbl_entry, str) else str(lbl_entry)
                cls = Path(fp).parent.name
                samples.append({
                    "frame_path":        resolve(fp),
                    "mask_path":         resolve(mp),
                    "task_object_class": str(cls),
                    "target_mask_value": None,
                })
    else:
        raise ValueError(f"Unsupported pkl format: {type(raw)}")

    return samples


# ── directory-based data loading (3DOI / no-pkl datasets) ─────────────────────

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg"}

def load_samples_from_3doi_dir(dataset_root: str) -> list[dict]:
    """
    Load 3DOI-style dataset from directory structure:
      <root>/images/**/*.<ext>
      <root>/masks/<object_name>/**/*.<ext>

    object_name is taken from the first subdirectory level under masks/.
    Masks are matched to images by filename stem (case-insensitive).
    Returns list of { frame_path, mask_path, task_object_class, target_mask_value }.
    """
    root      = Path(dataset_root).resolve()
    images_dir = root / "images"
    masks_dir  = root / "masks"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"masks dir not found: {masks_dir}")

    # build stem → image path index (case-insensitive, pick path-lex-smallest on collision)
    img_by_stem: dict[str, Path] = {}
    for p in images_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            key = p.stem.lower()
            if key not in img_by_stem or str(p) < str(img_by_stem[key]):
                img_by_stem[key] = p

    def _match_image(mask_path: Path) -> Path | None:
        stem = mask_path.stem.lower()
        if stem in img_by_stem:
            return img_by_stem[stem]
        # try stripping common mask suffixes
        stem2 = re.sub(r"(_mask|_m|_seg|_label|-mask|-seg|-label)$", "", stem)
        return img_by_stem.get(stem2)

    samples: list[dict] = []
    missing = 0
    for mask_path in masks_dir.rglob("*"):
        if not (mask_path.is_file() and mask_path.suffix.lower() in MASK_EXTS):
            continue
        rel = mask_path.relative_to(masks_dir)
        if len(rel.parts) < 2:
            continue  # skip stray files directly under masks/
        object_name = rel.parts[0]
        img_path = _match_image(mask_path)
        if img_path is None:
            missing += 1
            continue
        samples.append({
            "frame_path":        str(img_path),
            "mask_path":         str(mask_path),
            "task_object_class": object_name,
            "target_mask_value": None,
        })

    samples.sort(key=lambda s: (Path(s["frame_path"]).name.lower(), s["mask_path"]))
    if missing:
        print(f"[WARN] {missing} mask(s) could not be matched to any image.")
    return samples


# ── InstructPart loader ────────────────────────────────────────────────────────

def load_samples_from_instructpart(dataset_root: str) -> list[dict]:
    """
    Load InstructPart from data_all.json into the standard samples list.
    part_name is pre-labeled in JSON and stored as 'known_part' so
    process_sample can override VLM's part_name with it.

    Directory layout:
      <root>/data_all.json
      <root>/images/<image_filename>
      <root>/masks/<image_stem>-<object>-<part>.png
    """
    root          = Path(dataset_root).resolve()
    images_dir    = root / "images"
    src_masks_dir = root / "masks"

    with open(root / "data_all.json") as f:
        entries = json.load(f)

    samples: list[dict] = []
    missing = 0
    for entry in entries:
        img_filename = entry["image_path"]
        img_stem     = Path(img_filename).stem
        img_ext      = Path(img_filename).suffix  # usually .jpg

        for part_info in entry.get("part_list", []):
            obj_name  = part_info.get("object", "object")
            part_name = part_info.get("part", "part")
            # filenames use underscores instead of spaces
            obj_slug  = obj_name.replace(" ", "_")
            part_slug = part_name.replace(" ", "_")
            # images are stored per-part: <stem>-<object>-<part>.<ext>
            img_path  = images_dir / f"{img_stem}-{obj_slug}-{part_slug}{img_ext}"
            mask_path = src_masks_dir / f"{img_stem}-{obj_slug}-{part_slug}.png"
            if not img_path.exists():
                missing += 1
                continue
            if not mask_path.exists():
                missing += 1
                continue
            samples.append({
                "frame_path":        str(img_path),
                "mask_path":         str(mask_path),
                "task_object_class": obj_name,
                "target_mask_value": None,
                "known_part":        part_name,
            })

    if missing:
        print(f"[WARN] {missing} InstructPart mask(s) not found.")
    return samples


# ── VLM analysis ───────────────────────────────────────────────────────────────

def parse_vlm_response(text: str) -> dict | None:
    """Extract JSON from VLM response, tolerating markdown fences."""
    text = text.strip()
    # strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def analyze_with_vlm(image_b64: str, object_class: str,
                     client: OpenAI, model_name: str) -> dict:
    """
    Send the masked image to local Qwen3-VL via vLLM and return parsed JSON
    with keys: object_name, part_name, affordance, tasks.
    Falls back to a placeholder on failure.
    """
    prompt = USER_PROMPT_TEMPLATE.format(object_class=object_class)

    response = client.chat.completions.create(
        model=model_name,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
    )

    raw_text = response.choices[0].message.content
    parsed = parse_vlm_response(raw_text)

    if parsed is None:
        return {
            "object_name": object_class,
            "part_name":   object_class,
            "affordance":  "use",
            "tasks":       [f"use the {object_class}"],
            "_vlm_raw":    raw_text,
            "_parse_error": True,
        }

    # normalise tasks to list, cap at 5
    tasks = parsed.get("tasks", [])
    if isinstance(tasks, str):
        tasks = [tasks]
    tasks = [str(t) for t in tasks[:5]]

    return {
        "object_name": parsed.get("object_name", object_class),
        "part_name":   parsed.get("part_name",   object_class),
        "affordance":  parsed.get("affordance",  "use"),
        "tasks":       tasks,
    }


# ── per-sample processing ──────────────────────────────────────────────────────

def process_sample(
    sample: dict,
    idx: int,
    masks_dir: Path,
    client: OpenAI,
    model_name: str,
    dataset_name: str,
    strict_binary_datasets: set[str],
    mask_threshold: int,
    part_from_filename: bool = False,
) -> dict | None:
    """
    Full pipeline for one sample. Returns a record dict or None on failure.
    """
    frame_path = sample["frame_path"]
    mask_path  = sample["mask_path"]
    obj_class  = sample["task_object_class"]
    target_mask_value = sample.get("target_mask_value", None)

    # ── load image ──────────────────────────────────────────────────────────
    if not os.path.exists(frame_path):
        print(f"[SKIP] image not found: {frame_path}")
        return None
    image_bgr = cv2.imread(frame_path)
    if image_bgr is None:
        print(f"[SKIP] cannot read image: {frame_path}")
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # -- load & unify mask ---------------------------------------------------
    if not os.path.exists(mask_path):
        print(f"[SKIP] mask not found: {mask_path}")
        return None
    mask_arr = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_arr is None:
        print(f"[SKIP] cannot read mask: {mask_path}")
        return None
    binary, mask_error, unique_values, unify_method = unify_binary_mask(
        mask_arr,
        dataset_name=dataset_name,
        target_mask_value=target_mask_value,
        strict_binary_datasets=strict_binary_datasets,
        threshold=mask_threshold,
    )

    if mask_error == "non_0_255_mask_values_for_strict_dataset":
        print(f"[SKIP] {dataset_name} non-0/255 mask values={unique_values}: {mask_path}")
        return None
    if mask_error == "bad_mask_ndim":
        print(f"[SKIP] unsupported mask ndim={mask_arr.ndim}: {mask_path}")
        return None
    if mask_error == "empty_mask":
        print(f"[SKIP] empty mask unique={unique_values}: {mask_path}")
        return None
    if mask_error == "class_id_not_found":
        print(f"[SKIP] class_id={target_mask_value} not found unique={unique_values}: {mask_path}")
        return None
    if mask_error == "all_values_below_threshold":
        print(f"[SKIP] all mask values below threshold({mask_threshold}) unique={unique_values}: {mask_path}")
        return None
    if mask_error == "empty_mask_after_unify":
        print(f"[SKIP] empty mask after unify method={unify_method} unique={unique_values}: {mask_path}")
        return None
    if binary is None:
        print(f"[SKIP] invalid mask ({mask_error}): {mask_path}")
        return None

    # skip if mask size doesn't match image
    h, w = image_rgb.shape[:2]
    if binary.shape != (h, w):
        print(f"[SKIP] mask/image size mismatch: mask={binary.shape} image=({h},{w}): {mask_path}")
        return None

    # ── save binary mask ─────────────────────────────────────────────────────
    stem     = Path(frame_path).stem
    mask_out = masks_dir / f"{stem}_{idx:05d}.png"
    cv2.imwrite(str(mask_out), binary)   # save as real 0/1 uint8

    # ── VLM call (always) + optional filename-based part override ───────────
    overlaid  = overlay_mask_on_image(image_rgb, binary)
    image_b64 = encode_image_rgb_to_b64(overlaid)
    try:
        vlm_result = analyze_with_vlm(image_b64, obj_class, client, model_name)
    except Exception as e:
        print(f"[WARN] VLM error for {stem}: {e}")
        vlm_result = {
            "object_name": obj_class,
            "part_name":   obj_class,
            "affordance":  "use",
            "tasks":       [f"use the {obj_class}"],
            "_error":      str(e),
        }

    # override part_name if pre-labeled (HANDAL filename / InstructPart JSON)
    if part_from_filename:
        vlm_result["part_name"] = Path(mask_path).stem.split("_")[-1].replace("-", " ")
    known_part = sample.get("known_part")
    if known_part:
        vlm_result["part_name"] = known_part

    record = {
        "image_path":   frame_path,
        "image_name":   stem,
        "mask_path":    str(mask_out),
        "mask_values":  [0, 1],
        "source_mask_unique_values": unique_values,
        "mask_unify_method": unify_method,
        "target_mask_value": target_mask_value,
        "object_name":  vlm_result["object_name"],
        "part_name":    vlm_result["part_name"],
        "affordance":   vlm_result["affordance"],
        "tasks":        vlm_result["tasks"],
    }
    if "_vlm_raw" in vlm_result:
        record["_vlm_raw"]     = vlm_result["_vlm_raw"]
        record["_parse_error"] = vlm_result.get("_parse_error", False)
    if "_error" in vlm_result:
        record["_error"] = vlm_result["_error"]

    return record


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Filter, binarize masks and generate part/affordance annotations via VLM."
    )
    parser.add_argument("--pkl", default=None,
                        help="Path to RAGNet-style .pkl file (mutually exclusive with --dataset-root).")
    parser.add_argument("--dataset-root", default=None,
                        help="Root dir for directory-based datasets like 3DOI (images/ + masks/<obj>/).")
    parser.add_argument("--instructpart-root", default=None,
                        help="Root dir for InstructPart (data_all.json + images/ + masks/). Skips VLM entirely.")
    parser.add_argument("--base-dir",
                        default="data/affordance/RAGNet/data",
                        help="Base directory for resolving relative paths in pkl.")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory (masks/ sub-dir + results.json).")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name tag written into each record (optional).")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (inclusive).")
    parser.add_argument("--end", type=int, default=-1,
                        help="End index (exclusive). -1 means all.")
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL,
                        help=f"vLLM server URL (default: {DEFAULT_SERVER_URL}).")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help=f"Model name served by vLLM (default: {DEFAULT_MODEL_NAME}).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples whose mask PNG already exists.")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel VLM request workers (default: 8).")
    parser.add_argument("--mask-threshold", type=int, default=128,
                        help="Threshold used for non-binary grayscale masks outside strict datasets.")
    parser.add_argument("--part-from-filename", action="store_true",
                        help="Extract part name from mask filename (last _-separated token). "
                             "Skips VLM call. Use for HANDAL-style datasets.")
    parser.add_argument(
        "--strict-binary-datasets",
        default="graspnet,graspnet_train,graspnet_test_seen,graspnet_test_seen_val,graspnet_test_novel,graspnet_test_novel_val",
        help="Comma-separated dataset names that must use clean 0/255 masks.",
    )
    args = parser.parse_args()
    sources = [args.pkl, args.dataset_root, args.instructpart_root]
    if sum(s is not None for s in sources) == 0:
        parser.error("One of --pkl, --dataset-root, or --instructpart-root is required.")
    if sum(s is not None for s in sources) > 1:
        parser.error("--pkl, --dataset-root, and --instructpart-root are mutually exclusive.")

    # ── setup ────────────────────────────────────────────────────────────────
    client = OpenAI(
        base_url=f"{args.server_url}/v1",
        api_key="EMPTY",   # vLLM does not require a real key
    )
    model_name = args.model_name
    output_dir = Path(args.output_dir)
    masks_dir  = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "results.json"
    existing_records: list[dict] = []
    if args.resume and results_path.exists():
        with open(results_path) as f:
            existing_records = json.load(f)
        existing_stems = {r["image_name"] + f"_{r.get('_idx', '')}": True
                          for r in existing_records}
        print(f"[RESUME] loaded {len(existing_records)} existing records.")
    else:
        existing_stems = {}

    # ── load samples ─────────────────────────────────────────────────────────
    if args.instructpart_root:
        print(f"Loading InstructPart: {args.instructpart_root}")
        samples = load_samples_from_instructpart(args.instructpart_root)
        dataset_tag = args.dataset or "instructpart"
    elif args.pkl:
        print(f"Loading pkl: {args.pkl}")
        samples = load_samples(args.pkl, args.base_dir)
        dataset_tag = args.dataset or Path(args.pkl).stem
    else:
        print(f"Loading from directory: {args.dataset_root}")
        samples = load_samples_from_3doi_dir(args.dataset_root)
        dataset_tag = args.dataset or Path(args.dataset_root).name
    print(f"Total samples: {len(samples)}")

    end = len(samples) if args.end == -1 else args.end
    samples = samples[args.start:end]
    print(f"Processing: [{args.start}, {end}) → {len(samples)} samples  (workers={args.workers})")
    strict_binary_datasets = {
        x.strip() for x in args.strict_binary_datasets.split(",") if x.strip()
    }

    # ── filter already-done samples for resume ───────────────────────────────
    todo: list[tuple[int, dict]] = []   # (global_idx, sample)
    for i, sample in enumerate(samples):
        gidx = args.start + i
        stem = Path(sample["frame_path"]).stem
        key  = f"{stem}_{gidx:05d}"
        if args.resume and key in existing_stems:
            continue
        todo.append((gidx, sample))
    print(f"To process: {len(todo)} samples (skipped {len(samples) - len(todo)} resumed)")

    # ── parallel processing ──────────────────────────────────────────────────
    records     = list(existing_records)
    save_lock   = threading.Lock()
    save_counter = [0]   # mutable counter shared across threads

    def _save_incremental():
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

    def _worker(args_tuple):
        gidx, sample = args_tuple
        record = process_sample(
            sample,
            gidx,
            masks_dir,
            client,
            model_name,
            dataset_tag,
            strict_binary_datasets,
            args.mask_threshold,
            part_from_filename=args.part_from_filename,
        )
        if record is None:
            return None
        record["_idx"]     = gidx
        record["_dataset"] = dataset_tag
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            record = future.result()
            if record is None:
                continue
            with save_lock:
                records.append(record)
                save_counter[0] += 1
                if save_counter[0] % 10 == 0:
                    _save_incremental()

    # ── final save ───────────────────────────────────────────────────────────
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    ok    = sum(1 for r in records if "_error" not in r and not r.get("_parse_error"))
    warn  = len(records) - ok
    print(f"\n✓ Done.  {ok} OK  |  {warn} with warnings/errors")
    print(f"  Masks   → {masks_dir}/")
    print(f"  Results → {results_path}")


if __name__ == "__main__":
    main()
