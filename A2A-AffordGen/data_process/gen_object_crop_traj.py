"""
Generate click trajectories for affordance part masks, cropped to full object bbox.

Pipeline per instance:
  1. Find original image (Objects365 v1 or v2)
  2. Look up full object bbox from Objects365 index
  3. Crop + save the object image
  4. Map part mask (from data4sam3) into cropped coordinate space
  5. Run SAM3 click loop (monotone-increasing IoU stop, no threshold filter)
  6. Record trajectory in fliter_data.py-compatible JSON format

Output JSON schema (compatible with fliter_data.py):
  {
    "info": {"args": {...}},
    "data": [
      {
        "image_name": "/abs/path/to/cropped.jpg",   # <- cropped object image
        "height": H, "width": W,
        "gt_ann": {
          "caption": "surface of the desk",
          "segmentation": {...rle...},
          "bbox": [x,y,w,h],
          "area": N
        },
        "clicks_list": [
          {"idx": 0, "coor": [row, col], "is_positive": true,
           "mask": {...rle...}, "box": [...], "area": N, "iou": 0.45},
          ...
        ]
      }
    ]
  }

Usage:
    cd A2A-AffordGen
    python data_process/gen_object_crop_traj.py \\
        --sam3_ckpt models/sam3/sam3.pt \\
        --output_dir data/affordance/vlm_traj_data_v2 \\
        --output_json data/affordance/vlm_traj_data_v2/trajs_raw.json \\
        --gpus 0 \\
        --max_clicks 20 \\
        --bbox_pad 20 \\
        --start 0 --end -1
"""

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
import pycocotools.mask as mask_util

# ── Project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAM3_ROOT = PROJECT_ROOT / "sam3"                               # A2A-GroundingModel / SAM3-I repo (clone here or pip-install)
SIMPLECLICK_ROOT = PROJECT_ROOT / "third_party" / "SimpleClick" # clone SimpleClick here (see third_party/README.md)

for p in [str(PROJECT_ROOT), str(SAM3_ROOT), str(SIMPLECLICK_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from isegm.inference.clicker import Clicker

# ── Path constants ─────────────────────────────────────────────────────────────
DATA4SAM3_ROOT   = Path("data/affordance/affordance2act/data4sam3")
DATA4SAM3_MASK_ROOT = Path("data/affordance/affordance2act")
OBJ365_V1_TRAIN  = Path("data/Objects365_v1/OpenDataLab___Objects365_v1"
                         "/raw/Objects365_v1/2019-08-02/train")
OBJ365_V1_JSON   = Path("data/Objects365_v1/OpenDataLab___Objects365_v1"
                         "/raw/Objects365_v1/2019-08-02/objects365_train.json")
OBJ365_V2_DIR    = Path("data/affordance/assign_task/data_need_label")
V2_BBOX_INDEX    = Path("data/Objects365_v2/v2_bbox_index_compact.json")


# ── Objects365 indices ────────────────────────────────────────────────────────
class Obj365V1Index:
    def __init__(self, json_path: Path):
        print("Loading Objects365 v1 index ...")
        with open(json_path) as f:
            data = json.load(f)
        self._cat_id_to_name = {c["id"]: c["name"].lower() for c in data["categories"]}
        self._name_to_cat_ids: dict[str, set] = {}
        for cid, name in self._cat_id_to_name.items():
            self._name_to_cat_ids.setdefault(name, set()).add(cid)
        self._img_anns: dict[int, list] = {}
        for ann in data["annotations"]:
            self._img_anns.setdefault(ann["image_id"], []).append(ann)
        self._img_files = {img["id"]: img["file_name"] for img in data["images"]}
        print(f"  {len(self._img_files)} images loaded")

    def get_image_file(self, image_id: int) -> str | None:
        return self._img_files.get(image_id)

    def get_object_bbox(self, image_id: int, obj_name: str, obj_index: int) -> list | None:
        anns = self._img_anns.get(image_id, [])
        cat_ids = self._name_to_cat_ids.get(obj_name.lower(), set())
        if not cat_ids:
            obj_lower = obj_name.lower()
            cat_ids = {
                cid
                for cname, cid_set in self._name_to_cat_ids.items()
                if obj_lower in cname or cname in obj_lower
                for cid in cid_set
            }
        matching = [a for a in anns if a["category_id"] in cat_ids]
        if obj_index >= len(matching):
            return None
        x, y, w, h = matching[obj_index]["bbox"]
        return [float(x), float(y), float(x + w), float(y + h)]


class Obj365V2Index:
    def __init__(self, json_path: Path):
        if not json_path.exists():
            print(f"[WARN] v2 index not found: {json_path}. v2 will use part-bbox fallback.")
            self._data = {}
            return
        print(f"Loading Objects365 v2 compact index ...")
        with open(json_path) as f:
            self._data = json.load(f)
        print(f"  {len(self._data)} v2 images indexed")

    def get_object_bbox(self, img_stem: str, obj_name: str, obj_index: int) -> list | None:
        entries = self._data.get(img_stem, [])
        obj_lower = obj_name.lower()
        matching = [
            e for e in entries
            if obj_lower in e["category"].lower() or e["category"].lower() in obj_lower
        ]
        hits = [e for e in matching if e["cat_index"] == obj_index]
        if not hits:
            hits = sorted(matching, key=lambda e: abs(e["cat_index"] - obj_index))
        return hits[0]["bbox"] if hits else None


# ── Image resolution ──────────────────────────────────────────────────────────
def find_image_path(image_path_str: str, v1_index: Obj365V1Index):
    """Returns (abs_path, image_id_or_None, img_stem)."""
    filename = Path(image_path_str).name
    stem     = Path(filename).stem

    if stem.startswith("objects365_v1_"):
        img_id = int(stem.replace("objects365_v1_", ""))
        fname  = v1_index.get_image_file(img_id)
        if fname is None:
            return None, None, stem
        return OBJ365_V1_TRAIN / fname, img_id, stem

    if stem.startswith("objects365_v2_"):
        m = re.search(r"patch(\d+)", image_path_str)
        if m is None:
            return None, None, stem
        p = OBJ365_V2_DIR / f"patch{m.group(1)}" / filename
        return (p if p.exists() else None), None, stem

    return None, None, None


def part_bbox_union(parts: dict) -> list | None:
    xs1, ys1, xs2, ys2 = [], [], [], []
    for pd in parts.values():
        for inst in pd.get("instances", {}).values():
            b = inst.get("bbox")
            if b:
                xs1.append(b[0]); ys1.append(b[1])
                xs2.append(b[2]); ys2.append(b[3])
    if not xs1:
        return None
    return [min(xs1), min(ys1), max(xs2), max(ys2)]


# ── Crop helpers ──────────────────────────────────────────────────────────────
def crop_image(image_np: np.ndarray, bbox_xyxy: list, pad: int):
    H, W = image_np.shape[:2]
    x1 = max(0, int(bbox_xyxy[0]) - pad)
    y1 = max(0, int(bbox_xyxy[1]) - pad)
    x2 = min(W, int(bbox_xyxy[2]) + pad)
    y2 = min(H, int(bbox_xyxy[3]) + pad)
    return image_np[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def map_mask_to_crop(mask_full: np.ndarray, crop_box: tuple) -> np.ndarray:
    cx1, cy1, cx2, cy2 = crop_box
    return mask_full[cy1:cy2, cx1:cx2].copy()


# ── SAM3 wrapper ──────────────────────────────────────────────────────────────
class SAM3Model:
    def __init__(self, checkpoint_path: str, device: torch.device, pred_thres: float = 0.49):
        self.device     = device
        self.pred_thres = pred_thres
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            enable_inst_interactivity=True,
        )
        model.to(device)
        self.model     = model
        self.processor = Sam3Processor(model)
        self.state     = None

    def set_input_image(self, image_rgb: np.ndarray):
        # Sam3Processor.set_image correctly reads H/W from PIL images;
        # numpy HxWxC causes it to read shape[-2:]=(W,C) as (height,width)
        pil_img = Image.fromarray(image_rgb.astype(np.uint8))
        self.state = self.processor.set_image(pil_img)

    def get_prediction(self, clicker: Clicker, mask=None):
        clicks = clicker.get_clicks()
        points = np.array([[c.coords[1], c.coords[0]] for c in clicks])  # (x, y)
        labels = np.array([int(c.is_positive) for c in clicks])
        masks, scores, logits = self.model.predict_inst(
            self.state,
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
            mask_input=mask,
        )
        scores_np = scores if isinstance(scores, np.ndarray) else scores.detach().cpu().numpy()
        idx       = int(np.argmax(scores_np))
        pred_mask = masks[idx] > self.pred_thres
        last_logits = logits[[idx]]
        return pred_mask, last_logits


# ── IoU helpers ───────────────────────────────────────────────────────────────
def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union + 1e-6)


def mask_to_rle(mask: np.ndarray) -> dict:
    m = mask.astype(np.uint8)
    rle = mask_util.encode(np.array(m, order="F"))
    if isinstance(rle, list):
        rle = mask_util.merge(rle)
    return {"counts": rle["counts"].decode("utf-8"), "size": rle["size"]}


# ── Main click loop ───────────────────────────────────────────────────────────
def run_click_loop(
    predictor: SAM3Model,
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,          # binary uint8 (0/255 or 0/1, H×W of the crop)
    max_clicks: int,
    image_name: str,
) -> list[dict]:
    """
    Run interactive segmentation click loop with monotone-increasing IoU stop.
    Returns clicks_list compatible with fliter_data.py.
    """
    gt_bin  = (gt_mask > 0).astype(np.uint8)
    clicker = Clicker(gt_mask=gt_bin)
    clicker.visualize = False

    predictor.set_input_image(image_rgb)
    pred_mask     = np.zeros(image_rgb.shape[:2], dtype=bool)
    last_logits   = None
    prev_iou      = None
    clicks_list   = []

    for click_idx in range(max_clicks):
        clicker.make_next_click(pred_mask.astype(np.uint8), image_name)
        pred_mask, last_logits = predictor.get_prediction(clicker, mask=last_logits)

        iou = compute_iou(pred_mask, gt_bin)

        # Monotone-increasing stop: quit if IoU regresses
        if prev_iou is not None and iou < prev_iou:
            break

        pred_rle = mask_to_rle(pred_mask)
        click_entry = {
            "idx":         click_idx,
            "coor":        list(clicker.clicks_list[-1].coords),  # [row, col]
            "is_positive": bool(clicker.clicks_list[-1].is_positive),
            "mask":        pred_rle,
            "box":         mask_util.toBbox(pred_rle).tolist(),
            "area":        int(mask_util.area(pred_rle)),
            "iou":         float(iou),
        }
        clicks_list.append(click_entry)
        prev_iou = iou

    return clicks_list


# ── Per-annotation processing ─────────────────────────────────────────────────
def process_annotation(
    ann_path: Path,
    v1_index: Obj365V1Index,
    v2_index: Obj365V2Index,
    predictor: SAM3Model,
    output_root: Path,
    bbox_pad: int,
    max_clicks: int,
) -> list[dict]:
    with open(ann_path) as f:
        ann = json.load(f)

    if ann.get("skipped") or not ann.get("completed"):
        return []

    img_path, image_id, img_stem = find_image_path(ann["image_path"], v1_index)
    if img_path is None or not img_path.exists():
        return []

    try:
        image_np = np.array(Image.open(img_path).convert("RGB"))
    except Exception:
        return []

    parts = ann.get("parts", {})

    # ── Resolve full object bbox ──────────────────────────────────────────────
    first_inst = next(
        (inst for pd in parts.values() for inst in pd.get("instances", {}).values()),
        None,
    )
    if first_inst is None:
        return []

    obj_name  = first_inst.get("object", "")
    obj_index = first_inst.get("object_index", 0)

    if image_id is not None:
        obj_bbox = v1_index.get_object_bbox(image_id, obj_name, obj_index)
    else:
        obj_bbox = v2_index.get_object_bbox(img_stem, obj_name, obj_index)

    if obj_bbox is None:
        obj_bbox = part_bbox_union(parts)
    if obj_bbox is None:
        return []

    # ── Crop (don't save yet — only save if at least one valid part exists) ──
    obj_crop, crop_box = crop_image(image_np, obj_bbox, pad=bbox_pad)
    safe_obj = re.sub(r"[^a-zA-Z0-9]", "_", obj_name)
    # Mirror source dir structure: output_root / object_dir / patch_dir / image_stem /
    rel_dir = ann_path.parent.relative_to(DATA4SAM3_ROOT)
    crop_save_dir = output_root / rel_dir
    crop_save_dir.mkdir(parents=True, exist_ok=True)
    crop_save_path = crop_save_dir / f"{safe_obj}_{obj_index}__obj.jpg"
    crop_h, crop_w = obj_crop.shape[:2]
    samples = []
    crop_saved = crop_save_path.exists()

    for part_key, part_data in parts.items():
        for inst_key, inst in part_data.get("instances", {}).items():
            mask_rel_path = inst.get("mask_path", "")
            mask_path     = DATA4SAM3_MASK_ROOT / mask_rel_path
            if not mask_path.exists():
                continue

            part_mask_full = np.array(Image.open(mask_path).convert("L"))
            part_mask_crop = map_mask_to_crop(part_mask_full, crop_box)

            if (part_mask_crop > 0).sum() == 0:
                continue  # part not visible inside object crop

            caption    = inst.get("instruction", {}).get("orps", "the target object")
            image_name = f"{img_stem}__{inst_key}"

            clicks_list = run_click_loop(
                predictor, obj_crop, part_mask_crop,
                max_clicks=max_clicks,
                image_name=image_name,
            )

            if not clicks_list:
                continue

            # Save crop on first successful part
            if not crop_saved:
                Image.fromarray(obj_crop).save(crop_save_path, quality=95)
                crop_saved = True

            gt_bin  = (part_mask_crop > 0).astype(np.uint8)
            gt_rle  = mask_to_rle(gt_bin)
            sample  = {
                "image_name": str(crop_save_path),
                "height":     crop_h,
                "width":      crop_w,
                "gt_ann": {
                    "caption":      caption,
                    "object":       obj_name,
                    "part":         inst.get("part", ""),
                    "inst_key":     inst_key,
                    "segmentation": gt_rle,
                    "bbox":         mask_util.toBbox(gt_rle).tolist(),
                    "area":         int(mask_util.area(gt_rle)),
                },
                "clicks_list": clicks_list,
            }
            samples.append(sample)

    return samples


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3_ckpt",    default="models/sam3/sam3.pt")
    ap.add_argument("--output_dir",   default="data/affordance/traj_and_cropped_images")
    ap.add_argument("--output_json",  default="data/affordance/traj_and_cropped_images/trajs_raw.json")
    ap.add_argument("--bbox_pad",     type=int,   default=20,   help="pixel padding around object bbox")
    ap.add_argument("--max_clicks",   type=int,   default=20)
    ap.add_argument("--pred_thresh",  type=float, default=0.49)
    ap.add_argument("--gpus",         default="0")
    ap.add_argument("--object_dirs",  nargs="+",  default=None,
                    help="object dirs to process; default: all dirs in DATA4SAM3_ROOT")
    ap.add_argument("--start",        type=int,   default=0)
    ap.add_argument("--end",          type=int,   default=-1,   help="-1 means all")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpus.split(',')[0]}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    # ── Load indices ──────────────────────────────────────────────────────────
    v1_index = Obj365V1Index(OBJ365_V1_JSON)
    v2_index = Obj365V2Index(V2_BBOX_INDEX)

    # ── Load SAM3 ─────────────────────────────────────────────────────────────
    print(f"Loading SAM3 from {args.sam3_ckpt} ...")
    predictor = SAM3Model(args.sam3_ckpt, device, pred_thres=args.pred_thresh)
    print("SAM3 ready.")

    # ── Collect annotation paths ──────────────────────────────────────────────
    object_dirs = args.object_dirs or sorted(
        d.name for d in DATA4SAM3_ROOT.iterdir() if d.is_dir()
    )
    ann_paths = []
    for obj_dir in object_dirs:
        ann_paths.extend(sorted((DATA4SAM3_ROOT / obj_dir).rglob("annotation.json")))
    print(f"Object dirs: {object_dirs}")

    end = len(ann_paths) if args.end == -1 else args.end
    ann_paths = ann_paths[args.start:end]
    print(f"Processing {len(ann_paths)} annotations [{args.start}:{end}]")

    # ── Run ───────────────────────────────────────────────────────────────────
    all_data   = []
    total      = 0
    skipped    = 0

    for i, ann_path in enumerate(ann_paths):
        samples = process_annotation(
            ann_path, v1_index, v2_index, predictor,
            output_root, args.bbox_pad, args.max_clicks,
        )
        if not samples:
            skipped += 1
        else:
            all_data.extend(samples)
            total += len(samples)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(ann_paths)}] traj samples: {total}, skipped annotations: {skipped}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "info": {
            "args": {
                "sam3_ckpt":   args.sam3_ckpt,
                "bbox_pad":    args.bbox_pad,
                "max_clicks":  args.max_clicks,
                "pred_thresh": args.pred_thresh,
                "object_dirs": args.object_dirs,
                "start":       args.start,
                "end":         end,
            }
        },
        "data": all_data,
    }
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    with open(args.output_json, "w") as f:
        json.dump(out, f, ensure_ascii=False, cls=_Enc)

    print(f"\nDone.")
    print(f"  Traj samples : {total}")
    print(f"  Skipped anns : {skipped}")
    print(f"  Cropped imgs : {output_root}")
    print(f"  Output JSON  : {args.output_json}")


if __name__ == "__main__":
    main()
