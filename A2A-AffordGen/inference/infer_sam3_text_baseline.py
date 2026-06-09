#!/usr/bin/env python3
"""
Baseline: SAM3 with text prompt only (no SegAgent VLM, no iterative clicks).

For every test case in val_manifest.jsonl:
    SAM3.set_text_prompt(description)  ->  {masks, boxes, scores}
    pick highest-scoring mask
    compute IoU vs GT mask

Writes the same metrics_summary.json / metrics_per_image.json shape as
infer_segagent.py (but only the "best-step" / "step0" / "final" all collapse
to the single SAM3 prediction since there's no iteration).

Usage:
    bash run_infer_sam3_text_baseline.sh
or:
    python infer_sam3_text_baseline.py \\
        --sam3 models/sam3/sam3.pt \\
        --manifest data/affordance/train_and_val/val_set/val_manifest.jsonl \\
        --out_dir data/affordance/baselines/sam3_text \\
        --gpu 0
"""

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


SAM3_THRESH = 0.5


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_mask_bool(mask_path: str, image_size=None):
    try:
        m = np.array(Image.open(mask_path).convert("L")) > 0
    except Exception as e:
        print(f"  [WARN] cannot read GT mask {mask_path}: {e}")
        return None
    if image_size is not None:
        W, H = image_size
        if m.shape != (H, W):
            print(f"  [WARN] GT mask shape {m.shape} != image ({H},{W})")
            return None
    return m


def compute_iou_iu(pred: np.ndarray, gt: np.ndarray):
    p = pred.astype(bool); g = gt.astype(bool)
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    iou = (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)
    return float(iou), inter, union


def build_sam3(ckpt: str, gpu_id: int):
    sys.path.insert(0, str(Path(__file__).parent / "third_party" / "sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.cuda.set_device(gpu_id)
    model = build_sam3_image_model(checkpoint_path=ckpt)
    processor = Sam3Processor(model)
    return model, processor


def sam3_text_predict(processor, image: Image.Image, prompt: str):
    """Return (best_mask_bool, score). If SAM3 returns nothing, mask is all False."""
    state = processor.set_image(image)
    out = processor.set_text_prompt(state=state, prompt=prompt)
    masks  = to_numpy(out["masks"])
    scores = to_numpy(out["scores"])

    W, H = image.size
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim == 2:
        masks = masks[None]

    if masks.size == 0 or scores.size == 0:
        return np.zeros((H, W), dtype=bool), 0.0

    best = int(np.argmax(scores))
    m = masks[best]
    if m.dtype != np.bool_:
        m = m > SAM3_THRESH
    return m.astype(bool), float(scores[best])


def overlay_vis(img_np, mask, color=(0, 200, 50), alpha=0.45):
    out = img_np.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def save_demo(image: Image.Image, pred_mask, gt_mask, description, score, iou, out_path):
    img_np = np.array(image)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"SAM3+text: '{description}'  score={score:.3f}  IoU={iou:.3f}", fontsize=10)

    axes[0].imshow(img_np); axes[0].set_title("Original", fontsize=8); axes[0].axis("off")
    axes[1].imshow(overlay_vis(img_np, pred_mask, color=(0, 200, 50)))
    axes[1].set_title("SAM3 pred", fontsize=8); axes[1].axis("off")
    axes[2].imshow(overlay_vis(img_np, gt_mask, color=(255, 50, 50)))
    axes[2].set_title("GT", fontsize=8); axes[2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sam3", required=True)
    ap.add_argument("--manifest",
                    default="data/affordance/train_and_val/val_set/val_manifest.jsonl")
    ap.add_argument("--out_dir",
                    default="data/affordance/baselines/sam3_text")
    ap.add_argument("--n_images", type=int, default=100,
                    help="0 = run all manifest entries")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_demo_every", type=int, default=10,
                    help="save a demo PNG every N images (0 = none)")
    args = ap.parse_args()

    gpu_id = int(args.gpu.split(",")[0])
    torch.cuda.set_device(gpu_id)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    manifest_path = Path(args.manifest)
    val_root = manifest_path.parent
    entries = [json.loads(l) for l in open(manifest_path) if l.strip()]
    print(f"Manifest: {len(entries)} test cases")

    import random
    random.seed(args.seed)
    random.shuffle(entries)
    if args.n_images and args.n_images < len(entries):
        entries = entries[: args.n_images]
        print(f"Sampling first {len(entries)}")

    print(f"Loading SAM3 from {args.sam3} ...")
    sam3_model, sam3_processor = build_sam3(args.sam3, gpu_id)

    metric_rows = []
    for i, e in enumerate(entries):
        img_path  = val_root / e["image_path"]
        mask_path = val_root / e["gt_mask_path"]
        description = e["description"]
        print(f"[{i+1}/{len(entries)}] {img_path.name} | '{description}'", flush=True)

        try:
            image = Image.open(img_path).convert("RGB")
            gt = load_mask_bool(str(mask_path), image_size=image.size)
            if gt is None:
                continue

            with torch.no_grad():
                pred, score = sam3_text_predict(sam3_processor, image, description)

            # Resize pred to image size if needed (SAM3 should already match)
            if pred.shape != gt.shape:
                pred = np.array(Image.fromarray(pred.astype(np.uint8) * 255)
                                .resize(image.size, Image.NEAREST)) > 0

            iou, inter, union = compute_iou_iu(pred, gt)
            row = {
                "image_name": e["image_name"],
                "description": description,
                "gt_mask_path": e["gt_mask_path"],
                "sam3_score": score,
                "iou": iou,
                "inter": inter,
                "union": union,
                "success_at_50": iou >= 0.5,
            }
            metric_rows.append(row)
            print(f"  iou={iou:.4f}  score={score:.3f}")

            if args.save_demo_every and (i % args.save_demo_every == 0):
                stem = Path(e["image_name"]).stem
                safe = re.sub(r'[^a-z0-9]+', '_', description.lower()).strip('_')[:80]
                save_demo(image, pred, gt, description, score, iou,
                          out_dir / f"{stem}__{safe}_demo.png")
        except Exception as ex:
            print(f"  [ERROR] {ex}")
            import traceback; traceback.print_exc()

    # ── Summary (same metrics shape as infer_segagent.py) ─────────────────────
    if not metric_rows:
        print("\n[WARN] no metrics produced")
        return

    ious   = np.array([r["iou"]   for r in metric_rows], dtype=np.float64)
    inters = np.array([r["inter"] for r in metric_rows], dtype=np.int64)
    unions = np.array([r["union"] for r in metric_rows], dtype=np.int64)

    def ciou(i, u):
        s = int(u.sum())
        return float(i.sum() / s * 100.0) if s > 0 else 0.0
    def p50_95(x):
        ts = np.linspace(0.50, 0.95, 10)
        return float(np.mean([(x >= t).mean() for t in ts]) * 100.0)

    summary = {
        "method": "SAM3+text (zero-shot baseline)",
        "evaluated_images": int(len(metric_rows)),
        "requested_images": int(len(entries)),
        # Same field names as infer_segagent so cross-method aggregation scripts work.
        # All three groups (best/step0/final) are identical here -- one-shot baseline.
        "gIoU":          float(ious.mean() * 100.0),
        "cIoU":          ciou(inters, unions),
        "P@50":          float((ious >= 0.5).mean() * 100.0),
        "P@50-95":       p50_95(ious),
        "step0_gIoU":    float(ious.mean() * 100.0),
        "step0_cIoU":    ciou(inters, unions),
        "step0_P@50":    float((ious >= 0.5).mean() * 100.0),
        "step0_P@50-95": p50_95(ious),
        "refinement_gain": 0.0,
        "final_gIoU":    float(ious.mean() * 100.0),
        "final_cIoU":    ciou(inters, unions),
        "final_P@50":    float((ious >= 0.5).mean() * 100.0),
        "final_P@50-95": p50_95(ious),
    }

    with open(out_dir / "metrics_per_image.json", "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSAM3+text baseline ({summary['evaluated_images']} images):")
    print(f"  gIoU={summary['gIoU']:6.2f}  cIoU={summary['cIoU']:6.2f}  "
          f"P@50={summary['P@50']:6.2f}  P@50-95={summary['P@50-95']:6.2f}")
    print(f"  -> {out_dir}/metrics_summary.json")


if __name__ == "__main__":
    main()
