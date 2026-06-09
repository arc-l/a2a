#!/usr/bin/env python3
"""
Render val_set overlay images: for every (image, gt_mask) pair listed in
val_manifest.jsonl, save image with semi-transparent green mask overlay.

This mirrors the runtime "step k > 0" input the model sees during multi-step
inference: original image + green-tinted region where the previous mask was.

Layout (mirrors val_set/gt_mask/):
    val_set/overlap/{image_stem}/{safe_desc}.jpg

Usage:
    python data_process/build_val_overlap.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


VAL_ROOT     = Path('data/affordance/train_and_val/val_set')
MANIFEST     = VAL_ROOT / 'val_manifest.jsonl'
IMAGES_DIR   = VAL_ROOT / 'images'
GT_MASK_DIR  = VAL_ROOT / 'gt_mask'
OVERLAP_DIR  = VAL_ROOT / 'overlap'

OVERLAY_COLOR = (0, 255, 0)
OVERLAY_ALPHA = 0.5


def apply_green_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Same blending logic as gen_segagent_train_data.py / infer_segagent.py."""
    img = np.array(image, dtype=np.float32)
    if img.shape[:2] != mask.shape:
        raise ValueError(f'image shape {img.shape[:2]} != mask shape {mask.shape}')
    green = np.zeros_like(img)
    green[..., 0], green[..., 1], green[..., 2] = OVERLAY_COLOR
    m = mask.astype(bool)[..., None]
    blended = np.where(m, img * (1 - OVERLAY_ALPHA) + green * OVERLAY_ALPHA, img)
    return Image.fromarray(blended.astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--quality', type=int, default=92,
                    help='JPEG quality (default 92)')
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise FileNotFoundError(f'Manifest not found: {MANIFEST}\n'
                                f'Run build_val_gt_manifest.py first.')

    entries = [json.loads(l) for l in open(MANIFEST) if l.strip()]
    print(f'Manifest entries: {len(entries)}')

    if not args.dry_run:
        OVERLAP_DIR.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_err = 0
    for e in entries:
        img_rel  = e['image_path']        # "images/xxx.jpg"
        mask_rel = e['gt_mask_path']      # "gt_mask/{stem}/{desc}.png"

        img_path  = VAL_ROOT / img_rel
        mask_path = VAL_ROOT / mask_rel
        stem = Path(e['image_name']).stem
        # Same filename stem as gt_mask, but .jpg
        dst = OVERLAP_DIR / stem / (Path(mask_rel).stem + '.jpg')

        if dst.exists():
            n_skip += 1
            continue

        try:
            image = Image.open(img_path).convert('RGB')
            mask  = np.array(Image.open(mask_path).convert('L')) > 0
            overlay = apply_green_overlay(image, mask)
        except Exception as ex:
            print(f'  [ERR] {img_path.name} / {Path(mask_rel).name}: {ex}')
            n_err += 1
            continue

        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(dst, quality=args.quality)
        n_ok += 1

    print(f'\nResult:')
    print(f'  written : {n_ok}')
    print(f'  skipped : {n_skip}  (already existed)')
    print(f'  errors  : {n_err}')
    if not args.dry_run:
        print(f'\n[OK] Overlap -> {OVERLAP_DIR}/{{stem}}/{{desc}}.jpg')


if __name__ == '__main__':
    main()
