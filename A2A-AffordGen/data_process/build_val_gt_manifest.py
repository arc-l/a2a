#!/usr/bin/env python3
"""
Build a self-contained val set with GT masks.

GT source: data/affordance/trajs_with_captions/*.json
This is the SAME upstream pipeline that produced val.jsonl, so every
(image_name, caption) pair in val.jsonl is guaranteed to have a matching GT
record (with RLE mask) here. The previously-tried `image_mask_task_prompt`
results.json was annotated with a different captioning scheme, leading to
~50% no_part_match. Using trajs_with_captions should be near 100%.

Reads:
  - data/affordance/affordgen_train/val.jsonl
  - data/affordance/trajs_with_captions/*.json
  - data/affordance/train_and_val/val_set/images/

Writes:
  - val_set/gt_mask/{image_stem}/{safe_desc}.png   GT mask PNG (decoded from RLE)
  - val_set/val_manifest.jsonl                     one line per test case

Manifest line:
  {
    "image_name": "29.jpg",
    "image_path": "images/29.jpg",
    "orig_image_path": "/.../RAGNet/.../29.jpg",
    "description": "body of the hot dog",
    "gt_mask_path": "gt_mask/29/body_of_the_hot_dog.png",
    "gt_caption": "body of the hot dog",
    "gt_original_caption": "hot dog",
    "gt_bbox": [x, y, w, h],
    "gt_area": 1234,
    "height": 480, "width": 640
  }

Usage:
    python data_process/build_val_gt_manifest.py [--dry_run]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


VAL_ROOT  = Path('data/affordance/train_and_val/val_set')
VAL_JSONL = Path('data/affordance/affordgen_train/val.jsonl')

# Multiple upstream trajectory sources (must match what gen_affordgen_train_data.py used).
TRAJ_SOURCES = [
    Path('data/affordance/trajs_with_captions'),       # primary (dir of *.json)
    Path('data/affordance/traj_and_cropped_images/trajs_filtered.json'),  # multi-object extra
]

IMAGES_DIR    = VAL_ROOT / 'images'
GT_MASK_DIR   = VAL_ROOT / 'gt_mask'
MANIFEST_PATH = VAL_ROOT / 'val_manifest.jsonl'


def safe_name(s: str, max_len: int = 80) -> str:
    s = re.sub(r'[^a-z0-9]+', '_', s.lower()).strip('_')
    return s[:max_len] or 'unnamed'


def decode_rle(rle: dict) -> np.ndarray:
    """Decode COCO-style RLE (counts string + size) to a (H, W) bool mask."""
    try:
        from pycocotools import mask as mask_utils
        return mask_utils.decode(rle).astype(bool)
    except ImportError:
        pass
    # Fallback: pure-Python decoder for "Pbb13k7;F..." style RLE strings
    # (LEB128 + run-length, COCO mask format)
    raise RuntimeError("pycocotools not installed; install with `pip install pycocotools`")


def gather_val_step0():
    """Yield (orig_image_path, description) for every step-0 entry in val.jsonl."""
    re_desc = re.compile(r'<ref>(.+?)</ref>')
    with open(VAL_JSONL) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            img = s['images'][0]
            if '__step' in Path(img).stem:
                continue
            m = re_desc.search(s['messages'][0]['content'])
            if m:
                yield img, m.group(1)


def md5_file(path: Path) -> str:
    import hashlib
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_traj_index():
    """Index by (image_name, caption) -> gt_ann dict (with segmentation RLE).

    Searches all sources in TRAJ_SOURCES. A source can be either a directory
    (all *.json under it are read) or a single .json file.
    """
    json_paths = []
    for s in TRAJ_SOURCES:
        if not s.exists():
            print(f'[WARN] traj source missing: {s}')
            continue
        if s.is_dir():
            json_paths.extend(sorted(s.glob('*.json')))
        else:
            json_paths.append(s)

    idx = {}
    for p in json_paths:
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f'[WARN] cannot read {p}: {e}')
            continue
        arr = data.get('data', data) if isinstance(data, dict) else data
        if not isinstance(arr, list):
            continue
        for item in arr:
            if not isinstance(item, dict):
                continue
            img = item.get('image_name')
            ann = item.get('gt_ann')
            if not img or not isinstance(ann, dict):
                continue
            cap = ann.get('caption', '')
            if not cap:
                continue
            idx[(img, cap)] = {
                'gt_ann': ann,
                'height': item.get('height'),
                'width':  item.get('width'),
                'source_file': p.name,
            }
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry_run', action='store_true',
                    help='do not copy files or write manifest, just report counts')
    args = ap.parse_args()

    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f'{IMAGES_DIR} not found')

    val_set_files = list(IMAGES_DIR.glob('*.jpg')) + list(IMAGES_DIR.glob('*.png'))
    val_set_basenames = {p.name for p in val_set_files}
    print(f'val_set/images: {len(val_set_files)} files')

    # Collect val.jsonl step-0 entries whose basename appears in val_set.
    val_entries = []
    for orig, desc in gather_val_step0():
        if Path(orig).name in val_set_basenames:
            val_entries.append((orig, desc))
    print(f'val.jsonl step-0 entries covering val_set basenames: {len(val_entries)}')

    # md5-match: for each val_set image, find which orig_path's file is byte-identical.
    # Many basenames collide across RAGNet trajectories, so md5 is the only safe key.
    print(f'\nMatching val_set/images to orig_paths by md5 ...')
    val_set_md5 = {p.name: md5_file(p) for p in val_set_files}

    # Cache orig_path md5 (only for ones we haven't already mapped to a basename)
    orig_to_basename = {}  # orig_path -> matching val_set basename
    by_bn = defaultdict(list)
    for orig, desc in val_entries:
        by_bn[Path(orig).name].append(orig)

    for bn, origs in by_bn.items():
        target_md5 = val_set_md5[bn]
        for orig in dict.fromkeys(origs):  # unique-preserving
            try:
                if md5_file(Path(orig)) == target_md5:
                    orig_to_basename[orig] = bn
                    break
            except FileNotFoundError:
                continue
    print(f'  matched {len(orig_to_basename)} orig_paths to val_set basenames')

    # Now collect descriptions PER matched orig_path
    pairs = []  # (orig_path, basename, description)
    for orig, desc in val_entries:
        if orig in orig_to_basename:
            pairs.append((orig, orig_to_basename[orig], desc))
    # Deduplicate identical (orig, desc) pairs (val.jsonl can have repeats)
    pairs = list(dict.fromkeys(pairs))
    print(f'  total (orig_path, description) pairs to write: {len(pairs)}')

    print(f'\nBuilding trajs_with_captions index ...')
    traj_idx = build_traj_index()
    print(f'  {len(traj_idx)} unique (image_name, caption) entries indexed\n')

    # Output bookkeeping
    if not args.dry_run:
        # Wipe existing gt_mask dir so stale wrong-mask copies don't linger
        import shutil
        if GT_MASK_DIR.exists():
            shutil.rmtree(GT_MASK_DIR)
        GT_MASK_DIR.mkdir(parents=True, exist_ok=True)
    used_dst_names = defaultdict(set)
    manifest_lines = []
    n_match = 0
    n_missing = 0
    missing_examples = []

    for orig, bn, desc in sorted(pairs):
        stem = Path(bn).stem
        matched_orig = orig
        hit = traj_idx.get((orig, desc))

        if hit is None:
            n_missing += 1
            if len(missing_examples) < 8:
                missing_examples.append((bn, desc, [orig]))
            continue

        ann = hit['gt_ann']
        seg = ann.get('segmentation')
        if not (isinstance(seg, dict) and 'counts' in seg and 'size' in seg):
            n_missing += 1
            continue

        # Decode RLE -> bool mask -> PNG (uint8 0/255)
        mask = decode_rle(seg)
        H, W = mask.shape

        # Pick a unique destination filename.
        base = safe_name(desc)
        dst_name = base + '.png'
        counter = 1
        while dst_name in used_dst_names[stem]:
            counter += 1
            dst_name = f'{base}_{counter}.png'
        used_dst_names[stem].add(dst_name)

        sub_dir = GT_MASK_DIR / stem
        dst = sub_dir / dst_name

        if not args.dry_run:
            sub_dir.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                Image.fromarray((mask.astype(np.uint8) * 255), mode='L').save(dst)

        entry = {
            'image_name': bn,
            'image_path': f'images/{bn}',
            'orig_image_path': matched_orig,
            'description': desc,
            'gt_mask_path': f'gt_mask/{stem}/{dst_name}',
            'gt_caption': ann.get('caption', ''),
            'gt_original_caption': ann.get('original_caption', ''),
            'gt_bbox': ann.get('bbox'),
            'gt_area': ann.get('area'),
            'height': H, 'width': W,
            'source_file': hit['source_file'],
        }
        manifest_lines.append(entry)
        n_match += 1

    if not args.dry_run:
        with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
            for e in manifest_lines:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')

    print(f'Processed {len(pairs)} (orig_path, description) pairs:')
    print(f'  matched   : {n_match}')
    print(f'  missing   : {n_missing}')
    if missing_examples:
        print(f'\nMissing examples (up to 8):')
        for bn, desc, paths in missing_examples:
            print(f'  {bn} | {desc!r}')
            for p in paths:
                print(f'    {p}')

    if args.dry_run:
        print('\n[DRY RUN] no files written')
    else:
        print(f'\n[OK] Manifest -> {MANIFEST_PATH}')
        print(f'[OK] Masks    -> {GT_MASK_DIR}/{{stem}}/{{desc}}.png')


if __name__ == '__main__':
    main()
