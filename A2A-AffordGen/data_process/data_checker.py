#!/usr/bin/env python3
"""
data_checker.py — Visualize a SegAgent training trajectory from train.jsonl.

For a selected trajectory:
  - De-normalizes click coords from the JSONL answer format
  - Runs SAM3 step-by-step, feeding previous logits as mask_input
  - Plots each step: original image + accumulated clicks + predicted mask

Usage:
    python data_process/data_checker.py \\
        --jsonl data/affordance/segagent_train/train.jsonl \\
        --traj_index 0 \\
        --sam3_ckpt models/sam3/sam3.pt \\
        --output vis_data_checker/traj_0.png
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
from PIL import Image


RE_ANSWER = re.compile(
    r"<ref>(.+?)</ref>\s+Current IOU:\s*([0-9.]+),\s*(Positive|Negative)\s+point:\s*\((\d+),\s*(\d+)\),\s*Predicted next IOU:\s*([0-9.]+)"
)


def parse_answer(content: str):
    m = RE_ANSWER.search(content)
    if not m:
        return None
    desc, cur_iou, click_type, x_str, y_str, next_iou = m.groups()
    return {
        "description": desc,
        "current_iou": float(cur_iou),
        "is_positive": click_type == "Positive",
        "x_norm": int(x_str),   # col / W * 1000
        "y_norm": int(y_str),   # row / H * 1000
        "next_iou": float(next_iou),
    }


def load_trajectories(jsonl_path: str, max_trajs: int = None):
    """Group consecutive JSONL lines into trajectories.

    A new trajectory starts whenever the image filename stem has no '__step' in it.
    Steps of the same trajectory are written consecutively by gen_segagent_train_data.py.
    """
    trajs = []
    current: list = []

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            img_stem = Path(sample["images"][0]).stem
            is_step0 = "__step" not in img_stem

            if is_step0 and current:
                trajs.append(current)
                if max_trajs and len(trajs) >= max_trajs:
                    return trajs
                current = []

            parsed = parse_answer(sample["messages"][1]["content"])
            current.append({"img_path": sample["images"][0], "parsed": parsed})

    if current:
        trajs.append(current)
    return trajs


def find_trajectory_by_image(jsonl_path: str, step0_image: str):
    """Find and return the trajectory whose step-0 image matches step0_image.

    Scans the full JSONL without loading all trajectories into memory.
    Returns (traj_idx, steps) or raises ValueError if not found.
    """
    current: list = []
    traj_idx = -1
    found = None

    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            img_path = sample["images"][0]
            img_stem = Path(img_path).stem
            is_step0 = "__step" not in img_stem

            if is_step0:
                if found is not None:
                    # Previous traj was the one we wanted — current finished it
                    return traj_idx, found
                if current:
                    traj_idx += 1
                current = []
                traj_idx += 1  # count this new traj

            parsed = parse_answer(sample["messages"][1]["content"])
            step = {"img_path": img_path, "parsed": parsed}
            current.append(step)

            # Mark when we hit the target step-0
            if is_step0 and img_path == step0_image:
                found = current  # found points into current list

    # End of file — check last traj
    if found is not None:
        return traj_idx, found

    raise ValueError(f"No trajectory with step-0 image: {step0_image}")


def denormalize(x_norm: int, y_norm: int, W: int, H: int):
    """x_norm = col/W*1000, y_norm = row/H*1000  →  (col, row) in pixels."""
    col = int(x_norm / 1000 * W)
    row = int(y_norm / 1000 * H)
    return col, row


def overlay_mask(image_np: np.ndarray, mask: np.ndarray,
                 color=(0, 200, 50), alpha: float = 0.45) -> np.ndarray:
    out = image_np.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def run_sam3_trajectory(steps: list, sam3_ckpt: str, gpu: str = "0", thresh: float = 0.49):
    """Run SAM3 step-by-step on the trajectory. Returns (image_pil, W, H, results)."""
    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    orig_img_path = steps[0]["img_path"]
    image_pil = Image.open(orig_img_path).convert("RGB")
    W, H = image_pil.size

    torch.cuda.set_device(int(gpu.split(",")[0]))
    model = build_sam3_image_model(checkpoint_path=sam3_ckpt, enable_inst_interactivity=True)
    processor = Sam3Processor(model)
    state = processor.set_image(image_pil)

    acc_cols, acc_rows, acc_labels = [], [], []
    last_logits = None
    results = []

    with torch.no_grad():
        for step_idx, step in enumerate(steps):
            p = step["parsed"]
            col, row = denormalize(p["x_norm"], p["y_norm"], W, H)
            acc_cols.append(col)
            acc_rows.append(row)
            acc_labels.append(1 if p["is_positive"] else 0)

            # point_coords: each row is [col, row] = [x, y]
            point_coords = np.array(list(zip(acc_cols, acc_rows)), dtype=np.float32)
            point_labels = np.array(acc_labels, dtype=np.int64)

            masks, scores, logits = model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
                mask_input=last_logits,
            )

            scores_np = scores if isinstance(scores, np.ndarray) else scores.detach().cpu().numpy()
            masks_np  = masks  if isinstance(masks,  np.ndarray) else masks.detach().cpu().numpy()

            max_idx = int(np.argmax(scores_np))
            # masks shape may be (N, H, W) or (N, 1, H, W)
            m = masks_np[max_idx]
            if m.ndim == 3:
                m = m[0]
            pred_mask = m > thresh

            last_logits = logits[[max_idx]]

            results.append({
                "mask": pred_mask,
                "score": float(scores_np[max_idx]),
                "col": col,
                "row": row,
                "is_positive": p["is_positive"],
                "stored_current_iou": p["current_iou"],
                "stored_next_iou": p["next_iou"],
                "step_idx": step_idx,
            })

    return image_pil, W, H, results


def draw_figure(image_pil: Image.Image, sam3_results: list,
                traj_idx: int, description: str, output_path: str):
    image_np = np.array(image_pil)
    n = len(sam3_results)

    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.8),
                              squeeze=False)

    fig.suptitle(f"Traj {traj_idx} | {description}", fontsize=11, y=1.01)

    for step_idx, res in enumerate(sam3_results):
        ri, ci = step_idx // ncols, step_idx % ncols
        ax = axes[ri][ci]

        vis = overlay_mask(image_np, res["mask"])
        ax.imshow(vis)

        # Draw all accumulated clicks up to this step
        for k in range(step_idx + 1):
            r = sam3_results[k]
            if r["is_positive"]:
                ax.scatter([r["col"]], [r["row"]], s=80, c="#20d64f",
                           edgecolors="black", linewidths=1.2, zorder=5)
            else:
                ax.scatter([r["col"]], [r["row"]], s=80, c="#ff3333",
                           marker="x", linewidths=2.5, zorder=5)
            ax.text(r["col"] + 3, r["row"] + 3, str(k), color="white", fontsize=7,
                    weight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.1"),
                    zorder=6)

        ax.set_title(
            f"Step {step_idx} | {'Pos' if res['is_positive'] else 'Neg'} click\n"
            f"score={res['score']:.2f} | stored_next_iou={res['stored_next_iou']:.2f}",
            fontsize=8,
        )
        ax.axis("off")

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",
                    default="data/affordance/segagent_train/train.jsonl")
    ap.add_argument("--traj_index", type=int, default=None,
                    help="0-based trajectory index; random if omitted")
    ap.add_argument("--step0_image", default=None,
                    help="find trajectory by its step-0 image path (scans full JSONL)")
    ap.add_argument("--sam3_ckpt",
                    default="models/sam3/sam3.pt")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--thresh", type=float, default=0.49)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max_trajs_scan", type=int, default=1000,
                    help="max trajectories to scan when using --traj_index or random")
    args = ap.parse_args()

    # Add SAM3 to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "third_party" / "sam3"))

    if args.step0_image:
        print(f"Searching for trajectory with step-0 image: {args.step0_image}")
        traj_idx, steps = find_trajectory_by_image(args.jsonl, args.step0_image)
        print(f"Found at trajectory index {traj_idx}")
    else:
        print(f"Loading trajectories from {args.jsonl} ...")
        trajs = load_trajectories(args.jsonl, max_trajs=args.max_trajs_scan)
        print(f"Loaded {len(trajs)} trajectories (scanned up to {args.max_trajs_scan})")

        if args.traj_index is None:
            import random
            traj_idx = random.randint(0, len(trajs) - 1)
            print(f"Random trajectory index: {traj_idx}")
        else:
            traj_idx = args.traj_index

        if traj_idx >= len(trajs):
            print(f"Error: traj_index {traj_idx} >= {len(trajs)} (increase --max_trajs_scan)")
            sys.exit(1)

        steps = trajs[traj_idx]

    description = steps[0]["parsed"]["description"] if steps[0]["parsed"] else "?"
    print(f"Trajectory {traj_idx}: {len(steps)} steps | '{description}'")
    print(f"Original image: {steps[0]['img_path']}")

    if args.output is None:
        args.output = f"vis_data_checker/traj_{traj_idx}.png"

    print("Running SAM3 inference ...")
    image_pil, W, H, sam3_results = run_sam3_trajectory(
        steps, args.sam3_ckpt, gpu=args.gpu, thresh=args.thresh
    )

    draw_figure(image_pil, sam3_results, traj_idx, description, args.output)


if __name__ == "__main__":
    main()
