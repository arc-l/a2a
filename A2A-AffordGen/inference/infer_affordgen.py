#!/usr/bin/env python3
"""
infer_affordgen.py — Multi-step AffordGen inference demo.

Pipeline per image:
  step 0 : original image  → VLM → click → SAM3 → mask
  step k : original + green overlay of step k-1 mask → VLM → click → SAM3 → mask
  ...until max_steps or predicted_next_iou >= stop_iou

Usage:
    python infer_affordgen.py \
        --ckpt   data/affordance/affordgen_ckpt_0511/v13-20260511-130834/checkpoint-500 \
        --sam3   models/sam3/sam3.pt \
        --val_jsonl data/affordance/affordgen_train/val.jsonl \
        --img_dir   data/affordance/train_and_val/val_set/images \
        --out_dir   vis_affordgen_demo \
        --n_images  20 \
        --max_steps 8 \
        --gpu 0
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_ALPHA = 0.5

USER_PROMPT = (
    "<image>\n"
    "Please optimize the semi-transparent green mask in a point-wise manner "
    "based on the image and description content, so that it covers the target "
    "object as accurately as possible. "
    "The object description is as follows: <ref>{description}</ref>"
)

RE_ANSWER = re.compile(
    r"<ref>(.+?)</ref>\s+Current IOU:\s*([0-9.]+),\s*(Positive|Negative)\s+point:\s*\((\d+),\s*(\d+)\)"
    r"(?:,\s*Predicted next IOU:\s*([0-9.]+))?"
)


# ── Helpers ────────────────────────────────────────────────────────────────────
def apply_green_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    img = np.array(image, dtype=np.float32)
    green = np.zeros_like(img)
    green[..., 0], green[..., 1], green[..., 2] = OVERLAY_COLOR
    m = mask.astype(bool)[..., None]
    blended = np.where(m, img * (1 - OVERLAY_ALPHA) + green * OVERLAY_ALPHA, img)
    return Image.fromarray(blended.astype(np.uint8))


def overlay_vis(image_np: np.ndarray, mask: np.ndarray,
                color=(0, 200, 50), alpha=0.45) -> np.ndarray:
    out = image_np.astype(np.float32).copy()
    out[mask] = out[mask] * (1 - alpha) + np.array(color, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def parse_vlm_output(text: str):
    m = RE_ANSWER.search(text)
    if not m:
        return None
    desc, cur_iou, click_type, x_str, y_str, next_iou_str = m.groups()
    return {
        "description": desc,
        "current_iou": float(cur_iou),
        "is_positive": click_type == "Positive",
        "x_norm": int(x_str),
        "y_norm": int(y_str),
        "next_iou": float(next_iou_str) if next_iou_str else None,
    }


def denormalize(x_norm, y_norm, W, H):
    return int(x_norm / 1000 * W), int(y_norm / 1000 * H)


def normalize_text(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def compute_mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    iou, _, _ = compute_mask_iou_inter_union(pred_mask, gt_mask)
    return iou


def compute_mask_iou_inter_union(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Return (iou, inter, union) so cIoU can aggregate pixel sums across images."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    if union == 0:
        return (1.0 if inter == 0 else 0.0), inter, union
    return float(inter / union), inter, union


def load_mask_bool(mask_path: str, image_size=None) -> np.ndarray | None:
    try:
        mask = Image.open(mask_path).convert("L")
    except Exception as e:
        print(f"  [WARN] cannot read GT mask: {mask_path} | {type(e).__name__}: {e}")
        return None
    arr = np.array(mask)
    gt = arr > 0
    if image_size is not None:
        W, H = image_size
        if gt.shape != (H, W):
            print(f"  [WARN] GT mask size mismatch: mask={gt.shape} image=({H},{W}) path={mask_path}")
            return None
    return gt


def load_gt_records(gt_results_root: str | None, gt_results_json: list[str]) -> dict:
    """
    Build full image_path -> records index from image_mask_task_prompt results.json files.
    Each record should contain image_path, mask_path, and part_name.

    Indexing by full image_path (not basename) is required because RAGNet has
    many trajectories that share basenames like '29.jpg' across different
    scenes — basename keying causes cross-scene GT mismatches.
    """
    paths = []
    for p in gt_results_json or []:
        paths.append(Path(p))
    if gt_results_root:
        root = Path(gt_results_root)
        paths.extend(sorted(root.glob("*/results.json")))

    index = defaultdict(list)
    for p in paths:
        if not p.exists():
            print(f"[WARN] GT results json missing: {p}")
            continue
        try:
            rows = json.load(open(p, "r", encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] cannot load GT results json: {p} | {type(e).__name__}: {e}")
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            image_path = r.get("image_path", "")
            mask_path = r.get("mask_path", "")
            if not image_path or not mask_path:
                continue
            index[image_path].append(r)
    return dict(index)


_STOPWORDS = {"of", "the", "a", "an", "with", "on", "in", "at", "to", "for"}


def pick_gt_record(gt_index: dict, orig_image_path: str, description: str):
    """
    Look up GT record by full original image_path AND match part_name to the
    description.

    Three-tier matching:
      1. Strict: description equals part_name (or one of 4 part+object patterns)
         e.g. "screen of the phone" == part="screen" + obj="phone"
      2. Permissive subset: description's content tokens (minus stopwords) are
         a subset of GT's (part_name + object_name) content tokens.
         e.g. "handle of the refrigerator" ⊆ "left door handle refrigerator"
         When multiple records match, pick the one with the FEWEST extra tokens
         (most specific that still covers description).
      3. Else: no match.

    Returns (record_or_None, status, available_part_names) where status is:
      - "match"              : strict match
      - "match_permissive"   : subset-fallback match
      - "no_image_record"    : image_path has no records at all in gt_index
      - "no_part_match"      : image has records but none match description
    """
    hits = gt_index.get(orig_image_path, [])
    if not hits:
        return None, "no_image_record", []

    available = [r.get("part_name", "") for r in hits]
    desc_tokens = normalize_text(description)
    desc_content = desc_tokens - _STOPWORDS

    # Tier 1: strict equality on part_name / part+object patterns.
    def matches_strict(r):
        part = r.get("part_name", "") or ""
        obj  = r.get("object_name", "") or ""
        candidates = [
            part,
            f"{part} of the {obj}",
            f"{part} of {obj}",
            f"{obj} {part}",
            f"{part} {obj}",
        ]
        return any(normalize_text(c) == desc_tokens for c in candidates if c)

    for r in hits:
        if matches_strict(r):
            return r, "match", available

    # Tier 2: subset of part_name + object_name content tokens.
    if desc_content:
        best, best_extra = None, float("inf")
        for r in hits:
            part = r.get("part_name", "") or ""
            obj  = r.get("object_name", "") or ""
            rec_content = normalize_text(f"{part} {obj}") - _STOPWORDS
            if desc_content.issubset(rec_content):
                extra = len(rec_content - desc_content)
                if extra < best_extra:
                    best, best_extra = r, extra
        if best is not None:
            return best, "match_permissive", available

    return None, "no_part_match", available


# ── Load val descriptions ──────────────────────────────────────────────────────
def load_val_descriptions(val_jsonl: str) -> dict:
    """Returns {orig_image_path: description} for step-0 entries only."""
    desc_map = {}
    RE_DESC = re.compile(r"<ref>(.+?)</ref>")
    with open(val_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            img_path = sample["images"][0]
            if "__step" in Path(img_path).stem:
                continue
            user_content = sample["messages"][0]["content"]
            m = RE_DESC.search(user_content)
            if m:
                desc_map[img_path] = m.group(1)
    return desc_map


# ── VLM inference ─────────────────────────────────────────────────────────────
def build_vlm(ckpt: str, device: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText

    ckpt_path = Path(ckpt)
    adapter_cfg = ckpt_path / "adapter_config.json"

    if adapter_cfg.exists():
        # LoRA / PEFT checkpoint: load base model, then apply adapter, then merge.
        from peft import PeftModel
        with open(adapter_cfg) as f:
            cfg = json.load(f)
        base_path = cfg["base_model_name_or_path"]
        print(f"Loading LoRA adapter from {ckpt}")
        print(f"  base model: {base_path}")
        print(f"  rank={cfg.get('r')} alpha={cfg.get('lora_alpha')}")

        processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_path,
            dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        # --- Diagnostic: verify the adapter actually attached. -----------
        import re
        target_regex = cfg.get("target_modules")
        all_names = [n for n, _ in base_model.named_modules()]
        if isinstance(target_regex, str):
            try:
                pat = re.compile(target_regex)
                n_match = sum(1 for n in all_names if pat.match(n))
                print(f"  [diag] base_model has {len(all_names)} modules; "
                      f"{n_match} match target_modules regex")
                if n_match == 0:
                    sample = [n for n in all_names if "language_model" in n
                              and any(p in n for p in ("q_proj", "k_proj",
                                                       "v_proj", "o_proj"))][:5]
                    print(f"  [diag] zero matches! example LLM proj names: {sample}")
            except re.error as e:
                print(f"  [diag] regex compile error: {e}")
        # ----------------------------------------------------------------

        model = PeftModel.from_pretrained(base_model, str(ckpt_path))

        # --- Diagnostic: are LoRA modules actually attached? --------------
        n_lora = sum(1 for n, _ in model.named_modules()
                     if "lora_A" in n or "lora_B" in n)
        print(f"  [diag] after PeftModel.from_pretrained: {n_lora} lora_A/B modules attached")
        if n_lora == 0:
            print("  [ERROR] adapter loaded but ZERO LoRA modules attached. "
                  "target_modules regex likely doesn't match the runtime model "
                  "structure. Inference will run the BASE model (no AffordGen format).")
        # ----------------------------------------------------------------

        # Merge adapter into base for inference-time speed; do not call save.
        model = model.merge_and_unload()
        model.eval()
        return model, processor

    print(f"Loading full-finetune VLM from {ckpt} ...")
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        ckpt,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def vlm_predict(model, processor, image: Image.Image, description: str,
                current_iou: float, device: str) -> str:
    # Strip the leading "<image>\n" placeholder: with structured content the
    # image is supplied as an image item, and the chat template injects the
    # vision tokens itself (matching training, where swift expands "<image>"
    # to "<|vision_start|><|image_pad|><|vision_end|>").
    prompt = USER_PROMPT.format(description=description).removeprefix("<image>\n")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text",  "text": prompt},
    ]}]
    # enable_thinking=False is critical for LoRA: Qwen3 chat template by default
    # opens a <think> tag at the assistant turn, which a LoRA adapter cannot
    # override since only ~248 linear modules are tuned. Training data has no
    # thinking content in the answer, so this keeps inference aligned with
    # training. Full finetune happens to ignore this prompt-side bias because
    # all weights move, but LoRA does not.
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    # Workaround: the multimodal processor may not forward enable_thinking to
    # the underlying tokenizer. Manually rewrite the trailing "<think>\n" the
    # template emits at the start of the assistant turn so the model sees an
    # empty already-closed thinking block.
    if text.endswith("<think>\n"):
        text = text[: -len("<think>\n")] + "<think>\n\n</think>\n\n"
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    generated = out_ids[0][inputs["input_ids"].shape[1]:]
    return processor.decode(generated, skip_special_tokens=True)


# ── SAM3 inference ─────────────────────────────────────────────────────────────
def build_sam3(ckpt: str, gpu_id: int):
    sys.path.insert(0, str(Path(__file__).parent / "third_party" / "sam3"))
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.cuda.set_device(gpu_id)
    model = build_sam3_image_model(checkpoint_path=ckpt, enable_inst_interactivity=True)
    processor = Sam3Processor(model)
    return model, processor


def binary_mask_to_sam_logits(mask: np.ndarray, logit_scale: float = 20.0,
                              mask_input_size: int = 288) -> np.ndarray:
    """Convert an arbitrary-resolution binary mask to SAM-style low-res logits
    (1, S, S) suitable for the mask_input arg of predict_inst.

    SAM thresholds mask_input at 0.0 in logit space, so we map binary {0,1} to
    {-logit_scale/2, +logit_scale/2}. The default scale of 20 gives pos~+10,
    neg~-10, which matches the typical magnitude of SAM's own mask logits.

    mask_input_size defaults to 288 for SAM3 (its mask decoder operates on
    72x72 image embeddings, and mask_input passes through two stride-2 convs
    so the native input is 4x72 = 288). For SAM/SAM2 you'd use 256 (= 4x64).
    """
    import torch.nn.functional as F
    m = torch.from_numpy(mask.astype(np.float32))[None, None]  # (1,1,H,W)
    m_lr = F.interpolate(m, size=(mask_input_size, mask_input_size),
                         mode="bilinear", align_corners=False)
    logits_lr = (m_lr - 0.5) * logit_scale
    return logits_lr.squeeze(0).numpy()  # (1, S, S)


def sam3_predict(sam3_model, sam3_processor, orig_image: Image.Image,
                 acc_cols, acc_rows, acc_labels, last_logits, thresh=0.49,
                 init_mask_logits=None):
    """If last_logits is None and init_mask_logits is provided, the init logits
    are used as mask_input for this prediction. Lets SAM3 refine on top of an
    externally-provided initial mask (e.g. a SAM3 text-prompt seed) rather
    than starting fresh from the first VLM click."""
    state = sam3_processor.set_image(orig_image)
    point_coords = np.array(list(zip(acc_cols, acc_rows)), dtype=np.float32)
    point_labels = np.array(acc_labels, dtype=np.int64)
    mask_input = last_logits if last_logits is not None else init_mask_logits
    with torch.no_grad():
        masks, scores, logits = sam3_model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            mask_input=mask_input,
        )
    scores_np = scores if isinstance(scores, np.ndarray) else scores.detach().cpu().numpy()
    masks_np  = masks  if isinstance(masks,  np.ndarray) else masks.detach().cpu().numpy()
    max_idx = int(np.argmax(scores_np))
    m = masks_np[max_idx]
    if m.ndim == 3:
        m = m[0]
    pred_mask = m > thresh
    return pred_mask, logits[[max_idx]]


# ── Per-image multi-step inference ────────────────────────────────────────────
def run_affordgen(orig_image: Image.Image, description: str,
                 vlm_model, vlm_processor, sam3_model, sam3_processor,
                 device: str, max_steps: int = 16, stop_iou: float = 0.97,
                 sam3_thresh: float = 0.49,
                 gt_mask: np.ndarray | None = None,
                 init_mask: np.ndarray | None = None,
                 init_pred_iou: float | None = None,
                 use_init_as_mask_input: bool = False):
    """If init_mask is provided (e.g. SAM3 text-prompt mask), the VLM sees it
    overlayed at step 0 and the loop tries to REFINE it via clicks. init_mask
    is also recorded as a pseudo "step -1" so best-step selection can fall
    back to it if all clicks make things worse.

    When use_init_as_mask_input=True, init_mask is ALSO converted to SAM-style
    low-res logits and passed to SAM3 as mask_input at step 0. This means
    SAM3's first prediction starts FROM the init mask + the first VLM click,
    rather than building a fresh mask from the click alone. Lets the VLM emit
    a single negative click to "subtract" a region (e.g., remove the lid from
    a trash-can mask) instead of having to first re-establish the positive
    region before subtracting.
    """
    W, H = orig_image.size
    acc_cols, acc_rows, acc_labels = [], [], []
    last_logits = None
    current_iou = init_pred_iou if init_pred_iou is not None else 0.0
    steps = []

    # Precompute mask_input logits from init_mask so step 0 of SAM3 starts from
    # the init region. Only used when use_init_as_mask_input is on.
    init_mask_logits = None
    if init_mask is not None and use_init_as_mask_input:
        init_mask_logits = binary_mask_to_sam_logits(init_mask.astype(bool))

    # Pre-populate steps with init_mask as a seed entry. We tag predicted_next_iou
    # with init_pred_iou so the post-loop argmax can pick the seed if clicks hurt.
    if init_mask is not None:
        seed_real_iou = None
        if gt_mask is not None:
            seed_real_iou, _, _ = compute_mask_iou_inter_union(init_mask.astype(bool), gt_mask)
        steps.append({
            "step_idx": -1,
            "col": None, "row": None, "is_positive": None,
            "mask": init_mask.astype(bool),
            "raw_output": "<init_mask>",
            "predicted_next_iou": init_pred_iou,
            "real_iou": seed_real_iou,
            "real_inter": None, "real_union": None,
            # Filled in at step 0 of the loop: VLM's own evaluation of the
            # SAM3 seed mask it sees in the overlay. Used by delta-gating
            # routing in infer_affordgen_multi_instance.py.
            "vlm_seed_score": None,
        })

    for step_idx in range(max_steps):
        # Build input image: step 0 = original (or init_mask overlay if seeded),
        # step k = previous step's mask overlay
        if step_idx == 0:
            if init_mask is not None:
                input_image = apply_green_overlay(orig_image, init_mask.astype(bool))
            else:
                input_image = orig_image
        else:
            input_image = apply_green_overlay(orig_image, steps[-1]["mask"])

        # VLM predict
        raw_output = vlm_predict(vlm_model, vlm_processor, input_image, description,
                                 current_iou, device)
        parsed = parse_vlm_output(raw_output)
        if parsed is None:
            print(f"  [step {step_idx}] VLM parse failed: {raw_output!r}")
            break

        # First loop iteration with init_mask: capture VLM's evaluation of the
        # SAM3 seed mask it's looking at (parsed["current_iou"]). This is what
        # delta-gating routing compares against the best click step's
        # predicted_next_iou — both are VLM-emitted so calibration matches.
        if step_idx == 0 and init_mask is not None and steps and steps[0].get("step_idx") == -1:
            steps[0]["vlm_seed_score"] = parsed.get("current_iou")

        col, row = denormalize(parsed["x_norm"], parsed["y_norm"], W, H)
        col = max(0, min(col, W - 1))
        row = max(0, min(row, H - 1))

        # Early stop: model stuck repeating a near-identical click. Adding a
        # duplicate point to SAM3's prompt does not change the mask, so the
        # loop just burns compute. Tolerance = 2 px to also catch jitter.
        if acc_cols:
            dup_label = (1 if parsed["is_positive"] else 0) == acc_labels[-1]
            dup_coord = abs(col - acc_cols[-1]) <= 2 and abs(row - acc_rows[-1]) <= 2
            if dup_label and dup_coord:
                print(f"  -> stop: step {step_idx} click ({col},{row}) duplicates "
                      f"previous click ({acc_cols[-1]},{acc_rows[-1]})")
                break

        acc_cols.append(col)
        acc_rows.append(row)
        acc_labels.append(1 if parsed["is_positive"] else 0)

        # SAM3 predict. At step 0 with use_init_as_mask_input, init_mask_logits
        # is consumed (last_logits is still None on the first iteration). After
        # step 0, last_logits takes over and init_mask_logits is ignored.
        pred_mask, last_logits = sam3_predict(
            sam3_model, sam3_processor, orig_image,
            acc_cols, acc_rows, acc_labels, last_logits, sam3_thresh,
            init_mask_logits=init_mask_logits,
        )

        predicted_next_iou = parsed["next_iou"]
        if gt_mask is not None:
            real_iou, real_inter, real_union = compute_mask_iou_inter_union(pred_mask, gt_mask)
        else:
            real_iou, real_inter, real_union = None, None, None
        steps.append({
            "step_idx": step_idx,
            "col": col, "row": row,
            "is_positive": parsed["is_positive"],
            "mask": pred_mask,
            "raw_output": raw_output,
            "current_iou": parsed.get("current_iou"),
            "predicted_next_iou": predicted_next_iou,
            "real_iou": real_iou,
            "real_inter": real_inter,
            "real_union": real_union,
        })

        real_iou_str = f"{real_iou:.4f}" if real_iou is not None else "n/a"
        print(f"  step {step_idx}: {'Pos' if parsed['is_positive'] else 'Neg'} "
              f"({col},{row})  pred_iou={predicted_next_iou}  real_iou={real_iou_str}")

        # Early stop on REAL IoU (only if GT available). Without GT we run all
        # max_steps so the demo still produces a full trajectory.
        current_iou = predicted_next_iou if predicted_next_iou is not None else 0.0
        if real_iou is not None:
            if real_iou >= stop_iou:
                print(f"  -> stop: real_iou {real_iou:.4f} >= {stop_iou}")
                break
            # Stop on first decline: real_iou dropped below the running peak.
            prev_ious = [s["real_iou"] for s in steps[:-1]
                         if s.get("real_iou") is not None]
            if prev_ious and real_iou < max(prev_ious):
                best_step = int(np.argmax(prev_ious))
                print(f"  -> stop: real_iou {real_iou:.4f} dropped from "
                      f"peak {max(prev_ious):.4f} at step {best_step}; "
                      f"trimming to steps[:{best_step + 1}]")
                steps = steps[:best_step + 1]
                break

    return steps


# ── Visualization ──────────────────────────────────────────────────────────────
def save_demo_figure(orig_image: Image.Image, steps: list,
                     description: str, out_path: Path,
                     gt_mask: np.ndarray | None = None,
                     final_iou: float | None = None):
    image_np = np.array(orig_image)
    n = len(steps)
    if n == 0:
        return

    # Panels: Original + n step panels + (optional) GT panel
    n_panels = 1 + n + (1 if gt_mask is not None else 0)
    ncols = min(n_panels, 6)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 3.5),
                             squeeze=False)
    fig.suptitle(f"{description}", fontsize=11, y=1.01)

    def draw_panel(ax, img_np, clicks_upto, title):
        ax.imshow(img_np)
        for k in range(clicks_upto + 1):
            r = steps[k]
            color = "#20d64f" if r["is_positive"] else "#ff3333"
            marker = "o" if r["is_positive"] else "x"
            lw = 1.2 if r["is_positive"] else 2.5
            ax.scatter([r["col"]], [r["row"]], s=80, c=color, marker=marker,
                       edgecolors="black" if r["is_positive"] else None,
                       linewidths=lw, zorder=5)
            ax.text(r["col"] + 3, r["row"] + 3, str(k), color="white",
                    fontsize=7, weight="bold",
                    bbox=dict(facecolor="black", alpha=0.55, boxstyle="round,pad=0.1"),
                    zorder=6)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    # Original image (no mask)
    ri, ci = 0, 0
    axes[ri][ci].imshow(image_np)
    axes[ri][ci].set_title("Original", fontsize=8)
    axes[ri][ci].axis("off")

    for step_idx, step in enumerate(steps):
        panel_idx = step_idx + 1
        ri, ci = panel_idx // ncols, panel_idx % ncols
        vis = overlay_vis(image_np, step["mask"])
        sign = "Pos" if step["is_positive"] else "Neg"
        real_iou_str = (f"{step['real_iou']:.4f}"
                        if step.get("real_iou") is not None else "n/a")
        title = (f"Step {step_idx} | {sign}\n"
                 f"{sign} ({step['col']}, {step['row']})\n"
                 f"real_iou_after_click={real_iou_str}")
        draw_panel(axes[ri][ci], vis, step_idx, title)

    # GT panel at the end
    if gt_mask is not None:
        panel_idx = n + 1
        ri, ci = panel_idx // ncols, panel_idx % ncols
        vis_gt = overlay_vis(image_np, gt_mask.astype(bool), color=(255, 50, 50), alpha=0.45)
        title = "GT" if final_iou is None else f"GT\nfinal_iou={final_iou:.2f}"
        axes[ri][ci].imshow(vis_gt)
        axes[ri][ci].set_title(title, fontsize=8)
        axes[ri][ci].axis("off")

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Manifest-driven evaluation ─────────────────────────────────────────────────
def run_with_manifest(args, out_dir: Path, device: str, gpu_id: int):
    """Evaluate using val_manifest.jsonl. Each line of the manifest is one
    (image, description, gt_mask) test case with paths relative to the manifest's
    parent directory. Every entry is guaranteed to have a valid GT mask, so no
    skip / lookup-fallback logic is needed."""
    import random

    manifest_path = Path(args.manifest)
    val_root = manifest_path.parent

    entries = [json.loads(l) for l in open(manifest_path) if l.strip()]
    print(f"Loaded manifest: {len(entries)} test cases from {manifest_path}")

    random.seed(args.seed)
    random.shuffle(entries)
    if args.n_images and args.n_images < len(entries):
        entries = entries[: args.n_images]
        print(f"Sampling first {len(entries)} (after seeded shuffle)")

    vlm_model, vlm_processor = build_vlm(args.ckpt, device)
    sam3_model, sam3_processor = build_sam3(args.sam3, gpu_id)

    metric_rows = []
    for i, e in enumerate(entries):
        img_path  = val_root / e["image_path"]
        mask_path = val_root / e["gt_mask_path"]
        description = e["description"]
        print(f"\n[{i+1}/{len(entries)}] {img_path.name} | '{description}'")

        try:
            orig_image = Image.open(img_path).convert("RGB")
            gt_mask = load_mask_bool(str(mask_path), image_size=orig_image.size)
            if gt_mask is None:
                print(f"  [ERROR] failed to load GT mask: {mask_path}")
                continue

            steps = run_affordgen(
                orig_image, description,
                vlm_model, vlm_processor, sam3_model, sam3_processor,
                device=device,
                max_steps=args.max_steps,
                stop_iou=args.stop_iou,
                sam3_thresh=args.sam3_thresh,
                gt_mask=gt_mask,
            )
            if not steps:
                print("  [skip] no predicted mask")
                continue

            step_ious  = [s["real_iou"]   for s in steps]
            step_inter = [s["real_inter"] for s in steps]
            step_union = [s["real_union"] for s in steps]
            best_step = int(np.argmax(step_ious))
            best_iou  = step_ious[best_step]
            final_iou = step_ious[-1]

            metric_row = {
                "image_name": e["image_name"],
                "description": description,
                "gt_mask_path": e["gt_mask_path"],
                "num_steps": len(steps),
                "best_step": best_step,
                "best_iou": best_iou,
                "final_iou": final_iou,
                "step0_iou": step_ious[0],
                "step0_success_at_50": step_ious[0] >= 0.5,
                "success_at_50": best_iou >= 0.5,
                "step_ious": step_ious,
                "best_inter": step_inter[best_step],
                "best_union": step_union[best_step],
                "step0_inter": step_inter[0],
                "step0_union": step_union[0],
                "final_inter": step_inter[-1],
                "final_union": step_union[-1],
            }
            metric_rows.append(metric_row)

            # Demo file. Use a unique name: stem + safe(desc) so a single image
            # with multiple descriptions doesn't overwrite itself.
            stem = Path(e["image_name"]).stem
            safe_desc = re.sub(r'[^a-z0-9]+', '_', description.lower()).strip('_')[:80]
            out_path = out_dir / f"{stem}__{safe_desc}_demo.png"
            save_demo_figure(orig_image, steps, description, out_path,
                             gt_mask=gt_mask, final_iou=best_iou)
            print(f"  best_iou={best_iou:.4f} (step {best_step}/{len(steps)-1})  "
                  f"final_iou={final_iou:.4f}  -> {out_path.name}")
        except Exception as ex:
            print(f"  [ERROR] {ex}")
            import traceback
            traceback.print_exc()

    _write_summary(metric_rows, len(entries), 0, out_dir)


def _write_summary(metric_rows, requested, skipped, out_dir):
    """Compute and persist gIoU/cIoU/P@50/P@50-95 for best-step / step-0 / final."""
    if not metric_rows:
        print("\n[WARN] No images produced metrics; nothing to write.")
        return

    best_ious   = np.array([r["best_iou"]    for r in metric_rows], dtype=np.float64)
    final_ious  = np.array([r["final_iou"]   for r in metric_rows], dtype=np.float64)
    step0_ious  = np.array([r["step0_iou"]   for r in metric_rows], dtype=np.float64)
    best_inter  = np.array([r["best_inter"]  for r in metric_rows], dtype=np.int64)
    best_union  = np.array([r["best_union"]  for r in metric_rows], dtype=np.int64)
    step0_inter = np.array([r["step0_inter"] for r in metric_rows], dtype=np.int64)
    step0_union = np.array([r["step0_union"] for r in metric_rows], dtype=np.int64)
    final_inter = np.array([r["final_inter"] for r in metric_rows], dtype=np.int64)
    final_union = np.array([r["final_union"] for r in metric_rows], dtype=np.int64)

    def ciou(inter_arr, union_arr):
        u = int(union_arr.sum())
        return float(inter_arr.sum() / u * 100.0) if u > 0 else 0.0

    def p50_95(ious):
        ts = np.linspace(0.50, 0.95, 10)
        return float(np.mean([(ious >= t).mean() for t in ts]) * 100.0)

    summary = {
        "evaluated_images": int(len(metric_rows)),
        "requested_images": int(requested),
        "skipped_metric_images": int(skipped),
        "gIoU":          float(best_ious.mean() * 100.0),
        "cIoU":          ciou(best_inter, best_union),
        "P@50":          float((best_ious >= 0.5).mean() * 100.0),
        "P@50-95":       p50_95(best_ious),
        "step0_gIoU":    float(step0_ious.mean() * 100.0),
        "step0_cIoU":    ciou(step0_inter, step0_union),
        "step0_P@50":    float((step0_ious >= 0.5).mean() * 100.0),
        "step0_P@50-95": p50_95(step0_ious),
        "refinement_gain": float((best_ious - step0_ious).mean() * 100.0),
        "final_gIoU":    float(final_ious.mean() * 100.0),
        "final_cIoU":    ciou(final_inter, final_union),
        "final_P@50":    float((final_ious >= 0.5).mean() * 100.0),
        "final_P@50-95": p50_95(final_ious),
    }
    with open(out_dir / "metrics_per_image.json", "w", encoding="utf-8") as f:
        json.dump(metric_rows, f, indent=2, ensure_ascii=False)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nMetrics:")
    print(f"  evaluated_images={summary['evaluated_images']} "
          f"skipped_metric_images={summary['skipped_metric_images']}")
    print(f"  best-step:   gIoU={summary['gIoU']:6.2f}  cIoU={summary['cIoU']:6.2f}  "
          f"P@50={summary['P@50']:6.2f}  P@50-95={summary['P@50-95']:6.2f}")
    print(f"  step-0 only: gIoU={summary['step0_gIoU']:6.2f}  cIoU={summary['step0_cIoU']:6.2f}  "
          f"P@50={summary['step0_P@50']:6.2f}  P@50-95={summary['step0_P@50-95']:6.2f}   <-- pure grounding")
    print(f"  refinement_gain (best - step0 gIoU): {summary['refinement_gain']:.2f}")
    print(f"  (ref) final:  gIoU={summary['final_gIoU']:6.2f}  cIoU={summary['final_cIoU']:6.2f}  "
          f"P@50={summary['final_P@50']:6.2f}  P@50-95={summary['final_P@50-95']:6.2f}")
    print(f"  metrics_summary -> {out_dir / 'metrics_summary.json'}")
    print(f"  metrics_per_image -> {out_dir / 'metrics_per_image.json'}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",  required=True, help="finetuned VLM checkpoint dir")
    ap.add_argument("--sam3",  required=True, help="sam3.pt checkpoint path")
    ap.add_argument("--manifest",
                    default="data/affordance/train_and_val/val_set/val_manifest.jsonl",
                    help="val_manifest.jsonl with pre-resolved image_path + gt_mask_path "
                         "per (image, description) test case. Recommended.")
    ap.add_argument("--val_jsonl",
                    default="data/affordance/affordgen_train/val.jsonl",
                    help="(LEGACY) val.jsonl to read descriptions from. "
                         "Used only if --manifest is empty.")
    ap.add_argument("--img_dir",
                    default="data/affordance/train_and_val/val_set/images",
                    help="(LEGACY) directory of val images, used only if --manifest is empty.")
    ap.add_argument("--out_dir", default=None,
                    help="output dir; defaults to <ckpt_parent>/<ckpt_name>-infer-result")
    ap.add_argument("--n_images", type=int, default=20,
                    help="number of images to run (random sample)")
    ap.add_argument("--gt_results_root",
                    default="data/affordance/image_mask_task_prompt",
                    help="root containing */results.json with GT mask_path records; empty disables metrics")
    ap.add_argument("--gt_results_json", action="append", default=[],
                    help="explicit image_mask_task_prompt results.json path; can be passed multiple times")
    ap.add_argument("--max_steps", type=int, default=16)
    ap.add_argument("--stop_iou", type=float, default=0.97,
                    help="stop early if REAL IoU(pred_mask, gt_mask) >= this")
    ap.add_argument("--sam3_thresh", type=float, default=0.49)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gpu_id = int(args.gpu.split(",")[0])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    if args.out_dir is None:
        ckpt_path = Path(args.ckpt).resolve()
        out_dir = ckpt_path.parent / f"{ckpt_path.name}-infer-result"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Manifest mode (preferred) ────────────────────────────────────────────
    # Manifest entries already pair each image with its correct GT mask, so
    # we skip all the val.jsonl/results.json lookup logic.
    if args.manifest and Path(args.manifest).exists():
        run_with_manifest(args, out_dir, device, gpu_id)
        return

    gt_results_root = args.gt_results_root if args.gt_results_root else None
    gt_index = load_gt_records(gt_results_root, args.gt_results_json)
    if gt_index:
        print(f"Loaded GT mask index for {len(gt_index)} image filenames")
    else:
        print("[WARN] No GT mask index loaded. Demo will run without gIoU/P@50 metrics.")

    # Load description map from val.jsonl
    print("Loading val descriptions ...")
    desc_map = load_val_descriptions(args.val_jsonl)
    print(f"  {len(desc_map)} step-0 entries found")

    # Find val images that have descriptions
    img_dir = Path(args.img_dir)
    all_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    # Match by filename to original paths in desc_map. We keep orig_path so
    # GT lookup can use the full path (basename collides across scenes).
    fname_to_desc = {}
    for orig_path, desc in desc_map.items():
        fname = Path(orig_path).name
        fname_to_desc[fname] = (orig_path, desc)

    matched = [(img, fname_to_desc[img.name][0], fname_to_desc[img.name][1])
               for img in all_imgs if img.name in fname_to_desc]

    if not matched:
        # fallback: use all images without descriptions
        matched = [(img, "", "the target object") for img in all_imgs]
        print(f"  No description match found, using {len(matched)} images with generic prompt")
    else:
        print(f"  {len(matched)} val images matched with descriptions")

    import random
    random.seed(args.seed)
    random.shuffle(matched)
    selected = matched[:args.n_images]

    # Build models
    vlm_model, vlm_processor = build_vlm(args.ckpt, device)
    sam3_model, sam3_processor = build_sam3(args.sam3, gpu_id)

    # Run inference
    metric_rows = []
    missing_gt_rows = []
    skipped_metric = 0
    for i, (img_path, orig_image_path, description) in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}] {img_path.name} | '{description}'")
        try:
            orig_image = Image.open(img_path).convert("RGB")

            # Resolve GT up front so it can drive real-IoU early stop and
            # render into the demo figure. Look up by full original path
            # (basename collides across scenes in RAGNet).
            gt_rec, gt_status, available_parts = pick_gt_record(
                gt_index, orig_image_path, description
            )

            if gt_rec is None:
                missing_gt_rows.append({
                    "image_name": img_path.name,
                    "orig_image_path": orig_image_path,
                    "description": description,
                    "status": gt_status,
                    "available_part_names": available_parts,
                })
                if gt_status == "no_image_record":
                    print(f"  [GT MISSING] no record for image_path in results.json")
                    print(f"               orig_image_path={orig_image_path}")
                else:  # no_part_match
                    print(f"  [GT MISSING] image has {len(available_parts)} record(s) "
                          f"but none match description '{description}'")
                    print(f"               orig_image_path={orig_image_path}")
                    print(f"               available part_names: {available_parts}")
            gt_mask = None
            if gt_rec is not None:
                gt_mask = load_mask_bool(gt_rec["mask_path"], image_size=orig_image.size)

            steps = run_affordgen(
                orig_image, description,
                vlm_model, vlm_processor, sam3_model, sam3_processor,
                device=device,
                max_steps=args.max_steps,
                stop_iou=args.stop_iou,
                sam3_thresh=args.sam3_thresh,
                gt_mask=gt_mask,
            )

            metric_row = None
            best_iou = None
            if gt_mask is not None and steps:
                step_ious = [s["real_iou"] for s in steps]
                step_inter = [s["real_inter"] for s in steps]
                step_union = [s["real_union"] for s in steps]
                best_step = int(np.argmax(step_ious))
                best_iou = step_ious[best_step]
                final_iou = step_ious[-1]
                metric_row = {
                    "image_name": img_path.name,
                    "description": description,
                    "gt_mask_path": gt_rec["mask_path"],
                    "gt_part_name": gt_rec.get("part_name", ""),
                    "num_steps": len(steps),
                    "best_step": best_step,
                    "best_iou": best_iou,
                    "final_iou": final_iou,
                    "step0_iou": step_ious[0],
                    "step0_success_at_50": step_ious[0] >= 0.5,
                    "success_at_50": best_iou >= 0.5,
                    "step_ious": step_ious,
                    # Pixel sums needed for cIoU aggregation.
                    "best_inter": step_inter[best_step],
                    "best_union": step_union[best_step],
                    "step0_inter": step_inter[0],
                    "step0_union": step_union[0],
                    "final_inter": step_inter[-1],
                    "final_union": step_union[-1],
                }

            out_path = out_dir / f"{img_path.stem}_demo.png"
            save_demo_figure(orig_image, steps, description, out_path,
                             gt_mask=gt_mask, final_iou=best_iou)
            print(f"  -> saved: {out_path}")

            if metric_row is not None:
                metric_rows.append(metric_row)
                print(f"  [metric] best_iou={metric_row['best_iou']:.4f} "
                      f"(step {metric_row['best_step']}/{metric_row['num_steps']-1}) "
                      f"final_iou={metric_row['final_iou']:.4f} "
                      f"P@50={metric_row['success_at_50']}")
            else:
                skipped_metric += 1
                if gt_rec is None:
                    # Already logged + saved above to missing_gt_rows
                    print(f"  [metric] skipped: GT missing ({gt_status})")
                elif not steps:
                    print("  [metric] skipped: no predicted mask")
                else:
                    print("  [metric] skipped: GT mask unreadable")
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()

    if missing_gt_rows:
        out_missing = out_dir / "missing_gt.json"
        with open(out_missing, "w", encoding="utf-8") as f:
            json.dump(missing_gt_rows, f, indent=2, ensure_ascii=False)
        n_no_image = sum(1 for r in missing_gt_rows if r["status"] == "no_image_record")
        n_no_part  = sum(1 for r in missing_gt_rows if r["status"] == "no_part_match")
        print(f"\n[GT AUDIT] {len(missing_gt_rows)} val samples have no matching GT")
        print(f"  no_image_record (image_path absent in results.json): {n_no_image}")
        print(f"  no_part_match   (image present, part_name not found): {n_no_part}")
        print(f"  full log -> {out_missing}")

    if metric_rows:
        best_ious  = np.array([r["best_iou"]    for r in metric_rows], dtype=np.float64)
        final_ious = np.array([r["final_iou"]   for r in metric_rows], dtype=np.float64)
        step0_ious = np.array([r["step0_iou"]   for r in metric_rows], dtype=np.float64)
        best_inter  = np.array([r["best_inter"]  for r in metric_rows], dtype=np.int64)
        best_union  = np.array([r["best_union"]  for r in metric_rows], dtype=np.int64)
        step0_inter = np.array([r["step0_inter"] for r in metric_rows], dtype=np.int64)
        step0_union = np.array([r["step0_union"] for r in metric_rows], dtype=np.int64)
        final_inter = np.array([r["final_inter"] for r in metric_rows], dtype=np.int64)
        final_union = np.array([r["final_union"] for r in metric_rows], dtype=np.int64)

        def ciou(inter_arr, union_arr):
            u = int(union_arr.sum())
            return float(inter_arr.sum() / u * 100.0) if u > 0 else 0.0

        def p50_95(ious):
            # COCO-style: mean of P@T for T in 0.50, 0.55, ..., 0.95 (10 thresholds)
            thresholds = np.linspace(0.50, 0.95, 10)
            return float(np.mean([(ious >= t).mean() for t in thresholds]) * 100.0)

        summary = {
            "evaluated_images": int(len(metric_rows)),
            "requested_images": int(len(selected)),
            "skipped_metric_images": int(skipped_metric),

            # Best-step metrics (image-level: pick best IoU per image)
            "gIoU":        float(best_ious.mean() * 100.0),       # mean of per-image IoU
            "cIoU":        ciou(best_inter, best_union),           # pixel-pooled IoU
            "P@50":        float((best_ious >= 0.5).mean() * 100.0),
            "P@50-95":     p50_95(best_ious),

            # Step-0-only (pure grounding, no refinement)
            "step0_gIoU":    float(step0_ious.mean() * 100.0),
            "step0_cIoU":    ciou(step0_inter, step0_union),
            "step0_P@50":    float((step0_ious >= 0.5).mean() * 100.0),
            "step0_P@50-95": p50_95(step0_ious),

            # How much refinement gains over step 0
            "refinement_gain": float((best_ious - step0_ious).mean() * 100.0),

            # Final step (last predicted step), for reference
            "final_gIoU":    float(final_ious.mean() * 100.0),
            "final_cIoU":    ciou(final_inter, final_union),
            "final_P@50":    float((final_ious >= 0.5).mean() * 100.0),
            "final_P@50-95": p50_95(final_ious),
        }
        with open(out_dir / "metrics_per_image.json", "w", encoding="utf-8") as f:
            json.dump(metric_rows, f, indent=2, ensure_ascii=False)
        with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print("\nMetrics:")
        print(f"  evaluated_images={summary['evaluated_images']} skipped_metric_images={summary['skipped_metric_images']}")
        print(f"  best-step:   gIoU={summary['gIoU']:6.2f}  cIoU={summary['cIoU']:6.2f}  "
              f"P@50={summary['P@50']:6.2f}  P@50-95={summary['P@50-95']:6.2f}")
        print(f"  step-0 only: gIoU={summary['step0_gIoU']:6.2f}  cIoU={summary['step0_cIoU']:6.2f}  "
              f"P@50={summary['step0_P@50']:6.2f}  P@50-95={summary['step0_P@50-95']:6.2f}   <-- pure grounding")
        print(f"  refinement_gain (best - step0 gIoU): {summary['refinement_gain']:.2f}")
        print(f"  (ref) final:  gIoU={summary['final_gIoU']:6.2f}  cIoU={summary['final_cIoU']:6.2f}  "
              f"P@50={summary['final_P@50']:6.2f}  P@50-95={summary['final_P@50-95']:6.2f}")
        print(f"  metrics_summary -> {out_dir / 'metrics_summary.json'}")
        print(f"  metrics_per_image -> {out_dir / 'metrics_per_image.json'}")
    else:
        print("\n[WARN] No images had both predictions and GT masks; metrics were not computed.")

    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
