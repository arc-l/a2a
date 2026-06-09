<div align="center">

# A2A-AffordGen

**Agent-assisted annotation pipeline for task-conditioned part-affordance data**

Part of [**Affordance2Action (A2A)**](https://arc-l.github.io/a2a/) ·
[Project Page](https://arc-l.github.io/a2a/)

</div>

---

A2A-AffordGen is the data engine behind **A2A-Bench**. It scales up
scene-level, task-conditioned, one-to-many part-affordance annotation by pairing
two models:

1. **A SegAgent annotator** — a Qwen vision-language model fine-tuned to *act* as
   an interactive segmentation agent. Given an image, a part description, and the
   current (semi-transparent green) mask, it predicts the next **click point**
   that improves the mask. [SAM3](https://github.com/facebookresearch/sam3) turns
   each click into a mask, and the loop repeats until the mask converges. This
   imitates how a human annotator clicks to carve out a region, following
   [SegAgent (CVPR 2025)](https://github.com/aim-uofa/SegAgent) but with a SAM3
   backend and affordance parts.
2. **A labeler VLM** (Qwen3-VL-32B, served locally via vLLM) — given an object
   and a candidate part mask, it names the **part** and proposes up to five
   **affordance task instructions** (both object-referring and task-referring),
   which is what makes the benchmark one-to-many.

This folder contains everything needed to **(a) build the training/annotation
data, (b) train the SegAgent annotator, and (c) evaluate it.** It does *not*
ship model weights or datasets — see [Models & data](#models--data) for where to
get those.

> **Relationship to the other A2A components.** The grounding backbone (`sam3`)
> is released separately as **A2A-GroundingModel / SAM3-I**. A2A-AffordGen
> *uses* it; install it as a dependency (below). The benchmark this pipeline
> produces is **A2A-Bench**.

---

## Repository layout

```
A2A-AffordGen/
├── data_process/                     # (1) build data
│   ├── gen_object_crop_traj.py       #   part masks → cropped object → SAM3 click trajectory
│   ├── gen_part_captions.py          #   caption each part with the labeler VLM
│   ├── gen_segagent_train_data.py    #   trajectories → ms-swift JSONL (multi-step, mask overlays)
│   ├── gen_vlm_traj_data.py          #   trajectories → JSONL (single first-click variant)
│   ├── validate_train_data.py        #   sanity-check the generated JSONL
│   ├── build_val_gt_manifest.py      #   build the evaluation manifest (val_manifest.jsonl)
│   ├── build_val_overlap.py          #   add overlap/ambiguity info to the manifest
│   ├── build_v2_bbox_index.py        #   index Objects365-v2 bboxes (used by trajectory gen)
│   ├── data_checker.py               #   visual spot-checks of masks/trajectories
│   └── data_from_other_work/         #   agent-assisted expansion from external datasets
│       ├── start_vllm_qwen3.sh       #     launch the labeler VLM (vLLM, OpenAI API)
│       ├── filter_and_unify_other_data.py   # unify external masks + first-pass VLM labels
│       ├── cleaning_chain/           #     LLM/VLM quality control (judge & filter)
│       ├── orps_trps_chain/          #     object-referring (ORPS) + task-referring (TRPS) instructions
│       └── instruct_part_label/      #     variant of the chain for InstructPart-style inputs
├── train/                            # (2) train the annotator (ms-swift)
│   ├── train_segagent_full.sh        #   full fine-tune (DeepSpeed ZeRO-3)
│   ├── train_segagent_lora.sh        #   LoRA fine-tune (DeepSpeed ZeRO-2)
│   └── plot_training_curve.py        #   plot loss/eval curves from the run logs
├── inference/                        # (3) evaluate
│   ├── infer_segagent.py             #   SegAgent + SAM3 click loop, reports IoU vs GT
│   ├── run_infer_segagent_full.sh    #   wrapper for a full-finetune checkpoint
│   ├── run_infer_segagent_lora.sh    #   wrapper for a LoRA checkpoint
│   ├── infer_sam3_text_baseline.py   #   zero-shot SAM3+text baseline (no agent)
│   └── run_infer_sam3_text_baseline.sh
├── requirements.txt                  # core pip deps
└── env.full.txt                      # exact frozen environment (reference)
```

Every script reads its paths from CLI flags or environment variables; run any
script with `-h/--help`, or read its top-of-file docstring, for the full option
list. The defaults assume the [data/model layout](#models--data) below.

---

## Installation

```bash
conda create -n affordance python=3.10 -y
conda activate affordance

# 1) PyTorch — pick the build that matches your CUDA (see pytorch.org)
pip install torch torchvision

# 2) Core deps for this folder
pip install -r requirements.txt

# 3) Training backend
pip install ms-swift deepspeed

# 4) Labeler-VLM server (only needed for the VLM annotation steps)
pip install vllm

# 5) The grounding backbone used as the click backend.
#    Install the A2A-GroundingModel / SAM3-I package so that `import sam3` works.
#    (https://github.com/facebookresearch/sam3 — or the A2A-GroundingModel release)

# 6) SimpleClick, for the interactive Clicker used during trajectory generation.
#    Provides `isegm.inference.clicker.Clicker`.
#    (https://github.com/uncbiag/SimpleClick)
```

> The exact versions we ran are pinned in `env.full.txt`. `numpy<2` is required
> by both `sam3` and `ms-swift`.

---

## Models & data

Nothing large is committed to this repo. The script defaults expect this layout
(symlink your real locations here, or override every path via flags / env vars):

```
models/
├── Qwen3.5-9B/                 # VL backbone fine-tuned into the SegAgent annotator
├── Qwen3-VL-32B-Instruct/      # labeler VLM (served via vLLM)
└── sam3/sam3.pt                # SAM3 / A2A-GroundingModel checkpoint (click backend)

data/
└── affordance/
    ├── segagent_train/         # train.jsonl / val.jsonl (built by data_process)
    ├── train_and_val/val_set/  # images/, gt_mask/, val_manifest.jsonl (evaluation)
    └── ...                      # raw/intermediate data produced by the pipeline
```

| Asset | What it is | Where to get it |
|---|---|---|
| **A2A-Bench** | The released benchmark (images, masks, instructions, splits) | Released separately — see the [project page](https://arc-l.github.io/a2a/) |
| **SAM3 checkpoint** | Click backend + text baseline | A2A-GroundingModel / SAM3-I release |
| **Qwen3.5-9B**, **Qwen3-VL-32B-Instruct** | Base models | Their official model hubs |

---

## Quick start (TL;DR)

```bash
# A. Build the SegAgent training data from filtered click trajectories
python data_process/gen_segagent_train_data.py \
    --input_dir  data/affordance/trajs_dataset_after_filter \
    --output_dir data/affordance/segagent_train \
    --output_jsonl data/affordance/segagent_train/train.jsonl

# B. Train the annotator (LoRA is the cheaper/safer default)
BASE_MODEL=models/Qwen3.5-9B OUTPUT_DIR=runs/segagent_lora \
    bash train/train_segagent_lora.sh

# C. Evaluate it (agent + SAM3 click loop, reports IoU)
CKPT=runs/segagent_lora/checkpoint-XXXX \
    bash inference/run_infer_segagent_lora.sh
```

The rest of this README explains each stage in detail.

---

## 1. Data processing

There are three sub-pipelines. Most users only need **1.1 + 1.2**; **1.3** is
how we grew the benchmark by importing external affordance datasets.

### 1.1 Build SegAgent training data (click-trajectory imitation)

The annotator learns from *click trajectories*: sequences of (mask → click →
better mask) steps that reach a target part mask.

1. **Generate trajectories** — for each labeled part mask, crop to the full
   object box, then run a SAM3 click loop that keeps clicking until the IoU
   stops improving:
   ```bash
   python data_process/gen_object_crop_traj.py \
       --sam3_ckpt models/sam3/sam3.pt \
       --output_dir data/affordance/traj_and_cropped_images \
       --output_json data/affordance/traj_and_cropped_images/trajs_raw.json \
       --gpus 0
   ```
   We then keep only high-quality trajectories (final-step **IoU ≥ 0.7**); the
   filtered file is referred to below as `trajs_dataset_after_filter` /
   `trajs_filtered.json`.

2. **(optional) Re-caption parts** with the labeler VLM, so each trajectory has
   a clean natural-language part description (start the server first — see 1.3):
   ```bash
   python data_process/gen_part_captions.py \
       --input_dir  data/affordance/trajs_dataset_after_filter \
       --output_dir data/affordance/trajs_with_captions \
       --workers 4 --api_url http://localhost:8000/v1
   ```

3. **Convert to ms-swift JSONL** — turn every click step into one training
   example (image + green overlay of the previous mask → predict the next
   point). This produces `train.jsonl` / `val.jsonl`:
   ```bash
   python data_process/gen_segagent_train_data.py \
       --input_dir  data/affordance/trajs_dataset_after_filter \
       --output_dir data/affordance/segagent_train \
       --output_jsonl data/affordance/segagent_train/train.jsonl
   ```
   `gen_vlm_traj_data.py` is an alternative that emits only the first click per
   instance (no mask overlays) if you want a lighter, single-step dataset.

4. **Validate**:
   ```bash
   python data_process/validate_train_data.py --jsonl data/affordance/segagent_train/train.jsonl
   ```

### 1.2 Build the evaluation manifest

Evaluation reads a flat `val_manifest.jsonl` (one record per part query, with a
ground-truth mask). Build it from your val split:

```bash
python data_process/build_val_gt_manifest.py     # → val_manifest.jsonl
python data_process/build_val_overlap.py         # adds overlap/ambiguity fields
```

See [Data formats](#data-formats) for the manifest schema.

### 1.3 Agent-assisted expansion from external datasets (`data_from_other_work/`)

This is the pipeline that imports masks from external affordance datasets
(RAGNet, GraspNet, HANDAL, 3DOI, InstructPart, …) and turns them into A2A-style
one-to-many annotations using the labeler VLM.

```bash
# 0) Start the labeler VLM (OpenAI-compatible server on :8000)
VLM_MODEL=models/Qwen3-VL-32B-Instruct \
    bash data_process/data_from_other_work/start_vllm_qwen3.sh

# 1) Unify external masks to 0/1 and get a first pass of part + task labels
python data_process/data_from_other_work/filter_and_unify_other_data.py \
    --pkl   /path/to/source.pkl \
    --base-dir data/affordance/RAGNet/data \
    --output-dir runs/unify_out \
    --server-url http://localhost:8000

# 2) Quality control (judge mask quality / label correctness / object count,
#    then keep the good ones). See each script's --help:
#    cleaning_chain/{judge_quality, judge_affordance_label, judge_object_count_mllm,
#                    select_adaptive_quality_from_count, extract_yes_annotations,
#                    build_keep_quality_only_by_object_bbox, merge}.py

# 3) Generate the one-to-many instructions, then merge to the final part_list:
#    orps_trps_chain/build_annotation_input_from_count_filtered.py
#    orps_trps_chain/annotate_orps_from_affordance_mllm.py   # Object-Referring Part Seg
#    orps_trps_chain/annotate_trps_from_affordance_mllm.py   # Task-Referring   Part Seg
#    orps_trps_chain/merge_orps_trps_to_part_list.py
#    orps_trps_chain/split_final_annotations_by_count.py
```

`instruct_part_label/` is the same idea specialized for InstructPart-style
inputs. Each step is a small, single-purpose script with its own `argparse`
flags — read the top of the file for inputs/outputs.

---

## 2. Training

Training uses [ms-swift](https://github.com/modelscope/ms-swift) to fine-tune the
Qwen-VL backbone on `train.jsonl`. Two recipes are provided:

```bash
# LoRA (frozen backbone, recommended default — keeps base grounding ability)
BASE_MODEL=models/Qwen3.5-9B \
TRAIN_JSONL=data/affordance/segagent_train/train.jsonl \
OUTPUT_DIR=runs/segagent_lora \
CUDA_VISIBLE_DEVICES=0,1 \
    bash train/train_segagent_lora.sh

# Full fine-tune (DeepSpeed ZeRO-3)
BASE_MODEL=models/Qwen3.5-9B \
TRAIN_JSONL=data/affordance/segagent_train/train_step0_3x.jsonl \
OUTPUT_DIR=runs/segagent_full \
CUDA_VISIBLE_DEVICES=0,1 \
    bash train/train_segagent_full.sh
```

Override any of `BASE_MODEL`, `TRAIN_JSONL`, `VAL_JSONL`, `OUTPUT_DIR`,
`CUDA_VISIBLE_DEVICES`, `NPROC`, `MASTER_PORT` from the environment;
hyper-parameters live inside the scripts. Checkpoints are written to
`OUTPUT_DIR/checkpoint-*`.

```bash
# Plot loss / eval curves from a run (pass ms-swift's logging.jsonl)
python train/plot_training_curve.py runs/segagent_lora/<run>/logging.jsonl
```

> **Note on `train_step0_3x.jsonl`** — the full-finetune recipe oversamples the
> step-0 (no-overlay) examples ~3× so the model keeps grounding well on clean
> images; build it by concatenating the step-0 rows of `train.jsonl` twice. LoRA
> does not need this and trains directly on `train.jsonl`.

---

## 3. Testing / evaluation

### SegAgent annotator (agent + SAM3 click loop)

Runs the trained VLM and SAM3 over the manifest, performing up to `--max_steps`
clicks per image (stopping early at `--stop_iou`), and reports IoU against the
ground-truth masks:

```bash
CKPT=runs/segagent_lora/checkpoint-2332 \
SAM3_CKPT=models/sam3/sam3.pt \
MANIFEST=data/affordance/train_and_val/val_set/val_manifest.jsonl \
N_IMAGES=1167 \
    bash inference/run_infer_segagent_lora.sh     # or run_infer_segagent_full.sh
```

Or call the script directly for the full option list:

```bash
python inference/infer_segagent.py \
    --ckpt CKPT --sam3 models/sam3/sam3.pt \
    --manifest MANIFEST --n_images 100 --max_steps 8 --stop_iou 0.97 --gpu 0
```

### Zero-shot SAM3 + text baseline (lower bound)

```bash
N_IMAGES=1167 bash inference/run_infer_sam3_text_baseline.sh
```

---

## Data formats

**Training (`train.jsonl` / `val.jsonl`)** — one ms-swift chat record per click
step; `images` lists the image(s) referenced by the `<image>` token:

```json
{
  "messages": [
    {"role": "user", "content": "<image>\nPlease optimize the semi-transparent green mask ... The object description is as follows: <ref>handle of the jug</ref>"},
    {"role": "assistant", "content": "<ref>handle of the jug</ref> Current IOU: 0.94, Positive point: (412, 530), Predicted next IOU: 0.97"}
  ],
  "images": ["data/affordance/segagent_train/cropped_images/xxxx.jpg"]
}
```
Click coordinates are integers; `gen_segagent_train_data.py` normalizes them to
`[0, 1000]` (`x = col/W*1000`, `y = row/H*1000`).

**Evaluation manifest (`val_manifest.jsonl`)** — one record per part query:

```json
{
  "image_name": "1009786005_..._o-faucet-handle.jpg",
  "image_path": "images/1009786005_..._o-faucet-handle.jpg",
  "description": "handle of the faucet",
  "gt_mask_path": "gt_mask/.../handle_of_the_faucet.png",
  "gt_bbox": "[1600.0, 371.0, 274.0, 190.0]",
  "height": "1500", "width": "2000",
  "source_file": "instructpart_testA_..._filter.json"
}
```

**Raw trajectory (`trajs_*.json`)** — `{"info": {...}, "data": [ {image_name,
height, width, gt_ann{caption,object,part,segmentation(RLE),bbox,area},
clicks_list[{idx, coor:[row,col], is_positive, mask(RLE), iou}]} ]}`.

---

## Acknowledgements

This pipeline builds on [SegAgent](https://github.com/aim-uofa/SegAgent)
(CVPR 2025, BSD-2-Clause) for the human-like click-annotation formulation, uses
[SAM3](https://github.com/facebookresearch/sam3) as the segmentation backend and
[SimpleClick](https://github.com/uncbiag/SimpleClick) for the interactive
`Clicker`, and trains with [ms-swift](https://github.com/modelscope/ms-swift).
External affordance datasets imported via `data_from_other_work/` retain their
own licenses; please cite and comply with each source.

## Citation

```bibtex
@article{liu2026a2a,
  title   = {Affordance2Action: Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipulation},
  author  = {Liu, Litao and Han, Yifan and Yi, Pengfei and Yu, Wenbo and Wang, Hanqing and
             Du, Haoran and Yuan, Enze and Yuan, Zilin and Feng, Ruiding and Liu, Michael and
             Zhang, Qi and Yu, Jingjin},
  year    = {2026}
}
```
