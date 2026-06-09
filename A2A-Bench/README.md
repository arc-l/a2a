<div align="center">

# A2A-Bench

**A scene-level, task-conditioned, one-to-many part-affordance grounding benchmark**

Part of [**Affordance2Action (A2A)**](https://arc-l.github.io/a2a/) ·
[Project Page](https://arc-l.github.io/a2a/) ·
built with [A2A-AffordGen](../A2A-AffordGen)

</div>

---

> **Status: dataset card / format spec.** The annotations and images are being
> packaged for hosting on Hugging Face / ModelScope. This folder documents the
> schema, structure, and a reference loader so you can integrate A2A-Bench as
> soon as the data is up. Download links are filled in under
> [Download](#download). Maintainers: see the
> [pre-publish checklist](#maintainer-checklist-before-publishing).

A2A-Bench targets **task-conditioned part affordance grounding** in everyday,
multi-object scenes. Unlike grasp-only or object-level affordance datasets, it
grounds natural-language instructions to **functional parts**, and explicitly
covers the **one-to-many** setting: the same object can afford different
interactions across tasks, and a single task can map to *multiple* valid
functional regions in a cluttered scene.

Each annotation links an **object → part → affordance → action** and provides
**six instruction phrasings** that span the spectrum from *task-referring*
(implicit, "if I want to open the cabinet, which part …") to *object-referring*
(explicit, "the handle of the cabinet").

## Highlights

- **Real images**, natural multi-object scenes (not synthetic).
- **Part-level masks** paired with every annotation.
- **Six instruction styles** per part — task-referring ↔ object-referring.
- **Single- and multi-region** (one-to-many) instruction correspondences.
- A broad affordance vocabulary (*hold, support, cut, contain, grasp, grip,
  pull, lift, twist, press, pour, drag, pierce, display, cover, …*).

| Split | Content | Size |
|---|---|---|
| **InstructPart** (`all`) | 2,400 images · 2,400 annotated part instances · 2,400 masks · 6 instructions each | ~7 GB |
| **Evaluation manifest** | 1,167 part queries, each with a ground-truth mask | ~2 GB |
| **Training pool** | larger agent-annotated set (consolidated from the annotation batches) | _TODO: confirm final count_ |

> Numbers above are verified from the working set. The exact consolidated
> **release** composition (final dedup + split boundaries) is finalized by the
> maintainers before upload — see the checklist.

---

## Dataset structure

```
A2A-Bench/
├── InstructPart/
│   ├── data_all.json          # list of records (schema below)
│   ├── images/                # <image_path> RGB images
│   └── masks/                 # per-part binary masks
├── eval/
│   ├── val_manifest.jsonl     # one part query per line (schema below)
│   ├── images/
│   └── gt_mask/
└── train/
    └── annotations.json       # agent-annotated training records (same schema as data_all.json)
```

> The hosted archive may differ slightly in top-level naming; the **schemas**
> below are the contract that the reference loader and evaluation rely on.

### Annotation schema (`data_all.json`, `train/annotations.json`)

A JSON list; each record is one image with a `part_list` of one or more parts:

```json
{
  "image_path": "45514228_f89388bda9_o.jpg",
  "part_list": [
    {
      "object": "cabinet",
      "part": "handle",
      "affordance": "drag",
      "action": "open",
      "instruction": [
        "If I want to open the cabinet, which part in the picture should be used?",
        "Which part of the cabinet should I use to open it?",
        "Where is the handle of the cabinet in this image?",
        "Where is the handle of the cabinet that can be dragged in this image?",
        "handle of the cabinet",
        "handle of the cabinet that can be dragged"
      ]
    }
  ]
}
```

- `instruction[0:2]` — **task-referring** (implicit; the part is not named).
- `instruction[2:4]` — **object-referring** with location/affordance cues.
- `instruction[4:6]` — **short object-referring** phrases.
- Multiple entries in `part_list` (or multiple records sharing an image) express
  the **one-to-many** / multi-region cases.
- The matching mask lives in `masks/` (see the per-record/per-part naming used
  by the reference loader).

### Evaluation manifest (`val_manifest.jsonl`)

One JSON object per line, one part query each:

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

Grounding models are scored by mask IoU / P@0.5 of the predicted region against
`gt_mask_path`, given `description` and the image.

---

## Download

> _TODO (maintainer): publish and link the archives._

| Host | Link |
|---|---|
| 🤗 Hugging Face | `TODO: https://huggingface.co/datasets/<org>/A2A-Bench` |
| 🪄 ModelScope | `TODO: https://www.modelscope.cn/datasets/<org>/A2A-Bench` |

```bash
# Example (after the HF dataset is published)
huggingface-cli download <org>/A2A-Bench --repo-type dataset --local-dir ./A2A-Bench-data
```

## Loading

A dependency-light reference loader is provided:

```bash
python load_a2a_bench.py --root ./A2A-Bench-data --split instructpart --stats
```

See [`load_a2a_bench.py`](./load_a2a_bench.py) for iterating records, resolving
image/mask paths, and reading the evaluation manifest.

---

## Image sources & licensing

A2A-Bench annotations (part labels, masks, instructions) are an original
contribution of this work. The underlying **images are sourced from existing
datasets** and retain their original licenses:

- **Flickr** photos (filenames are Flickr photo IDs) — per-image Creative
  Commons licenses.
- **Objects365** and **COCO / RefCOCO**-derived images — under their respective
  dataset licenses.

Please cite and comply with each image source. We recommend releasing the
**annotations** under CC BY 4.0 and distributing images by reference / per their
source terms (or only where redistribution is permitted).

> _TODO (maintainer): finalize the annotation license and, per source,
> decide redistribute-vs-reference; add a `LICENSES.md` mapping image source →
> license._

---

## Maintainer checklist (before publishing)

- [ ] **Consolidate the release set** from the working dirs (`InstructPart/`,
      `assign_task/` merged annotations, `train_and_val/val_set/`) into the
      `InstructPart / eval / train` layout above; record final counts in the
      table.
- [ ] **Sanitize JSONs** — strip absolute/internal paths from annotation files
      (e.g. `orig_image_path`, `output_files` entries point at internal machine
      paths like `/home/.../...`); keep paths repo-relative.
- [ ] **Resolve image redistribution** per source (Flickr/Objects365/COCO);
      add `LICENSES.md`.
- [ ] **Choose the annotation license** (CC BY 4.0 recommended).
- [ ] **Upload** to Hugging Face and/or ModelScope; fill in the links above.
- [ ] **Verify** `python load_a2a_bench.py --root <downloaded> --stats` runs and
      counts match.

---

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
