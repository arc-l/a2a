<div align="center">

# A2A-Bench

**A scene-level, task-conditioned, one-to-many part-affordance grounding benchmark**

Part of [**Affordance2Action (A2A)**](https://arc-l.github.io/a2a/) ·
[Project Page](https://arc-l.github.io/a2a/) ·
built with [A2A-AffordGen](../A2A-AffordGen)

</div>

---

> **Status: training release available.** Download the grounding training pool
> and AffordGen training records from
> [Rutgers Box](https://rutgers.box.com/s/inc77v41aehwfdyp6utwmxpmhrwo1j2p).
> Validation and test annotations remain pending. The current archive layout and
> JSONL schema are documented under [Download](#download); the historical
> annotation/evaluation formats below are not the layout of this training release.

A2A-Bench targets **task-conditioned part affordance grounding** in everyday,
multi-object scenes. Unlike grasp-only or object-level affordance datasets, it
grounds natural-language instructions to **functional parts**, and explicitly
covers the **one-to-many** setting: the same object can afford different
interactions across tasks, and a single task can map to *multiple* valid
functional regions in a cluttered scene.

The historical annotation format links an **object → part → affordance → action**
and provides **six instruction phrasings** that span the spectrum from *task-referring*
(implicit, "if I want to open the cabinet, which part …") to *object-referring*
(explicit, "the handle of the cabinet"). The released training pool combines
multiple source datasets, and its number of task phrasings varies by record.

## Highlights

- **Real images**, natural multi-object scenes (not synthetic).
- **Part-level masks** paired with every annotation.
- **Six instruction styles** per part — task-referring ↔ object-referring.
- **Single- and multi-region** (one-to-many) instruction correspondences.
- A broad affordance vocabulary (*hold, support, cut, contain, grasp, grip,
  pull, lift, twist, press, pour, drag, pierce, display, cover, …*).

| Split | Content | Size |
|---|---|---|
| **Grounding training pool** | 43,179 records; 97,498 referenced images/masks, including available scene-instance masks | Released |
| **AffordGen training records** | 149,134 records; 108,309 referenced RGB/overlay images | Released |
| **Validation / test annotations** | Being organized separately | Pending |

> The two training datasets share 31,310 assets; the archives contain 174,497
> unique resources. These counts describe the released training manifests, not
> a finalized evaluation benchmark. Existing source splits are retained; source
> train/validation image overlap has not been redefined, and image-level
> disjointness is not certified.

---

## Historical annotation and evaluation formats

The following layout describes the earlier annotation/evaluation tooling. For
the released training ZIPs, use the [download layout](#download) and the
`training-pool` loader option below.

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

The legacy `instructpart`, `train`, and `eval` loader options use this layout.
Their evaluation annotations are not included in the current training download.

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

[Download training data and model weights from Rutgers Box](https://rutgers.box.com/s/inc77v41aehwfdyp6utwmxpmhrwo1j2p).

Download `A2A-Training-metadata.zip` and **all 15** files named
`A2A-Training-assets-001-of-015.zip` through `A2A-Training-assets-015-of-015.zip`.
Each ZIP is independently extractable; extract every ZIP into the same directory.
The folder also contains the model weights, `README.txt`, `release-manifest.json`,
`SHA256SUMS`, and a Windows checksum script. The complete data-and-model download
is approximately 37.4 GB; allow about 80 GB for download plus extraction.

```bash
# Linux, from the folder containing the downloaded files
sha256sum -c SHA256SUMS
mkdir -p extracted
for archive in A2A-Training-*.zip; do unzip -oq "$archive" -d extracted; done
```

For Windows PowerShell, see the downloaded `README.txt` and `verify-downloads.ps1`.
Every ZIP passed full CRC validation. Individual asset SHA256 hashes are included
in the metadata ZIP.

```text
extracted/A2A-Training/
├── A2A-Bench-training-pool/
│   ├── train.jsonl
│   ├── files.jsonl
│   ├── dataset_info.json
│   └── README.txt
├── A2A-AffordGen-train/
│   ├── train.jsonl
│   ├── files.jsonl
│   ├── dataset_info.json
│   └── README.txt
├── assets/                       # resources shared by both training datasets
└── asset-integrity.jsonl          # paths relative to A2A-Training/
```

Paths in each dataset's `train.jsonl` and `files.jsonl` are **relative to that
manifest's directory**, so `../assets/` resolves to the shared resource tree.
Resolve them before passing records to tools that require absolute paths:

```python
image_path = (manifest_path.parent / row["image_path"]).resolve()
```

Grounding records contain `image_path`, `mask_path`, `object_name`, `part_name`,
`part_phrase`, `affordance`, `tasks`, `source`, and `split`; some records also
contain `instance_mask_paths` and `scene_instructions`. `mask_path` is the target
union mask, and available per-instance masks are listed separately.
AffordGen records contain `messages` and `images`, with the same relative-path
convention. IoU values in AffordGen training messages are annotation-derived
supervision, not an online ground-truth measurement interface.

Only the existing training manifests and their referenced assets are included.
Validation/test annotations and final evaluation packaging remain pending.

## Loading

A dependency-light reference loader is provided:

```bash
python load_a2a_bench.py --root ./extracted/A2A-Training --split training-pool --stats
```

See [`load_a2a_bench.py`](./load_a2a_bench.py) for iterating the released grounding
records and resolving image/mask paths. Its other options support the historical
formats documented above.

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

## Remaining release work

- [x] Package the existing training manifests, with recorded counts and shared assets.
- [x] Replace internal asset paths with portable relative paths and verify all references.
- [x] Publish training resources and checksum manifests on Rutgers Box.
- [ ] Organize and publish validation/test annotations and document final split boundaries.
- [ ] Complete source-by-source attribution and licensing documentation.
- [ ] Finalize the annotation license; this release does not relicense upstream images.

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
