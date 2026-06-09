# Third-party dependencies (not bundled)

A2A-AffordGen depends on a couple of external repositories. To keep this
repository free of other projects' source code, we **do not vendor them** — get
them as follows. They are gitignored so cloned code is never committed here.

| Dependency | How to get it | Used by |
|---|---|---|
| PyTorch · ms-swift · vLLM | `pip` (see [`../requirements.txt`](../requirements.txt) + README) | training / inference / labeler server |
| **SimpleClick** | `git clone https://github.com/uncbiag/SimpleClick third_party/SimpleClick` (MIT) | `data_process/gen_object_crop_traj.py` — the `Clicker` used to derive click trajectories |
| **sam3** (A2A-GroundingModel / SAM3-I) | install the A2A-GroundingModel / SAM3-I package, or clone its repo to `../sam3` | the click→mask backend (`import sam3`); also needs `models/sam3/sam3.pt` |

## SimpleClick

```bash
git clone https://github.com/uncbiag/SimpleClick third_party/SimpleClick
```

`gen_object_crop_traj.py` adds `third_party/SimpleClick` to `sys.path`, so after
cloning, `from isegm.inference.clicker import Clicker` resolves. Only the
`Clicker` class is used. Upstream: <https://github.com/uncbiag/SimpleClick> (MIT).

## sam3 (A2A-GroundingModel / SAM3-I)

Provides `sam3.model_builder` and `sam3.model.sam3_image_processor`. Install the
**A2A-GroundingModel / SAM3-I** package (released separately), or place its repo
at `../sam3`. Upstream SAM3: <https://github.com/facebookresearch/sam3>. You also
need the SAM3 checkpoint at `models/sam3/sam3.pt`.
