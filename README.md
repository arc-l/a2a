<h1 align="center">Affordance2Action (A2A)</h1>
<p align="center"><b>Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipulation</b></p>

<p align="center">
  <a href="https://arc-l.github.io/a2a/"><img src="https://img.shields.io/badge/Project-Page-1e3a8a"></a>
  <img src="https://img.shields.io/badge/Code-Partial%20Release-f97316">
  <img src="https://img.shields.io/badge/Paper-Coming%20Soon-6b7280">
</p>

---

> ## 🚀 Code Release (in progress)
>
> **A2A-AffordGen** — our agent-assisted annotation pipeline — is now available in
> [`A2A-AffordGen/`](./A2A-AffordGen). **Training data and model weights are available on
> [Rutgers Box](https://rutgers.box.com/s/inc77v41aehwfdyp6utwmxpmhrwo1j2p)**:
> the grounding training pool, AffordGen training records, **A2A-GroundingModel**, and a
> **LIBERO-Object masked_tokens policy**. Validation/test annotations and the full
> grounding/policy training and inference integration remain pending. See the component
> cards below for the exact scope of this release.

---

## About

**Affordance2Action (A2A)** is a benchmark-centered learning framework for scene-level,
task-conditioned part affordance grounding. Task-conditioned manipulation requires grounding
instructions to task-relevant functional parts rather than object categories — a setting that is
scene-dependent and often *one-to-many*: the same object may afford different interactions across
tasks, while a single task may correspond to multiple valid functional regions.

At its core is **A2A-Bench**, a manipulation-oriented benchmark that covers both single-region and
multi-region instruction correspondences in everyday scenes. To build it at scale, we develop
**A2A-AffordGen**, an agent-assisted annotation pipeline. A2A-Bench's supervision enables a
real-time grounding model (**A2A-GroundingModel**) and a manipulation policy (**A2A-Policy**).

- 🌐 **Project page:** https://arc-l.github.io/a2a/

## Components

| Component | Description | Status |
|---|---|---|
| **A2A-AffordGen** | Agent-assisted annotation pipeline | ✅ [Released](./A2A-AffordGen) |
| **A2A-Bench** | Scene-level, task-conditioned, one-to-many affordance benchmark | [Training pool released](./A2A-Bench#download); evaluation data pending |
| **A2A-GroundingModel** | Task-conditioned part grounding, implemented with SAM3-I | [Weights and model notes](./A2A-GroundingModel); code integration pending |
| **A2A-Policy** | Manipulation policy guided by affordance priors | [LIBERO simulation checkpoint](./A2A-Policy); real-robot policy pending |

## Downloads

[Download training data and weights from Rutgers Box](https://rutgers.box.com/s/inc77v41aehwfdyp6utwmxpmhrwo1j2p)
(approximately 37.4 GB total).

| Artifact | Contents |
|---|---|
| `A2A-Training-assets-001-of-015.zip` through `015-of-015.zip`, plus `A2A-Training-metadata.zip` | 43,179 grounding training records and 149,134 AffordGen training records; 174,497 shared image/mask assets |
| `A2A-GroundingModel.pt` | Fine-tuned grounding weights, with optimizer state removed |
| `a2a_policy_libero_masked_tokens.pt` | LIBERO-Object Arm A1 masked_tokens policy, with configuration and action normalization |
| `README.txt`, `SHA256SUMS`, `release-manifest.json` | Extraction instructions, file checksums, and release inventory |

Extract all 16 training ZIPs into the same directory; follow the
[training-data instructions](./A2A-Bench#download) to resolve the shared asset paths.
The released policy is a simulation variant; it is not the Piper Explicit checkpoint
or a reproduction package for the paper's headline manipulation results.
Source dataset splits and third-party terms are retained; this release does not certify
image-level train/validation disjointness or assign a new license to upstream assets.

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
