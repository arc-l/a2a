<h1 align="center">Affordance2Action (A2A)</h1>
<p align="center"><b>Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipulation</b></p>

<p align="center">
  <a href="https://arc-l.github.io/a2a/"><img src="https://img.shields.io/badge/Project-Page-1e3a8a"></a>
  <img src="https://img.shields.io/badge/Code-Coming%20Soon-f97316">
  <img src="https://img.shields.io/badge/Paper-Coming%20Soon-6b7280">
</p>

---

> ## 🚧 Code Coming Soon
>
> The code for **A2A-AffordGen**, **A2A-GroundingModel**, **A2A-Policy**, and the **A2A-Bench**
> dataset is being cleaned up for release. **Please stay tuned** — we will update this repository
> as soon as it is ready. ⭐ Star/Watch the repo to get notified.

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

## Components (to be released)

| Component | Description | Status |
|---|---|---|
| **A2A-Bench** | Scene-level, task-conditioned, one-to-many affordance benchmark | 🚧 Coming soon |
| **A2A-AffordGen** | Agent-assisted annotation pipeline | 🚧 Coming soon |
| **A2A-GroundingModel** | Real-time task-conditioned part grounding (SAM3-based) | 🚧 Coming soon |
| **A2A-Policy** | Manipulation policy guided by affordance priors | 🚧 Coming soon |

## Project page (this repo)

This repository currently hosts the project website (`index.html` + `static/`), deployed via
GitHub Pages. To preview locally:

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

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
