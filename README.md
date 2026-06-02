# Affordance2Action (A2A) — Project Page

Project website for **Affordance2Action (A2A): Task-Conditioned Part Affordance Grounding for Real-Time Manipulation**.

A static, dependency-free academic project page (Nerfies-style). No build step.

## Structure

```
index.html              # The project page
static/css/style.css    # Styles
static/images/          # Figures (converted from the paper PDFs)
.nojekyll               # Serve static/ verbatim on GitHub Pages
```

## Preview locally

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Deploy on GitHub Pages

1. Commit and push to `github.com/arc-l/a2a`.
2. Repo **Settings → Pages → Build and deployment**: Source = *Deploy from a branch*, Branch = `main`, Folder = `/ (root)`.
3. The site will be served at `https://arc-l.github.io/a2a/`.

## Before publishing — fill in the placeholders in `index.html`

- **Authors & affiliations** — currently placeholder names (the paper is an anonymized CORL 2026 submission).
- **Link buttons** — `Paper`, `arXiv`, and `A2A-Bench` point to `#`; set real URLs. `Code` already points to this repo.
- **BibTeX** — update once the citation is finalized.

Figures were rendered from `Affordance_project_paper/figure/*.pdf` at 150 DPI.
