# Paper

This directory contains the current CVPR-style draft aligned to the repo README story:

**When do oracle segmentation masks help fine-grained fungal recognition?**

## Build

Run:

```bash
cd paper
./build.sh
```

The script writes the PDF to `paper/build/main.pdf`. If that file already exists, it asks before overwriting it.

## Tooling

The local build uses:

- `pdflatex`
- `bibtex`

## Scope

The draft is now organized around the paper structure in `../README.md`:

- teaser A: 200-class oracle-segmentation comparison across the DINOv3 suite with MLP classification
- teaser B: 2000-class SAM-3 comparison across the DINOv3 suite with MLP classification
- backbone ablation: CLIP, SigLIP, MAE, DINOv1, DINOv2, DINOv3, InternViT
- fusion ablation: shared-space gated combinations, with diversity-vs-quantity comparisons such as DINO+SigLIP versus larger mixtures
- classifier ablation: prototype, k-NN, linear probe, MLP

Current completed evidence already written into the draft:

- DINOv3 ViT-7B MLP runs at 224 and 448
- DINOv3 ViT-7B 224 k-NN sweep
- exploratory 224 register-token rows from the latest MLP rerun

Teaser A now uses the exported oracle-mask DINOv3 MLP plot under `paper/media/teaser/`. Teaser B remains a placeholder until the 2000-class SAM-3 plot is exported. Final paper-specific figures should live under `paper/media/`.
