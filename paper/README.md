# Paper

This directory contains the current CVPR-style draft for the repo story:

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

The draft is tied to the runs already stored in `../dashboard/results.json`:

- frozen DINOv3 ViT-7B features
- oracle mask-aware token pooling
- `224` and `448` resolutions
- MLP token ablations only

It does not claim results for pending prototype, k-NN, linear-probe, or stratified analyses.
