<div align="center">
  <h1>Training-Free FungiTastic Segmentation</h1>
  <h3>Fine-grained semantic segmentation in low-data regimes with SAM3 and DINOv3.</h3>
  <p><strong>Sebastian Cavada · Francesco Pelosin · Lapo Faggi</strong></p>
  <p><sub>Accepted at the 13th Workshop on Fine-Grained Visual Categorization, CVPR 2026</sub></p>
  <p><a href="https://arxiv.org/pdf/2605.22492">Paper</a></p>
</div>

---

## Overview

This repository contains the code for a training-free, two-stage FungiTastic baseline. SAM3 produces class-agnostic mushroom masks, while DINOv3 assigns fine-grained labels through prototype matching in the embedding space.

The method targets one-shot to few-hundred-shot settings without retraining the segmentation or feature-extraction backbones.

## Setup

The environment requires Python 3.12 and uses CUDA 12.8 builds of PyTorch.

```bash
uv sync
```

Run scripts from the repository root with `uv run python`. Dataset and output paths are configured as uppercase constants near the top of each script.

## Repository structure

- `segmentation/` — SAM3 mask generation and evaluation.
- `extraction/` — DINOv2, DINOv3, OpenCLIP, and InternViT feature extraction.
- `classification_paper/` — prototype matching and low-data experiments.
- `visualize/` — plots, tables, and class-distribution utilities.
- `paper/` — paper source and figures.
- `FungiTastic/` — dataset utilities and reference baselines.

## Paper

**Training-Free Fine-Grained Semantic Segmentations in Low Data Regimes: A FungiTastic Baseline**

Sebastian Cavada, Francesco Pelosin, and Lapo Faggi.

[arXiv:2605.22492](https://arxiv.org/abs/2605.22492) · [PDF](https://arxiv.org/pdf/2605.22492)

```bibtex
@article{cavada2026trainingfree,
  title={Training-Free Fine-Grained Semantic Segmentations in Low Data Regimes: A FungiTastic Baseline},
  author={Cavada, Sebastian and Pelosin, Francesco and Faggi, Lapo},
  journal={arXiv preprint arXiv:2605.22492},
  year={2026}
}
```
