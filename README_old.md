# Fungitastic experiments

## Scope

This repo currently supports feature-based closed-set experiments on top of extracted vision features.

The current practical focus is:
- DINOv3 image features
- oracle segmentation masks
- `224` and `448` image sizes
- simple downstream classifiers on frozen features

The first paper/story should stay narrow:

**When do oracle segmentation masks help fine-grained fungi classification?**

## Main questions

We want to answer:
- Does `masked patch pooling` beat plain `CLS` or plain `patch pooling`?
- Does the gain from masks increase at higher image resolution?
- Is the gain visible with simple classifiers, or only with an MLP?
- Which samples benefit most: small specimen area, cluttered background, rare classes?

## Phase 1: Must-run experiments

These are the runs that matter most and should happen first.

### 1. Resolution study
- Same experiment at `224`
- Same experiment at `448`
- Goal: measure whether higher resolution helps the model capture useful fungal structure

### 2. Token / feature study
- `cls`
- `patch`
- `masked`
- `cls+patch`
- `cls+patch_norm`
- `cls+masked`
- `cls+masked_norm`
- Goal: isolate whether the useful signal comes from global image context, full patch pooling, or mask-aware pooling

### 3. Classifier study
- prototype classifier
- k-NN
- linear probe
- MLP
- Goal: separate representation quality from classifier capacity

### 4. Stratified analysis
- head / medium / tail classes
- small / medium / large mask area
- easy vs hard species pairs
- Goal: understand when masks help, not only whether they help on average

## Phase 2: Strong follow-up directions

These are worth doing after Phase 1 is stable.

### Background study
- `normal`
- `crop`
- `crop_black`
- `masked_black`
- `masked_blurred`

Question:
- Are the gains from masked features actually about removing background clutter?

### Backbone study
- `dinov3-vits16`
- `dinov3-vits16plus`
- `dinov3-vitb16`
- `dinov3-vitl16`
- `dinov3-vith16plus`
- `dinov3-vit7b16`

Question:
- Do masks matter less as the backbone gets stronger?

### Patch-level study
- save full patch features
- test better pooling than simple mean
- test learned fusion on patch tokens

Question:
- Is simple mean pooling leaving performance on the table?

## Phase 3: Broader research directions

These are good extensions, but they should not dilute the first story.

### Multimodal fusion
- image + metadata
- image + climate
- image + location
- image + taxonomy

### Generalization tracks
- few-shot
- open-set
- chronological shift

### Error-aware evaluation
- taxonomic distance between prediction and target
- cost-aware errors
- confidence / calibration

## Metrics

The default report should include:
- top-1 accuracy
- macro accuracy
- top-5 accuracy
- accuracy by frequency bin
- accuracy by mask-area bin
- model size / feature dimension / runtime when relevant

Top-1 alone is not enough.

## Recommended order

- Fix the current path / extraction issues
- Run the full token study at `224`
- Run the full token study at `448`
- Add prototype, k-NN, and linear probe baselines
- Stratify the gains by class frequency and mask area
- Only then decide whether background or backbone sweeps are worth the compute

## Current blockers

- Masked pooling is currently tied to a hardcoded `224` mask resize, so masked `448` comparisons are not fully trustworthy yet
- Some downstream scripts still point to old feature directories
- Only `normal` background is currently active in extraction
- Only the `vit7b` backbone is currently enabled
- Full patch features are not being saved yet

## Decision

The main story for now is:

**oracle masks + token fusion + resolution**

Everything else is secondary until that story is clean and reproducible.
