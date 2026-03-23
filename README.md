# Fungitastic experiments

## Structure of the paper

### Teaser thing

2 graphs
- 200 classes+oracle segmentation -> all dinov3 suite, no-segmentation, oracle segmentation, sam-3 segmentation with MLP classification
- 2000 classes+sam3 segmentation -> all dinov3 suite, no-segmentation, sam-3 segmentation with MLP classification

(segmentation -> patch segmented mean, whole mean and registers stuff)

### ablation different methods with single class/feature tokens

- CLIP, siglip
- MAE
- dino1/dino2/dino3
- InternVit

### Ablation on the different methods of classification
- protoype
- knn 
- linear probing
- MLP