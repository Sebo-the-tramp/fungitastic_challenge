#!/bin/bash

# sweep_dim=(
#     128
#     256
#     # 512
#     1024
#     2056
# )

# for dim in "${sweep_dim[@]}"; do
#     echo "Running prototypes_pca_white_fast.py with DIM=${dim}"
#     DIM="${dim}" python prototypes_pca_white_fast.py
# done


BACKBONES=(
    # dinov3-vit7b16-pretrain-lvd1689m
    # dinov3-vith16plus-pretrain-lvd1689m
    # dinov3-vits16plus-pretrain-lvd1689m
    dinov3-vitb16-pretrain-lvd1689m
    dinov3-vitl16-pretrain-lvd1689m
    dinov3-vits16-pretrain-lvd1689m

    dinov2-with-registers-giant
    dinov2-with-registers-small
    dinov2-with-registers-base
    dinov2-with-registers-large
)

for backbone in "${BACKBONES[@]}"; do
    echo "Running prototypes_pca_white_fast.py with DIM=${backbone}"
    BACKBONE="${backbone}" python prototypes_pca_white_fast.py
done