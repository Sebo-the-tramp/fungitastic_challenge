#!/bin/bash

sweep_dim=(
    128
    256
    # 512
    1024
    2056
)

for dim in "${sweep_dim[@]}"; do
    echo "Running prototypes_pca_white_fast.py with DIM=${dim}"
    DIM="${dim}" python prototypes_pca_white_fast.py
done