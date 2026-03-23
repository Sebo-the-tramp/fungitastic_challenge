PYTHONPATH="/home/cavadalab/Documents/scsv/fungitastic2026_2/.venv/bin/python"

# image size -> 224
# gbatch --gpus 1 --time 2:00:00 --name dinov3-train MODEL_NAME=facebook/dinov3-vit7b16-pretrain-lvd1689m DATASET_SIZE=300 SPLIT=train $PYTHONPATH dinov3.py
# gbatch --gpus 1 --time 2:00:00 --name dinov3-val MODEL_NAME=facebook/dinov3-vit7b16-pretrain-lvd1689m DATASET_SIZE=300 SPLIT=val $PYTHONPATH dinov3.py
# gbatch --gpus 1 --time 2:00:00 --name dinov3-test MODEL_NAME=facebook/dinov3-vit7b16-pretrain-lvd1689m DATASET_SIZE=300 SPLIT=test $PYTHONPATH dinov3.py

# image size -> 448
gbatch --gpus 1 --time 2:00:00 --name dinov3-train MODEL_NAME=facebook/dinov3-vit7b16-pretrain-lvd1689m DATASET_SIZE=720 SPLIT=train $PYTHONPATH dinov3.py
gbatch --gpus 1 --time 2:00:00 --name dinov3-val MODEL_NAME=facebook/dinov3-vit7b16-pretrain-lvd1689m DATASET_SIZE=720 SPLIT=val $PYTHONPATH dinov3.py
gbatch --gpus 1 --time 2:00:00 --name dinov3-test MODEL_NAME=facebook/dinov3-vit7b16-pretrain-lvd1689m DATASET_SIZE=720 SPLIT=test $PYTHONPATH dinov3.py