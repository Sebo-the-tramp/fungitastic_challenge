import torch
import torch.nn.functional as F

from pathlib import Path
from rich.console import Console
from rich.table import Table
from utils import seed_everything, load_shards

SEED = 7
BACKBONE = "dinov3-vit7b16-pretrain-lvd1689m"
IMAGE_SIZE = 224
BACKGROUND_AUG = "normal"
FINAL_LAYER_CLASSIFIER_METHOD = "prototype"
MODEL_TRAIN = Path(f"facebook/{BACKBONE}/bfloat16_{BACKGROUND_AUG}_{IMAGE_SIZE}/train")
MODEL_TEST = Path(f"facebook/{BACKBONE}/bfloat16_{BACKGROUND_AUG}_{IMAGE_SIZE}/test")
MODEL_VAL = Path(f"facebook/{BACKBONE}/bfloat16_{BACKGROUND_AUG}_{IMAGE_SIZE}/val")

ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")


def create_prototypes(cls_tokens, labels):
    unique_classes = labels.unique()
    prototypes = torch.stack([cls_tokens[labels == c].mean(dim=0) for c in unique_classes])
    return prototypes, unique_classes


def cosine_accuracy_topk(test_image_norm, test_labels, protos_norm, prototype_labels, k=5):
    cosine_similarities = torch.mm(test_image_norm, protos_norm.T)
    topk_values, topk_indices = cosine_similarities.topk(k, dim=1)
    topk_labels = prototype_labels[topk_indices]
    correct = (topk_labels == test_labels.unsqueeze(1)).any(dim=1).sum().item()
    accuracy = correct / len(test_labels)
    mean_relative_top1_distances = (topk_values[:, :1] - topk_values[:, 1:]).mean(dim=0)

    return accuracy, mean_relative_top1_distances, 0.0

def single_prototypes(X_train_image_features, y_train, X_test_image_features, y_test):
    X_prototypes, y_prototypes = create_prototypes(X_train_image_features, y_train)    
    test_image_norm = F.normalize(X_test_image_features, p=2, dim=1)
    proto_norm = F.normalize(X_prototypes, p=2, dim=1)
    cosine_acc_top1, mean_relative_top1_distances, balanced_accuracy = cosine_accuracy_topk(test_image_norm, y_test, proto_norm, y_prototypes, k=1)
    print(f"Cosine similarity classification accuracy (top-1): {cosine_acc_top1:.4f}")
    print(f"Cosine similarity classification BALANCED accuracy (top-1): {balanced_accuracy:.4f}")

    cosine_acc_top5, mean_relative_top1_distances, balanced_accuracy = cosine_accuracy_topk(test_image_norm, y_test, proto_norm, y_prototypes, k=5)
    print(f"Cosine similarity classification accuracy (top-5): {cosine_acc_top5:.4f}")
    print(f"Cosine similarity classification BALANCED accuracy (top-1): {balanced_accuracy:.4f}")


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = ROOT / MODEL_TRAIN
    val_path = ROOT / MODEL_VAL
    test_path = ROOT / MODEL_TEST

    train_data = load_shards(train_path)
    val_data = load_shards(val_path)
    test_data = load_shards(test_path)
    
    # only use cls tokens for the prototype
    X_train, y_train = train_data["cls_tokens"], train_data["labels"]
    X_val, y_val = val_data["cls_tokens"], val_data["labels"]
    X_test, y_test = test_data["cls_tokens"], test_data["labels"]
   
    full_train_data = torch.cat([X_train, X_val])
    full_train_labels = torch.cat([y_train, y_val])

    # prototypes, unique_classes = create_prototypes(full_train_data, full_train_labels)

    single_prototypes(full_train_data, full_train_labels, X_test, y_test)

    



if __name__ == "__main__":
    main()
