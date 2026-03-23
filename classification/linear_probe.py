import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from pathlib import Path
from rich.table import Table
from rich.console import Console
from utils import seed_everything, load_shards

SEED = 7
MODEL_TRAIN = Path("facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_224/train")
MODEL_TEST = Path("facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_224/test")
MODEL_VAL = Path("facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_224/val")

ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")

def linear_probe(X_train, y_train, X_test, y_test): 
    from sklearn.linear_model import LogisticRegression 
    from sklearn.metrics import accuracy_score 
    
    X_train_np = X_train.cpu().numpy() 
    y_train_np = y_train.cpu().numpy() 
    X_test_np = X_test.cpu().numpy() 
    y_test_np = y_test.cpu().numpy() 

    clf = LogisticRegression(max_iter=1000, random_state=SEED) 
    clf.fit(X_train_np, y_train_np) 
    
    y_pred = clf.predict(X_test_np) 
    acc = accuracy_score(y_test_np, y_pred) 
    
    print(f"Linear probe classification accuracy: {acc:.4f}")

def linear_probe_torch(X_train, y_train, X_test, y_test, seed=42, epochs=100, lr=1e-2, weight_decay=1e-4):
    torch.manual_seed(seed)
    device = X_train.device

    X_train = X_train.detach()
    y_train = y_train.detach()
    X_test = X_test.detach()
    y_test = y_test.detach()

    num_classes = int(torch.max(y_train).item()) + 1
    input_dim = X_train.shape[1]

    clf = nn.Linear(input_dim, num_classes).to(device)

    optimizer = optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        clf.train()
        logits = clf(X_train)
        loss = criterion(logits, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    clf.eval()
    with torch.no_grad():
        logits = clf(X_test)
        y_pred = logits.argmax(dim=1)
        acc = (y_pred == y_test).float().mean().item()

    print(f"Linear probe classification accuracy: {acc:.4f}")

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

    linear_probe(full_train_data, full_train_labels, X_test, y_test)
    linear_probe_torch(full_train_data.to(device), full_train_labels.to(device), X_test.to(device), y_test.to(device), seed=SEED, epochs=100, lr=1e-2, weight_decay=1e-4)


if __name__ == "__main__":
    main()
