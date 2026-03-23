import torch
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from rich.table import Table
from rich.console import Console

from utils import seed_everything, load_shards, remap_labels

SEED = 7
MODEL_TRAIN = Path("facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_224/train")
MODEL_TEST = Path("facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_224/test")
MODEL_VAL = Path("facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_224/val")

ROOT = Path("/home/cavadalab/Documents/scsv/fungitastic2026_2/data_processed")

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def mlp_train(X_train, y_train, X_test, y_test):

    model = MLP(input_dim=X_train.shape[1], hidden_dim=512, num_classes=len(torch.unique(y_train)))
    model.to(X_train.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 40
    batch_size = 64
    num_batches = (len(X_train) + batch_size - 1) // batch_size

    for epoch in tqdm(range(num_epochs), desc="Training MLP", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            inputs = X_train[start_idx:end_idx]
            labels = y_train[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()
        print(f"MLP classification accuracy: {accuracy:.4f}")


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_path = ROOT / MODEL_TRAIN
    val_path = ROOT / MODEL_VAL
    test_path = ROOT / MODEL_TEST

    train_data = load_shards(train_path)
    val_data = load_shards(val_path)
    test_data = load_shards(test_path)

    X_train = torch.cat([train_data["cls_tokens"]], dim=1).to(device)
    X_val = torch.cat([val_data["cls_tokens"]], dim=1).to(device)
    X_test = torch.cat([test_data["cls_tokens"]], dim=1).to(device)

    y_train = train_data["labels"].to(device)
    y_val = val_data["labels"].to(device)
    y_test = test_data["labels"].to(device)

    full_train_data = torch.cat([X_train, X_val])
    full_train_labels = torch.cat([y_train, y_val])

    full_train_labels, y_test = remap_labels(full_train_labels, y_test)

    mlp_train(full_train_data, full_train_labels, X_test, y_test)    


if __name__ == "__main__":
    main()
