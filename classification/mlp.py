import torch

from tqdm import tqdm
from pathlib import Path

from utils import seed_everything, load_shards


SEED = 7
IMG_SIZE=448
MODEL_TRAIN = Path(f"facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_{IMG_SIZE}/train")
MODEL_TEST = Path(f"facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_{IMG_SIZE}/test")
MODEL_VAL = Path(f"facebook/dinov3-vit7b16-pretrain-lvd1689m/bfloat16_normal_{IMG_SIZE}/val")

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

def remap_labels_from_reference(y_reference, *y_target_sets):
    reference_labels = torch.unique(y_reference, sorted=True)

    for y_target in y_target_sets:
        assert torch.isin(y_target, reference_labels).all().item()

    label_map = {int(label): idx for idx, label in enumerate(reference_labels.tolist())}
    mapped_tensors = []
    for labels in (y_reference, *y_target_sets):
        mapped_tensors.append(
            torch.tensor([label_map[int(label)] for label in labels.tolist()], dtype=torch.long)
        )

    return tuple(mapped_tensors)

def make_stratified_split(X, y, val_fraction=0.1):
    train_indices = []
    val_indices = []

    for label in torch.unique(y, sorted=True):
        label_indices = torch.nonzero(y == label, as_tuple=False).squeeze(1)
        label_indices = label_indices[torch.randperm(len(label_indices))]

        if len(label_indices) == 1:
            train_indices.append(label_indices)
            continue

        num_val = max(1, int(round(len(label_indices) * val_fraction)))
        num_val = min(num_val, len(label_indices) - 1)
        val_indices.append(label_indices[:num_val])
        train_indices.append(label_indices[num_val:])

    train_indices = torch.cat(train_indices)
    val_indices = torch.cat(val_indices)
    train_indices = train_indices[torch.randperm(len(train_indices))]
    val_indices = val_indices[torch.randperm(len(val_indices))]

    return X[train_indices], y[train_indices], X[val_indices], y[val_indices]

def mlp_train(X_train, y_train, X_val, y_val, X_test, y_test):

    model = MLP(input_dim=X_train.shape[1], hidden_dim=512, num_classes=len(torch.unique(y_train)))
    model.to(X_train.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 400
    batch_size = 64
    patience = 100
    min_delta = 1e-5
    num_batches = (len(X_train) + batch_size - 1) // batch_size
    best_val_loss = float("inf")
    best_state = {name: param.detach().clone() for name, param in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in tqdm(range(num_epochs), desc="Training MLP", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        perm = torch.randperm(len(X_train), device=X_train.device)
        X_train_epoch = X_train[perm]
        y_train_epoch = y_train[perm]
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(X_train))
            inputs = X_train_epoch[start_idx:end_idx]
            labels = y_train_epoch[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val).item()

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {name: param.detach().clone() for name, param in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best validation loss: {best_val_loss:.4f})"
                )
                break

    model.load_state_dict(best_state)

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
    
    # only use cls tokens for the prototype
    X_train_raw, y_train_raw = train_data["cls_tokens"], train_data["labels"]
    X_val_raw, y_val_raw = val_data["cls_tokens"], val_data["labels"]
    X_test, y_test = test_data["cls_tokens"], test_data["labels"]

    full_train_data = torch.cat([X_train_raw, X_val_raw])
    full_train_labels = torch.cat([y_train_raw, y_val_raw])
    full_train_labels, y_test = remap_labels_from_reference(full_train_labels, y_test)
    X_train, y_train, X_val, y_val = make_stratified_split(full_train_data, full_train_labels)

    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    mlp_train(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    main()
