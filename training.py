import gc
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import AudioClassifier
from dataset_class import MyDataset
from data_handling import read_data_from_pkl


NUM_CLASSES = 23
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
DROPOUT = 0.3
FREEZE_AST = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_data():
    """Load pre-extracted features from pkl files and create DataLoaders."""
    train_data1, train_data2, val_data, test_data = read_data_from_pkl()

    # Merge the two training halves
    train_features = train_data1["features"] + train_data2["features"]
    train_labels = train_data1["labels"] + train_data2["labels"]

    val_features = val_data["features"]
    val_labels = val_data["labels"]

    test_features = test_data["features"]
    test_labels = test_data["labels"]

    # Create datasets (use_ast_extractor=False because features are pre-extracted mel specs)
    train_dataset = MyDataset(train_features, train_labels, use_ast_extractor=False)
    val_dataset = MyDataset(val_features, val_labels, use_ast_extractor=False)
    test_dataset = MyDataset(test_features, test_labels, use_ast_extractor=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)

    # Free raw data references now that datasets hold them -> to save memory
    del train_data1, train_data2, val_data, test_data
    del train_features, train_labels, val_features, val_labels, test_features, test_labels
    gc.collect()

    return train_loader, val_loader, test_loader



def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Run evaluation. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)

        logits = model(features)
        loss = criterion(logits, labels)

        running_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy



def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """Plot training & validation loss and accuracy curves and save to file."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_losses, label="Train Loss")
    ax1.plot(epochs, val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(epochs, train_accs, label="Train Acc")
    ax2.plot(epochs, val_accs, label="Val Acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Saved training curves to training_curves.png")



def main():
    print(f"Using device: {DEVICE}")


    train_loader, val_loader, test_loader = load_data()
    print(f"Train batches: {len(train_loader)}  |  "
          f"Val batches: {len(val_loader)}  |  "
          f"Test batches: {len(test_loader)}")


    model = AudioClassifier(
        num_classes=NUM_CLASSES,
        freeze_ast=FREEZE_AST,
        dropout=DROPOUT,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}")



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )


    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  |  "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}"
        )

        # Save best model & early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pt")
            print(f"  ↳ Saved best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ↳ No improvement ({patience_counter}/{EARLY_STOPPING_PATIENCE})")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

    # ── Quick test evaluation ───────────────────────────────────────────────
    # To run the full evaluation and get the full metrics (per-class accuracy, hierarchical metrics,
    # confusion matrix), run evaluation.py separately. This way it is not dependent on the
    # model retraining since it does inference on an already existing model.
    print("\n── Loading best model for test evaluation ──")
    model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}  |  Test Acc: {test_acc:.4f}")

    # ── Plot curves ───────────────────────────────────────────────────
    plot_metrics(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()
