"""
evaluation.py - evaluation for the BSD10k AST classifier. Can (and should) be run independently,
                retraining isn't done every time.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from model import AudioClassifier
from dataset_class import MyDataset
from data_handling import read_data_from_pkl

# Config
NUM_CLASSES = 23
BATCH_SIZE = 10
MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 0.5 # = partial credit for same top-level, different second-level


# BST label mapping
# derived from the provided data split .csv files
CLASS_NAMES = [
    "Solo percussion", # 0, m-sp
    "Solo instrument", # 1, m-si
    "Multiple instruments", # 2 m-m
    "Percussion", # 3 is-p
    "String", # 4 is-s
    "Wind", # 5 is-w
    "Piano / Keyboard", # 6 is-k
    "Synths / Electronic", # 7 is-e
    "Solo speech", # 8 sp-s
    "Conversation / Crowd", # 9 sp-c
    "Processed / Synthetic", # 10 sp-p
    "Objects / House appliances", # 11 fx-o
    "Vehicles", # 12 fx-v
    "Other mechanisms / machines", # 13 fx-m
    "Human sounds and actions", # 14 fx-h
    "Animals", # 15 fx-a
    "Natural elements & explosions", # 16 fx-n
    "Experimental", # 17 fx-ex
    "Electronic / Design", # 18 fx-el
    "Nature", # 19 ss-n
    "Indoors", # 20 ss-i
    "Urban", # 21 ss-u
    "Synthetic / Artificial" # 22 ss-s
]

# Short labels for CM
CLASS_KEYS = [
    "m-sp", "m-si", "m-m",
    "is-p", "is-s", "is-w", "is-k", "is-e",
    "sp-s", "sp-c", "sp-p",
    "fx-o", "fx-v", "fx-m", "fx-h", "fx-a", "fx-n", "fx-ex", "fx-el",
    "ss-n", "ss-i", "ss-u", "ss-s",
]

PARENT = {
    0: "Music", 1: "Music", 2: "Music",
    3: "Instrument samples", 4: "Instrument samples", 5: "Instrument samples",
    6: "Instrument samples", 7: "Instrument samples",
    8: "Speech", 9: "Speech", 10: "Speech",
    11: "Sound effects", 12: "Sound effects", 13: "Sound effects",
    14: "Sound effects", 15: "Sound effects", 16: "Sound effects",
    17: "Sound effects", 18: "Sound effects",
    19: "Soundscapes", 20: "Soundscapes", 21: "Soundscapes",
    22: "Soundscapes",
}

GROUP_COLOURS = {
    "Music": "#e05c5c",
    "Instrument samples": "#f0a830",
    "Speech": "#4db8b8",
    "Sound effects": "#5b9bd5",
    "Soundscapes": "#5cb85c",
}

# Testing data
def load_test_data() -> DataLoader:
    """Loads the test data only. Returns the DataLoader."""
    _, _, _, test_data = read_data_from_pkl()
    dataset = MyDataset(
        test_data["features"], test_data["labels"], use_ast_extractor=False
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Inference
@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader):
    """Runs inference on a model and returns the predictions."""
    model.eval()
    all_preds, all_labels = [], []
    for features, labels in loader:
        logits = model(features.to(DEVICE))
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

# Standard metrics
def overall_accuracy(preds, labels) -> float:
    """Computation of overall accuracy."""
    return float((preds == labels). mean())

def per_class_accuracy(preds, labels) -> np.ndarray:
    """Computation of per class accuracy."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    return cm.diagonal() / row_sums.squeeze()

# Hierarchical metrics
def _weight(i: int, j: int) -> float:
    """
    Weight w_ij for the hierarchical metric formula:
        w_ij = 1, if i == j
        w_ij = lambda (0.5), if i != j but same top-level parents (partial credit)
        w_ij = 0, otherwise
    """
    if i == j:
        return 1.0
    if PARENT[i] == PARENT[j]:
        return LAMBDA
    return 0.0

def hierarchical_metrics(preds, labels):
    """Computation of hierarchical matrics from project slides."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))

    hP_list, hR_list, hF_list = [], [], []

    for i in range(NUM_CLASSES):
        num_P = 0.0 # numerator sum_j(w_ij * TP_ij)
        den_P = 0.0 # denominator sum_j(w_ij * (TP_ij + FP_ij))
        den_R = 0.0 # denominator sum_j(w_ij * (TP_ij + FN_ij))

        for j in range(NUM_CLASSES):
            w = _weight(i, j)
            if w == 0.0:
                continue

            if i == j:
                tp_ij = cm[i, i]
                fp_ij = cm[:, j].sum() - cm[i, j] # others predicted as i
                fn_ij = cm[i, :].sum() - cm[i, i] # truly i but predicted elsewhere
            else:
                # Same parent, off-diagonal: no TP but partial FP/FN
                tp_ij = 0.0
                fp_ij = cm[:, j].sum() - cm[i, j] # predicted j, not truly i
                fn_ij = cm[i, j] # truly i, predicted as j

            num_P += w * tp_ij
            den_P += w * (tp_ij + fp_ij)
            den_R += w * (tp_ij + fn_ij)
        hp_i = num_P / den_P if den_P > 0 else 0.0
        hr_i = num_P / den_R if den_R > 0 else 0.0
        hf_i = (2 * hp_i * hr_i / (hp_i + hr_i)) if (hp_i + hr_i) > 0 else 0.0

        hP_list.append(hp_i)
        hR_list.append(hr_i)
        hF_list.append(hf_i)

    return (
        float(np.mean(hP_list)),
        float(np.mean(hR_list)),
        float(np.mean(hF_list)),
    )

# Plots
def plot_confusion_matrix(preds, labels):
    """Computes and plots the confusion matrix."""
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    thresh = 0.5
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}",
                        ha="center", va="center", fontsize=6,
                        color="white" if val>thresh else "black")

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_KEYS, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(CLASS_KEYS, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalized confusion matrix (rows = true class)", fontsize=13)

    plt.tight_layout()
    plt.show()

def plot_per_class_accuracy(pca):
    """Plots a histogram of the per-class accuracy. Colour-coded to match the parent classes."""
    idx = np.argsort(pca)
    accs = pca[idx]
    names = [f"{CLASS_KEYS[i]}" for i in idx]
    bar_colours = [GROUP_COLOURS[PARENT[i]] for i in idx]
    mean = pca.mean()

    x_pos = np.arange(len(names))

    fig, ax = plt.subplots(figsize=(11,9))
    bars = ax.bar(x_pos, accs, color=bar_colours, edgecolor="white")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title(
        "Per-Class Accuracy (sorted ascending)", fontsize=12,
    )
    ax.set_ylim(0, 1.12)
    ax.axhline(mean, color="crimson", linestyle="--", linewidth=1.5)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.show()

# Print metrics
def print_metrics(oa, hp, hr, hf) -> str:
    """Prints the most important metrics."""
    print(
        f"Overall accuracy: {oa:.4f} ({oa*100:.2f}%)\n"
        f"Hierarchical metrics (lambda = {LAMBDA})\n"
        f"  hPrecision: {hp:.4f} ({hp*100:.2f}%)\n"
        f"  hRecall: {hr:.4f} ({hr*100:.2f}%)\n"
        f"  hF-Score: {hf:.4f} ({hf*100:.2f}%)\n"
    )

# Main
def main():
    print("-------- TESTING --------")
    print(f"Device: {DEVICE}")
    print(f"Loading model from '{MODEL_PATH}'...")
    model = AudioClassifier(num_classes=NUM_CLASSES, freeze_ast=True).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    print("Loading test data...")
    test_loader = load_test_data()
    print(f"Test batches: {len(test_loader)}")

    print("Running inference...")
    preds, labels = get_predictions(model, test_loader)

    oa = overall_accuracy(preds, labels)
    pca = per_class_accuracy(preds, labels)
    hp, hr, hf = hierarchical_metrics(preds, labels)

    plot_confusion_matrix(preds, labels)
    plot_per_class_accuracy(pca)
    print("-------- TEST RESULTS --------")
    print_metrics(oa, hp, hr, hf)

if __name__ == "__main__":
    main()
