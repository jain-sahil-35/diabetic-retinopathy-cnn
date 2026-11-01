import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from torch.utils.data import DataLoader, Subset

from dataset import RetinopathyDataset
from model import BaselineCNN


# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# --------------------------------------------------
# 2. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset" / "B. Disease Grading"

TRAIN_IMG_DIR = DATA_DIR / "1. Original Images" / "a. Training Set"

TRAIN_CSV = (
    DATA_DIR
    / "2. Groundtruths"
    / "a. IDRiD_Disease Grading_Training Labels.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. Training settings
# --------------------------------------------------

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001


# --------------------------------------------------
# 4. Device selection
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 5. Load dataset and split it
# --------------------------------------------------

full_dataset = RetinopathyDataset(TRAIN_CSV, TRAIN_IMG_DIR)

all_labels = full_dataset.data["Retinopathy grade"].to_numpy()

train_indices, val_indices = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,
    random_state=SEED,
    stratify=all_labels,
)

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)


# --------------------------------------------------
# 6. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# 7. Initialize model, loss and optimizer
# --------------------------------------------------

model = BaselineCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

best_val_macro_f1 = 0.0


# --------------------------------------------------
# 8. Training loop
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    # Training batches
    for images, labels in train_loader:

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * labels.size(0)

        train_correct += (
            (outputs.argmax(dim=1) == labels).sum().item()
        )

        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total


    # --------------------------------------------------
    # 9. Validation
    # --------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * labels.size(0)

            val_correct += (
                (outputs.argmax(dim=1) == labels).sum().item()
            )

            val_total += labels.size(0)

            all_predictions.extend(
                outputs.argmax(dim=1).cpu().tolist()
            )

            all_targets.extend(
                labels.cpu().tolist()
            )

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    val_macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    val_confusion_matrix = confusion_matrix(
        all_targets,
        all_predictions,
        labels=[0, 1, 2, 3, 4],
    )


    # --------------------------------------------------
    # 10. Print epoch results
    # --------------------------------------------------

    print(
        f"\nEpoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(val_confusion_matrix)

    print("\nClassification Report:")

    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=[0, 1, 2, 3, 4],
            target_names=[
                "No DR",
                "Mild",
                "Moderate",
                "Severe",
                "Proliferative DR",
            ],
            zero_division=0,
        )
    )


    # --------------------------------------------------
    # 11. Save best model checkpoint
    # --------------------------------------------------

    if val_macro_f1 > best_val_macro_f1:

        best_val_macro_f1 = val_macro_f1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "seed": SEED,
            },
            RESULTS_DIR / "baseline_cnn_best.pth",
        )


# --------------------------------------------------
# 12. Training summary
# --------------------------------------------------

print("\nTraining completed!")

print(f"Best validation Macro F1: {best_val_macro_f1:.4f}")

print(
    "Best checkpoint:",
    RESULTS_DIR / "baseline_cnn_best.pth",
)

