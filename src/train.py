import csv
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from torch.utils.data import DataLoader, Subset

from dataset import RetinopathyDataset
from model import BaselineCNN


# --------------------------------------------------
# 1. Experiment configuration
# --------------------------------------------------

EXPERIMENT_ID = "BASELINE"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
OPTIMIZER_NAME = "Adam"
WEIGHT_DECAY = 0.0

# Metadata only: preprocessing currently uses 512 x 512.
IMAGE_SIZE = 512


# --------------------------------------------------
# 2. Reproducibility
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Reproducible DataLoader shuffling
loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)


# --------------------------------------------------
# 3. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset" / "B. Disease Grading"

TRAIN_IMG_DIR = (
    DATA_DIR
    / "1. Original Images"
    / "a. Training Set"
)

TRAIN_CSV = (
    DATA_DIR
    / "2. Groundtruths"
    / "a. IDRiD_Disease Grading_Training Labels.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# CSV tracking files
EXPERIMENT_CSV = RESULTS_DIR / "experiment_results.csv"
HISTORY_CSV = RESULTS_DIR / "epoch_history.csv"

# Unique checkpoint for every experiment
CHECKPOINT_PATH = (
    RESULTS_DIR / f"{EXPERIMENT_ID.lower()}_best.pth"
)


# --------------------------------------------------
# 4. CSV helper function
# --------------------------------------------------

def append_to_csv(file_path, fieldnames, row):

    file_exists = file_path.exists()

    with open(
        file_path,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


# --------------------------------------------------
# 5. Device selection
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# 6. Load dataset and split it
# --------------------------------------------------

full_dataset = RetinopathyDataset(
    TRAIN_CSV,
    TRAIN_IMG_DIR,
)

all_labels = full_dataset.data[
    "Retinopathy grade"
].to_numpy()

train_indices, val_indices = train_test_split(
    np.arange(len(full_dataset)),
    test_size=0.2,
    random_state=SEED,
    stratify=all_labels,
)

train_dataset = Subset(
    full_dataset,
    train_indices,
)

val_dataset = Subset(
    full_dataset,
    val_indices,
)

print("Total images:", len(full_dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# 7. Create DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    generator=loader_generator,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# --------------------------------------------------
# 8. Initialize model, loss and optimizer
# --------------------------------------------------

model = BaselineCNN(num_classes=5).to(device)

criterion = nn.CrossEntropyLoss()

if OPTIMIZER_NAME == "Adam":

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

elif OPTIMIZER_NAME == "AdamW":

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

elif OPTIMIZER_NAME == "SGD":

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY,
    )

else:
    raise ValueError(
        f"Unsupported optimizer: {OPTIMIZER_NAME}"
    )


# --------------------------------------------------
# 9. Initialize experiment tracking
# --------------------------------------------------

best_val_macro_f1 = -1.0
best_val_accuracy = 0.0
best_epoch = 0

start_time = time.time()


# --------------------------------------------------
# 10. Training loop
# --------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    # Training batches
    for images, labels in train_loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_sum += (
            loss.item() * labels.size(0)
        )

        train_correct += (
            (outputs.argmax(dim=1) == labels)
            .sum()
            .item()
        )

        train_total += labels.size(0)

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total


    # --------------------------------------------------
    # 11. Validation
    # --------------------------------------------------

    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            labels = labels.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss_sum += (
                loss.item() * labels.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            val_total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
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
    # 12. Save epoch history to CSV
    # --------------------------------------------------

    epoch_row = {
        "experiment_id": EXPERIMENT_ID,
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
    }

    append_to_csv(
        HISTORY_CSV,
        list(epoch_row.keys()),
        epoch_row,
    )


    # --------------------------------------------------
    # 13. Print epoch results
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
    # 14. Save best model checkpoint
    # --------------------------------------------------

    if val_macro_f1 > best_val_macro_f1:

        best_val_macro_f1 = val_macro_f1
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            {
                "experiment_id": EXPERIMENT_ID,
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "seed": SEED,
            },
            CHECKPOINT_PATH,
        )

        print("Best model checkpoint updated!")


# --------------------------------------------------
# 15. Training summary
# --------------------------------------------------

training_time = time.time() - start_time

print("\nTraining completed!")

print(f"Best epoch: {best_epoch}")
print(f"Best validation accuracy: {best_val_accuracy:.4f}")
print(f"Best validation Macro F1: {best_val_macro_f1:.4f}")

print("Best checkpoint:", CHECKPOINT_PATH)


# --------------------------------------------------
# 16. Save experiment summary to CSV
# --------------------------------------------------

experiment_row = {
    "experiment_id": EXPERIMENT_ID,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "optimizer": OPTIMIZER_NAME,
    "weight_decay": WEIGHT_DECAY,
    "image_size": IMAGE_SIZE,
    "epochs": EPOCHS,
    "best_epoch": best_epoch,
    "best_val_accuracy": best_val_accuracy,
    "best_val_macro_f1": best_val_macro_f1,
    "training_time_seconds": round(training_time, 2),
    "seed": SEED,
}

append_to_csv(
    EXPERIMENT_CSV,
    list(experiment_row.keys()),
    experiment_row,
)

print("\nExperiment recorded successfully!")
print("Experiment CSV:", EXPERIMENT_CSV)
print("Epoch history CSV:", HISTORY_CSV)