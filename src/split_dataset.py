from torch.utils.data import Subset, DataLoader
from pathlib import Path
from collections import Counter

from sklearn.model_selection import train_test_split

from dataset import RetinopathyDataset


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define dataset paths
DATA_DIR = PROJECT_ROOT / "dataset" / "B. Disease Grading"

TRAIN_IMG_DIR = DATA_DIR / "1. Original Images" / "a. Training Set"

TRAIN_CSV = DATA_DIR / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"


# Create the complete dataset
full_dataset = RetinopathyDataset(TRAIN_CSV, TRAIN_IMG_DIR)

# Extract disease grades for stratification
labels = full_dataset.data["Retinopathy grade"].values

# Split dataset indices into training and validation indices
train_indices, val_indices = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Print dataset sizes
print("Total images:", len(full_dataset))
print("Training images:", len(train_indices))
print("Validation images:", len(val_indices))


# Print class distribution
print("\nTraining set distribution:")
print(Counter(labels[train_indices]))

print("\nValidation set distribution:")
print(Counter(labels[val_indices]))

# Create training and validation subsets
train_subset = Subset(full_dataset, train_indices)
val_subset = Subset(full_dataset, val_indices)

# Define batch size
BATCH_SIZE = 16

# Create DataLoaders
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Inspect one batch
images, labels = next(iter(train_loader))

print("\nBatch information:")
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
print("Labels:", labels)
