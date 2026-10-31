from pathlib import Path

from dataset import RetinopathyDataset


# Find the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define dataset paths
DATA_DIR = PROJECT_ROOT / "dataset" / "B. Disease Grading"

TRAIN_IMG_DIR = DATA_DIR / "1. Original Images" / "a. Training Set"

TRAIN_CSV = DATA_DIR / "2. Groundtruths" / "a. IDRiD_Disease Grading_Training Labels.csv"


# Create the Dataset object
train_dataset = RetinopathyDataset(TRAIN_CSV, TRAIN_IMG_DIR)

# Check dataset length
print("Number of training images:", len(train_dataset))

# Retrieve one sample
image, label = train_dataset[0]

# Inspect the sample
print("\nImage information:")
print("Shape:", image.shape)
print("Data type:", image.dtype)
print("Minimum pixel value:", image.min().item())
print("Maximum pixel value:", image.max().item())

print("\nLabel information:")
print("Label:", label.item())
print("Label data type:", label.dtype)
