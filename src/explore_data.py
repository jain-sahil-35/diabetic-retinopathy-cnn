from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use a non-GUI backend for SSH sessions

import matplotlib.pyplot as plt

from PIL import Image

# 1. Define project and Dataset Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "dataset" / "B. Disease Grading"

TRAIN_IMG_DIR = DATA_DIR / "1. Original Images" / "a. Training Set"
TEST_IMG_DIR = DATA_DIR / "1. Original Images" / "b. Testing Set"

GT_DIR = DATA_DIR / "2. Groundtruths"

TRAIN_CSV = GT_DIR / "a. IDRiD_Disease Grading_Training Labels.csv"
TEST_CSV = GT_DIR / "b. IDRiD_Disease Grading_Testing Labels.csv"

# 2. Load the CSV files
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df = train_df.loc[:, ~train_df.columns.str.startswith("Unnamed")]
test_df = test_df.loc[:, ~test_df.columns.str.startswith("Unnamed")]

# 3. Inspect the training dataset
print("\n========== TRAINING DATASET ==========")

print("\nFirst 5 rows:")
print(train_df.head())

print("\nDataset shape:")
print(train_df.shape)

print("\nColumn names:")
print(train_df.columns.tolist())

print("\nData types:")
print(train_df.dtypes)

print("\nMissing values:")
print(train_df.isnull().sum())

# 4. Inspect the testing dataset
print("\n========== TESTING DATASET ==========")

print("\nFirst 5 rows:")
print(test_df.head())

print("\nDataset shape:")
print(test_df.shape)

# 5. Check disease grade distribution
print("\n========== CLASS DISTRIBUTION ==========")

grade_counts = train_df["Retinopathy grade"].value_counts().sort_index()

print(grade_counts)

# 6. Check image counts
train_images = list(TRAIN_IMG_DIR.glob("*.jpg"))
test_images = list(TEST_IMG_DIR.glob("*.jpg"))

print("\n========== IMAGE COUNTS ==========")

print("Training images:", len(train_images))
print("Testing images:", len(test_images))

# 7. Create results directory
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 8. Plot disease grade distribution
plt.figure(figsize=(9, 5))

grade_counts.plot(kind="bar")

plt.title("Distribution of Diabetic Retinopathy Grades")
plt.xlabel("Retinopathy Grade")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(RESULTS_DIR / "class_distribution.png", dpi=300)
plt.close()

# 9. Save a sample retinal image

sample_image_path = train_images[0]
image = Image.open(sample_image_path)

print("\nSample image:", sample_image_path.name)
print("Image dimensions:", image.size)
print("Image mode:", image.mode)

plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title(sample_image_path.name)
plt.axis("off")
plt.tight_layout()

plt.savefig(RESULTS_DIR / "sample_retina.png", dpi=300)
plt.close()

print("\nGraphs saved successfully in:", RESULTS_DIR)
