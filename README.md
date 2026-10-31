# Diabetic Retinopathy Detection Using CNN

A learning-focused deep learning project to classify diabetic
retinopathy severity from retinal fundus images using a Convolutional
Neural Network (CNN) built with PyTorch.

> **Project status:** In development. Dataset exploration and the
> initial image-preprocessing pipeline are complete. The custom PyTorch
> Dataset, DataLoader, and model-training pipeline are the next
> milestones.

## Overview

Diabetic retinopathy (DR) is a diabetes-related eye disease that can
damage the retina. This project aims to build an image-classification
pipeline that predicts DR severity from retinal fundus photographs.

The project uses the **Disease Grading** portion of the **IDRiD (Indian
Diabetic Retinopathy Image Dataset)** and treats the task as a
five-class classification problem.

The goal is not only to build a working model, but also to understand
each part of the deep learning workflow: image preprocessing, data
loading, CNN architecture, training, evaluation, and model improvement.

## Dataset

-   **Dataset:** IDRiD --- Indian Diabetic Retinopathy Image Dataset
-   **Source:** [Official IDRiD
    website](https://idrid.grand-challenge.org/)
-   **Task:** Five-class diabetic retinopathy severity grading

    Label Severity
  ------- ------------------
        0 No DR
        1 Mild DR
        2 Moderate DR
        3 Severe DR
        4 Proliferative DR

### Dataset split

  Split                  Images Usage
  ------------------- --------- ------------------------------------------------
  Training pool             413 Used to create training and validation subsets
  Official test set         103 Reserved for final evaluation
  **Total**             **516** 

The current training-pool class distribution is:

    Grade Severity             Images
  ------- ------------------ --------
        0 No DR                   134
        1 Mild DR                  20
        2 Moderate DR             136
        3 Severe DR                74
        4 Proliferative DR         49

The dataset is imbalanced, especially for Grade 1. We will therefore
report per-class metrics and investigate appropriate imbalance-handling
methods rather than relying on accuracy alone.

**Data-splitting rule:** The official test set must not be used for
training, preprocessing decisions, hyperparameter tuning, or model
selection. A stratified validation subset will be created from the
413-image training pool.

The dataset is not included in this repository. Download it from the
official source and follow the dataset's terms of use.

## Current preprocessing pipeline

The initial preprocessing implementation is in `src/preprocessing.py`.

1.  Open each image with Pillow and convert it to RGB.
2.  Resize it to fit within `512 × 512` while preserving the aspect
    ratio.
3.  Place the resized image in the centre of a black `512 × 512` canvas.
4.  Convert the Pillow image to a NumPy array.
5.  Convert the array to a PyTorch tensor.
6.  Change the tensor layout from **HWC** to **CHW**.
7.  Convert pixel values to `float32` and scale them from `[0, 255]` to
    `[0, 1]`.

A single processed image is expected to have shape:

``` text
(3, 512, 512)  # channels, height, width
```

A batch of 16 images is expected to have shape:

``` text
(16, 3, 512, 512)  # batch, channels, height, width
```

The current pipeline is an initial baseline. Its effect on model
performance will be evaluated during development.

## Technology stack

-   Python 3.10
-   PyTorch 2.4.1
-   Torchvision 0.19.1
-   NumPy and Pandas
-   Pillow
-   Matplotlib and Seaborn
-   Scikit-learn
-   Conda for environment management

Training and development are being performed on a college NVIDIA DGX
server with NVIDIA Tesla V100-SXM2-32GB GPUs.

## Repository structure

``` text
diabetic-retinopathy-cnn/
├── src/
│   ├── explore_data.py
│   ├── preprocessing.py
│   ├── dr_dataset.py       # in progress
│   ├── model.py            # planned
│   ├── train.py            # planned
│   ├── evaluate.py         # planned
│   └── predict.py          # planned
├── results/
│   ├── class_distribution.png
│   ├── sample_retina.png
│   └── preprocessing_comparison.png
├── dataset/                # local only; not committed
├── notebooks/
├── models/                 # trained checkpoints; not committed by default
├── .gitignore
└── README.md
```

Some files and folders in the tree are planned and may not exist in the
repository yet.

## Development roadmap

-   [x] Explore the dataset and inspect labels and missing values
-   [x] Visualize the class distribution and sample retinal images
-   [x] Implement the initial image-resizing and padding pipeline
-   [x] Implement and test Pillow-to-tensor conversion
-   [ ] Finish and test the custom `DRDataset`
-   [ ] Implement and test the PyTorch `DataLoader`
-   [ ] Create a stratified training/validation split
-   [ ] Build and train a simple baseline CNN
-   [ ] Evaluate validation performance and inspect errors
-   [ ] Investigate class imbalance, augmentation, and regularization
-   [ ] Compare the baseline with a pretrained CNN if appropriate
-   [ ] Evaluate the selected model once on the untouched official test
    set
-   [ ] Build a simple image-upload demonstration

This is a step-by-step learning project: understand a component,
implement it, test it, debug it, and commit the milestone to GitHub.

## Setup

The project is currently developed on a remote DGX server using
SSH/PuTTY and a Conda environment named `dr-cnn`.

After connecting to the server, activate the environment:

``` bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dr-cnn
```

Verify the active Python interpreter and version:

``` bash
which python
python --version
```

Move to the project directory:

``` bash
cd ~/projects/diabetic-retinopathy
```

The environment was created for Python 3.10 with PyTorch 2.4.1 and
Torchvision 0.19.1. Package availability on a particular DGX node can
depend on its CUDA setup and the shared server configuration.

### Dataset setup

Place the downloaded IDRiD Disease Grading files in the local dataset
directory, following the actual CSV columns and image-folder layout.
Keep the dataset out of Git; do not commit retinal images or other large
dataset files.

### Running scripts

From the project root, scripts can be run with commands such as:

``` bash
python src/explore_data.py
python src/preprocessing.py
```

Run a script only after confirming that its file exists and any required
paths or CSV column names match your local dataset. Training and
prediction commands will be added once those components are implemented.

## Evaluation plan

Model selection will be based on validation data. Planned metrics
include:

-   Accuracy
-   Per-class precision, recall, and F1-score
-   Macro F1-score
-   Confusion matrix

Because the class distribution is uneven, macro-averaged and per-class
metrics will help reveal weak performance on minority grades. No model
performance results are reported yet.

## Responsible use

This project is an educational and research prototype. It is **not a
medical device** and must not be used to diagnose diabetic retinopathy
or make treatment decisions. Predictions may be incorrect, and clinical
decisions must be made by qualified healthcare professionals.

## Acknowledgements

-   The IDRiD team for providing the Indian Diabetic Retinopathy Image
    Dataset.
-   The PyTorch and scientific Python communities for the tools used in
    this project.
