from preprocessing import preprocess_image, image_to_tensor

import torch
from torch.utils.data import Dataset

import pandas as pd
from pathlib import Path

class RetinopathyDataset(Dataset):

    def __init__(self, csv_file, image_dir):
        self.data = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Retrieve the CSV row at the given index
        row = self.data.iloc[index]

        # Extract the image filename
        image_name = str(row["Image name"]).strip()

        # Add .jpg if the extension is missing
        if Path(image_name).suffix == "":
            image_name += ".jpg"

        # Construct the complete image path
        image_path = self.image_dir / image_name

        # Retrieve the corresponding disease grade
        label = int(row["Retinopathy grade"])

        # Preprocess the image
        image = preprocess_image(image_path)

        # Convert the image into a PyTorch tensor
        image = image_to_tensor(image)

        # Convert the label into a PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label
