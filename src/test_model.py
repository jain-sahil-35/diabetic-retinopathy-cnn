
import torch
from model import BaselineCNN

model = BaselineCNN()
images = torch.randn(2, 3, 512, 512)

outputs = model(images)

print("Input shape:", images.shape)
print("Output shape:", outputs.shape)
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)
