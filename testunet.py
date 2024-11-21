# testunet.py
import torch
from utils import DiceLoss,load_config

# Load configuration
config = load_config('unet_1.yaml')
print("Loaded Config:", config)

# Initialize DiceLoss
dice_loss = DiceLoss()

# Sample input and target tensors for testing DiceLoss
inputs = torch.tensor([1.1, 1.0, 1.0, 0.0])  # Simulated prediction
targets = torch.tensor([1.1, 1.0, 0.0, 1.0])  # Simulated ground truth

# Calculate Dice Loss
loss = dice_loss(inputs, targets)
print("Dice Loss:", loss.item())
