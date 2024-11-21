import torch
from unet import UNet

model = UNet()
input_tensor = torch.randn(1, 1, 256, 256)  # Batch size 1, 1 channel, 256x256 image
output_tensor = model(input_tensor)
print(f"Output shape: {output_tensor.shape}")
