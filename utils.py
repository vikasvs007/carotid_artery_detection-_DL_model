import torch
import torch.nn.functional as F
import yaml
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, prediction, target, smooth=1e-6):
        intersection = torch.sum(prediction * target)
        total = torch.sum(prediction) + torch.sum(target)
        return 1 - (2. * intersection + smooth) / (total + smooth)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

