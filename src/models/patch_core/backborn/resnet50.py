import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

from .base import BackbornBase

class ResNet50(BackbornBase):
    """ResNet50 backborn
    """
    def __init__(self, device):
        super().__init__(device)

        self.model = torchvision.models.resnet50(weights="IMAGENET1K_V1").to(device)        
        self.layers = ['layer2', 'layer3']
        self.patch_size = 28
        # layer2: [1, 512, 28, 28]
        # layer3: [1, 1024, 14, 14]

        self.extractor = create_feature_extractor(self.model, self.layers)
