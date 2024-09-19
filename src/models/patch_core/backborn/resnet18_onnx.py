import torch
import torchvision
import onnxruntime

from .base import BackbornBase

class ResNet18_ONNX(BackbornBase):
    """ResNet18 backborn
    """
    def __init__(self, device):
        super().__init__(device)

        self.model = onnxruntime.InferenceSession('models/patch_core/backborn/model/resnet18_features.onnx')
        self.layers = ['layer2', 'layer3']
        self.patch_size = 28
        # layer2: [1, 128, 28, 28]
        # layer3: [1, 256, 14, 14]

        self.input_name = self.model.get_inputs()[0].name

    def get_features(self, x: torch.Tensor) -> dict:
        """特徴ベクトルを取得する

        Args:
            x (torch.Tensor): 入力ベクトル

        Returns:
            dict: 特徴ベクトル
        """
        device = x.device
        x = x.detach().cpu().numpy()

        outputs = self.model.run(self.layers, { self.input_name: x })
        features = { layer_name: torch.tensor(x).to(device) for layer_name, x in zip(self.layers, outputs) }

        return features
    