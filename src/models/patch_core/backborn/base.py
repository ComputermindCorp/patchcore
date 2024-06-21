from __future__ import annotations

from abc import ABCMeta, abstractmethod
import torch
from torch.fx import GraphModule

class BackbornBase(metaclass=ABCMeta):
    """Backborn基底クラス
    """
    layers: list[str]
    extractor: GraphModule
    patch_size: int

    def __init__(self, device: str) -> None:
        self.device = device

    def get_features(self, x: torch.Tensor) -> dict:
        """特徴ベクトルを取得する

        Args:
            x (torch.Tensor): 入力ベクトル

        Returns:
            dict: 特徴ベクトル
        """
        self.extractor.eval()
    
        with torch.no_grad():
            features = self.extractor(x)

        return features
    