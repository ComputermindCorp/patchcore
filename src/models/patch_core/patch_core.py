from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchmetrics import PrecisionRecallCurve, AUROC
from pathlib import Path
import numpy as np

from .backborn import backborn_list
#from ._sampler import KCenterGreedy
from . import sampler
from .anomaly_map import compute_anomaly_map
from common.benchmark import Benchmark

class PatchCore:
    """PatchCoreクラス
    """
    def __init__(self,
                device: str | None,
                input_size: tuple[int, int],
                backborn_id: str="wide_resnet50",
                coreset_sampling_ratio: float=0.1,
                num_neighbors: int=9,
                ):
        """イニシャライザ

        Args:
            device (str | None): デバイスID(Noneの場合は自動設定)
            input_size (tuple[int, int]): 入力サイズ
            backborn_id (str, optional): 特徴量抽出モデルID. Defaults to "wide_resnet50".
            coreset_sampling_ratio (float, optional): コアセットサンプリング比率. Defaults to 0.1.
            num_neighbors (int, optional): 近傍法の抽出数. Defaults to 9.
        """
        self._initialize(device, input_size, backborn_id, coreset_sampling_ratio, num_neighbors)

        self.bench = Benchmark.create_timers([
            "[predict] get_features",
            "[predict] average_pooling",
            "[predict] resize",
            "[predict] nearest_neighbors",
            "[predict] anomaly_map_generator",
            "[predict] normalization",
            "[predict] total",
        ], enable=False, visible=False, except_first=True)

    def _initialize(self,
                device: str | None,
                input_size: tuple[int, int],
                backborn_id: str="wide_resnet50",
                coreset_sampling_ratio: float=0.1,
                num_neighbors: int=9,
                ):
        """イニシャライザ

        Args:
            device (str | None): デバイスID(Noneの場合は自動設定)
            input_size (tuple[int, int]): 入力サイズ
            backborn_id (str, optional): 特徴量抽出モデルID. Defaults to "wide_resnet50".
            coreset_sampling_ratio (float, optional): コアセットサンプリング比率. Defaults to 0.1.
            num_neighbors (int, optional): 近傍法の抽出数. Defaults to 9.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.device = device
        self.backborn_id = backborn_id
        self.input_size = input_size
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors

        self.thresould = None
        self.min_value = None
        self.max_value = None

        if backborn_id != "":
            self._set_backborn(backborn_id)

    def _set_backborn(self, backborn_id: str):
        """特徴量抽出モデルを設定

        Args:
            backborn_id (str): 特徴量抽出モデルID

        """
        if backborn_id not in backborn_list.keys():
            raise ValueError(f'Invalid feature network ID "{backborn_id}".')
        self.backborn = backborn_list[backborn_id](self.device)

    @property
    def layers(self) -> list[str]:
        """特徴ベクトル抽出レイヤー名のリストを取得する

        Returns:
            list[str]: 特徴ベクトル抽出レイヤー名のリスト
        """
        return self.backborn.layers
    
    @property
    def patch_size(self) -> int:
        """パッチサイズを取得

        Returns:
            int: パッチサイズ
        """
        return self.backborn.patch_size
    
    def _get_feature_size(self) -> dict:
        """特徴量サイズを取得（内部使用用）

        Returns:
            dict: 特徴量サイズ情報
        """
        x = torch.zeros((1, 3, self.input_size[0], self.input_size[1])).to(self.device)
        features = self.backborn.get_features(x)
        feature_size = { layer: self.backborn.get_features(x)[layer].shape for layer in self.layers}
        return feature_size
        
    def get_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """異常マップ、異常スコアを算出

        Args:
            x (torch.Tensor): 入力ベクトル

        Returns:
            tuple[torch.Tensor, torch.Tensor]: 異常マップと異常スコアのタプル
        """
        # 特徴ベクトルを抽出
        self.bench["[predict] get_features"].start()
        features = self.backborn.get_features(x)
        self.bench["[predict] get_features"].show()

        # Average Pooling
        self.bench["[predict] average_pooling"].start()
        features = { layer: torch.nn.AvgPool2d(3, 1, 1)(feature) for layer, feature in features.items() }
        self.bench["[predict] average_pooling"].show()

        #for k, v in features.items():
        #    print(f"features - > {k}: {v.shape}")
        
        # 複数レイヤーの特徴ベクトルを統合しリサイズする
        self.bench["[predict] resize"].start()
        embedding = self._generate_embedding(features)
        feature_map_shape = embedding.shape[-2:]
        embedding = self._reshape_embedding(embedding)

        self.bench["[predict] resize"].show()

        # 近傍法でスコアを算出
        self.bench["[predict] nearest_neighbors"].start()
        patch_scores = self._nearest_neighbors(embedding=embedding, n_neighbors=self.num_neighbors)
        self.bench["[predict] nearest_neighbors"].show()

        self.bench["[predict] anomaly_map_generator"].start()
        anomaly_map, anomaly_score = compute_anomaly_map(
            patch_scores,
            feature_map_shape,
            self.input_size,
        )
        self.bench["[predict] anomaly_map_generator"].show()

        return anomaly_map, anomaly_score
    
    def _nearest_neighbors(self, embedding: torch.Tensor, n_neighbors: int=9) -> torch.Tensor:
        """最近傍法により各パッチの異常スコアを算出する

        Args:
            embedding (Tensor): 特徴ベクトル [N, n_features]
            n_neighbors (int): 最近傍のTop N個

        Returns:
            Tensor: 各パッチの異常スコア [N, top-k]

        Note:
            サンプルごとに総当りでユークリッド距離を算出し、サンプルごとにその中の最小n_neighbors個の距離を取得する
        """
        # メモリバンクと入力画像特徴のユークリッド距離を総当りで求める
        distances = torch.cdist(embedding, self.memory_bank, p=2.0)

        # 求めた総当りのユークリッド距離の中からサンプルごとに最小n_neighbors個を取得
        patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)
        return patch_scores
    
    def train_init(self):
        """学習初期化処理
        """
        self.embeddings = []

    def train_step(self, x: torch.Tensor):
        """学習ステップ処理

        Args:
            x (torch.Tensor): 入力ベクトル
        """
        x = x.to(self.device)

        # 特徴ベクトルを抽出する
        embedding = self.get_features(x)
        self.embeddings.append(embedding)

    def train_epoch_end(self):
        """学習エポック終了時処理
        """
        # 特徴量を貪欲法でサブサンプリングする
        self.sub_sampling(self.embeddings)

    def validation_init(self):
        """バリデーション初期化処理
        """
        self.outputs = []
        self.min_value = torch.tensor(float("inf"))
        self.max_value = torch.tensor(float("-inf"))
        self._precision_recall_curve = PrecisionRecallCurve(num_classes=1)
        self._auroc_metrics = AUROC(num_classes=1)

    def validation_step(self, x: torch.Tensor, label: torch.tensor):
        """バリデーションステップ処理

        Args:
            x (torch.Tensor): 入力ベクトル
            label (torch.tensor): ラベル
        """
        x = x.to(self.device)

        # 異常スコアと異常マップを算出
        anomaly_map, anomaly_score = self.get_score(x)

        # 異常スコアとマップの最大・最小値を記録
        self.min_value = torch.min(self.min_value, torch.min(anomaly_map))
        self.max_value = torch.max(self.max_value, torch.max(anomaly_map))

        # PRカーブ
        self._precision_recall_curve(anomaly_score.unsqueeze(0).cpu(), label)
        
        # AUROC
        self._auroc_metrics(anomaly_score.unsqueeze(0).cpu(), label)

    def validation_epoch_end(self) -> tuple[dict[str, float], dict[str, float]]:
        """バリデーションエポック終了時処理

        Returns:
            dict: 精度指標のdictと標準化パラメータのdictのタプル
        
        Notes: 戻り値
            # 精度指標
            {
                "auroc": AUROC
                "precison": Precision
                "recall": Recall
                "f1_score": F1-score
            }

            # 標準化パラメータ
            {
                "threshould": 最良のしきい値（=一番F1-scoreの高いしきい値）
                "min": 異常スコアの最小値
                "max": 異常スコアの最大値
            }
        """
        # 複数のしきい値でF1-scoreを求める
        precision, recall, thresoulds = self._precision_recall_curve.compute()
        f1_score = (2 * precision * recall) / (precision + recall + 1e-10)

        # 一番精度の良い（=F1-scoreの高い）しきい値を求める
        best_index = torch.argmax(f1_score)
        self.thresould = thresoulds[best_index]

        # AUROCを求める
        auroc = self._auroc_metrics.compute()
        
        return ({
            "auroc": auroc.cpu(),
            "precision": precision[best_index].cpu(),
            "recall": recall[best_index].cpu(),
            "f1_score": f1_score[best_index].cpu(),
        },
        {
            "thresould": self.thresould.cpu(),
            "min": self.min_value.cpu(),
            "max": self.max_value.cpu(),
        })
        
    def predict(self, x: torch.Tensor, th=0.5) -> tuple[torch.tensor, torch.Tensor]:
        """推論を行う

        Args:
            x (tuple[torch.tensor, torch.Tensor]): 異常スコアと異常マップのtuple
        """
        self.bench['[predict] total'].start()
        x = x.to(self.device)

        # 異常スコアと異常マップを算出
        anomaly_map, anomaly_score = self.get_score(x)

        # 異常スコアと異常マップを標準化
        if self.thresould is not None and self.min_value is not None and self.max_value is not None:
            self.bench['[predict] normalization'].start()
            anomaly_score = self._normalization(anomaly_score, self.thresould, self.min_value, self.max_value)
            anomaly_map = self._normalization(anomaly_map, self.thresould, self.min_value, self.max_value)
            self.bench['[predict] normalization'].show()

            result = 1 if anomaly_score >= th else 0
        else:
            result = None

        self.bench['[predict] total'].show()

        return anomaly_score, anomaly_map, result

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """特徴ベクトルを抽出する

        Args:
            x (torch.Tensor): 入力ベクトル

        Returns:
            torch.Tensor: 特徴ベクトル
        """
        # 特徴ベクトルを抽出
        features = self.backborn.get_features(x)

        # Average Poolin
        features = { layer: torch.nn.AvgPool2d(3, 1, 1)(feature) for layer, feature in features.items() }
        
        # 複数レイヤーの特徴ベクトルを統合しリサイズする
        embedding = self._generate_embedding(features)
        embedding = self._reshape_embedding(embedding)

        return embedding
    
    def sub_sampling(self, embeddings: torch.Tensor):
        """特徴量のサブサンプリングを行う

        Args:
            embeddings (torch.Tensor): 特徴ベクトル
        """
        # 特徴量を連結
        embeddings = torch.vstack(embeddings)

        # 貪欲法で特徴量をサブサンプリングする
        coreset, n = sampler.k_center_greedy(embeddings, sampling_ratio=self.coreset_sampling_ratio, progress=True)
        print(f"{len(embeddings)} -> {n}")

        # サブサンプリングした特徴量を格納
        self.memory_bank = coreset
    
    def _generate_embedding(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        """複数レイヤーのサイズの異なる特徴ベクトルを統合する

        Args:
            features (dict[str, torch.Tensor]): 特徴ベクトル

        Returns:
            torch.Tensor: 統合した特徴ベクトル

        Note:
            一番浅いレイヤーの特徴ベクトルに後段レイヤーの特徴ベクトルのサイズを合わせる
        """

        # 一番浅いレイヤーの特徴ベクトル
        embeddings = features[self.layers[0]]

        # 後段レイヤーの特徴ベクトルを一番浅いレイヤーの特徴ベクトルサイズに合わせる
        for layer in self.layers[1:]:
            layer_embedding = features[layer]
            layer_embedding = F.interpolate(layer_embedding, size=embeddings.shape[-2:], mode="nearest")
            embeddings = torch.cat((embeddings, layer_embedding), 1)

        return embeddings
    
    @staticmethod
    def _reshape_embedding(embedding: torch.Tensor) -> torch.Tensor:
        """特徴量のリシェイプ

        Args:
            embedding (Tensor): 特徴ベクトル

        Returns:
        
            torch.Tensor: リシェイプした特徴ベクトル.
        
        Note:
            [N, Embedding, Patch, Patch] -> [N*Patch*Patch, Embedding]
        """
        embedding_size = embedding.size(1)
        embedding = embedding.permute(0, 2, 3, 1).reshape(-1, embedding_size)
        return embedding


    def get_transform(self) -> transforms:
        """前処理（transform）オブジェクトを取得する.

        Returns:
            transforms: transformオブジェクト
        """
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        transform = transforms.Compose(transform_list)

        return transform
    
    def get_resize(self) -> transforms:
        """リサイズ処理（transform）オブジェクトを取得する

        Returns:
            transforms: transformオブジェクト
        """
        transform = transforms.Compose([
            transforms.Resize(self.input_size),
        ])

        return transform
    
    def _normalization(self, x: torch.Tensor, thresould: float, min: float, max: float) -> torch.Tensor:
        """標準化

        Args:
            x (torch.Tensor): 入力ベクトル
            thresould (float): しきい値パラメータ
            min (float): 最小値
            max (float): 最大値

        Returns:
            torch.Tensor: 標準化したベクトル（値範囲 0.0〜1.0）

        Notes:
            学習時にValidationデータで一番F1-scoreの高いしきい値と異常スコアの最大最小値を
            標準化に使用するパラメータとして保存しておく。
            それらのパラメータを使い、正常・異常の境界しきい値が0.5にくるように標準化を行う。
        """
        # 標準化
        norm = ((x - thresould) / (max - min)) + 0.5

        # 値範囲 0.0〜1.0でCLIP
        norm = torch.minimum(norm, torch.tensor(1))
        norm = torch.maximum(norm, torch.tensor(0))

        return norm
    
    def save_weights(self, path: str | Path):
        """重みファイルを保存する

        Args:
            path (str | Path): 保存パス
        """
        torch.save({
            "thresould": self.thresould,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "backborn_id": self.backborn_id,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "input_size": self.input_size,
            "memory_bank": self.memory_bank,
            "layers": self.layers,
        },
            path
        )

    @staticmethod
    def load_weights(path: str, device: str) -> PatchCore:
        """重みファイルをロードする

        Args:
            path (str): 重みファイルのパス
        """
        net = PatchCore(device, backborn_id="", input_size=(224, 224)) # backborn_id, input_sizeはダミー

        w = torch.load(path, map_location=torch.device(net.device))

        net._initialize(
            device,
            w['input_size'],
            w['backborn_id'],
            w['coreset_sampling_ratio'],
            w['num_neighbors'],
        )

        net.thresould = w['thresould']
        net.min_value = w['min_value']
        net.max_value = w['max_value']
        net.memory_bank = w['memory_bank']

        return net

    def _show_bench(self):
        """ベンチマーク結果を出力
        """
        for b in self.bench.values():
            b.visible = True

    def _enable_bench(self):
        """ベンチマーク結果を有効化
        """
        for b in self.bench.values():
            b.enable = True
