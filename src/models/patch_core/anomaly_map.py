from __future__ import annotations

import torch
import torch.nn.functional as F

def compute_anomaly_map(
        patch_scores: torch.Tensor,
        feature_map_shape: torch.Size,
        input_size: tuple[int, int],
        ) -> tuple[torch.Tensor, torch.Tensor]:
    """異常マップと異常スコアを算出する

    Args:
        patch_scores (torch.Tensor): 入力特徴ベクトルとメモリバンクの距離の行列 [n_features, top-k]
        feature_map_shape (torch.Size): 特徴ベクトルのshape
        input_size (tuple[int, int]): 入力画像サイズ

    Returns:
        tuple[torch.Tensor, torch.Tensor]: 異常マップと異常スコアのタプル

    Note:
        patch_scoresは、サンプル数N分の近傍法で求めた最小top-k個の距離を持つ。
    """
    # 異常マップの算出
    anomaly_map = _compute_anomaly_map(patch_scores, feature_map_shape, input_size)

    # 異常スコアの算出
    anomaly_score = _compute_anomaly_score(patch_scores)

    return anomaly_map, anomaly_score

def _compute_anomaly_map(patch_scores: torch.Tensor, feature_map_shape: torch.Size, input_size) -> torch.Tensor:
    """異常マップを算出する

    Args:
        patch_scores (torch.Tensor): 入力特徴ベクトルとメモリバンクの距離の行列 [N, top-k]
        feature_map_shape (torch.Size): 特徴ベクトルのshape
        input_size (_type_): 入力画像サイズ

    Returns:
        torch.Tensor: 異常マップ

    Note:
        patch_scoresは、サンプル数N分の近傍法で求めた最小top-k個の距離を持つ。
    """
    # 特徴ベクトルとメモリバンクの距離をリシェイプ [B, 1, W, H]
    w, h = feature_map_shape
    batch_size = len(patch_scores) // (w * h)
    anomaly_map = patch_scores[:, 0].reshape((batch_size, 1, w, h))

    # 異常マップを入力画像サイズにリサイズ
    anomaly_map = F.interpolate(anomaly_map, size=(input_size[0], input_size[1]), mode='nearest')

    return anomaly_map

def _compute_anomaly_score(patch_scores: torch.Tensor) -> torch.Tensor:
    """異常スコアを算出する

    Args:
        patch_scores (torch.Tensor): 入力特徴ベクトルとメモリバンクの距離の行列 [N, top-k]

    Returns:
        torch.Tensor: 異常スコア

    Note:
        * patch_scoresは、サンプル数N分の近傍法で求めた最小top-k個の距離を持つ。
        * 最小top-k個の距離から、それらのバラツキを考慮して異常スコアを算出する
    """
    # 1. 一番距離の遠い要素を取得（近傍法で算出したTop-k個の距離が内包）
    max_scores_index = torch.argmax(patch_scores[:, 0])
    max_scores = torch.index_select(patch_scores, 0, max_scores_index) # [1, Top-N]

    # 2. 最小top-k個の距離から、それらのバラツキを考慮して異常スコアを算出する
    #       weights = 1 - max_scoreの最大値のsoftmax
    weights = 1 - (torch.max(torch.exp(max_scores)) / torch.sum(torch.exp(max_scores)))
    score = weights * torch.max(patch_scores[:, 0])

    return score
