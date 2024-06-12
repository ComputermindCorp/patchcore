from __future__ import annotations

import torch
import torch.nn.functional as F
from sklearn.random_projection import SparseRandomProjection
import random
import tqdm

def k_center_greedy(
        x: torch.Tensor,
        sampling_ratio: float,
        seed: int | None=None,
        progress: bool=False,
    ) -> tuple[torch.Tensor, int]:
    """K-Center Greedyアルゴリズム

    Args:
        x (torch.Tensor): 行列（[データ数, 特徴ベクトル数]の2次元）.
        sampling_ratio (float): コアセットサンプリング比率.
        seed (int | None, optional): 乱数シード. Defaults to None.
        progress (bool, optional): 進捗表示の有無. Defaults to False.

    Returns:
        tuple[torch.Tensor, int]: サンプリングした行列とサンプリング後の要素数のタプル
    """
    device = x.device
    
    # 比率からサンプリングするデータ数を決定 
    n_sample = int(len(x) * sampling_ratio)
    
    # 乱数シード設定
    random.seed(seed)

    # 処理高速化のためにスパースランダム投影を使って特徴ベクトルの次元を削減
    random_projection = SparseRandomProjection(n_components='auto', eps=0.9)     
    random_projection.fit(x.detach().cpu())
    x_dash = x.detach().cpu()
    x_dash = random_projection.transform(x_dash)
    x_dash = torch.tensor(x_dash).to(device)

    # 初期中心をランダムで選択
    center_index = random.randint(0, len(x) - 1)
    # サンプリングした要素のインデックスリストを初期化
    selected_indexes = [center_index]

    # 進捗表示設定
    if progress:
        itr = tqdm.tqdm(range(n_sample))
    else:
        itr = range(n_sample)

    # 予定サンプリング数だけ実行
    min_distance = None
    for _ in itr:
        # 中心と各データのユークリッド距離を計算
        distance = F.pairwise_distance(x_dash, x_dash[center_index])

        # 新たに算出した距離とこれまで算出した距離を、各データごとに比較して最小のものを選択
        if min_distance is None:
            min_distance = distance
        else:
            min_distance = torch.minimum(min_distance, distance)

        # 中心から一番遠いデータを次の中心とする
        center_index = int(torch.argmax(min_distance).item())
        selected_indexes.append(center_index)
        # 一度中心としたデータの距離は以後使わない
        min_distance[selected_indexes] = 0

    return x[selected_indexes], n_sample
