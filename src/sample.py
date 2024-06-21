### (1) 前処理
import torch
from PIL import Image
from torchvision import transforms

# 前処理の設定
transform_list = [
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]
transform = transforms.Compose(transform_list)

# 画像読み込み
im = Image.open("data/images/wood/train/OK/IMG_3790_0000.png").convert('RGB')

# 前処理
im = transform(im)

# [N, C, H, W]の形式にする
x = torch.unsqueeze(im, 0)

print(x.shape)
# torch.Size([1, 3, 224, 224])

### (2) 特徴ベクトルの抽出
import torch
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor

device = "cpu"

# WideResNet50のImageNet学習済のモデル
model = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").to(device)

# 中間レイヤー特徴ベクトル抽出のための設定
layers = ['layer2', 'layer3'] # 特徴ベクトルを抽出するレイヤー名
extractor = create_feature_extractor(model, layers)

# 推論モードに設定
extractor.eval()
    
# 指定したレイヤーの特徴ベクトルを抽出する
with torch.no_grad():
    features = extractor(x) # xは前処理済の[N, C, H, W]の画像データテンソル

# featuresには辞書形式でlayer2とlayer3の特徴ベクトルが返される
# {
#   'layer2': layer2の特徴ベクトルtorch.Tensor,
#   'layer3': layer3の特徴ベクトルtorch.Tensor,
# }

print("layer2: ", features['layer2'].shape) # layer2:  torch.Size([1, 512, 28, 28])
print("layer3: ", features['layer3'].shape) # layer3:  torch.Size([1, 1024, 14, 14])

### (3) Average Pooling
import torch

# 抽出した特徴ベクトルに対して中間レイヤーごとにAverage Poolingをかける
pooling = torch.nn.AvgPool2d(3, 1, 1)
features = { layer: pooling(feature) for layer, feature in features.items() }

# Average pooing後のシェイプを表示
for k, v in features.items():
    print(f"average pooling - > {k}: {v.shape}")
# average pooling - > layer2: torch.Size([1, 512, 28, 28])
# average pooling - > layer3: torch.Size([1, 1024, 14, 14])

### (4) 特徴ベクトルの合成
import torch.nn.functional as F

# 深いレイヤーの特徴ベクトルを浅いレイヤーのサイズに合わせるためにUpsampleする
upsample_layer3 = F.interpolate(features['layer3'], size=features['layer2'].shape[-2:], mode="nearest")

# 浅いレイヤーとUpsampleした深いレイヤーをチャンネル方向で結合
features = torch.cat((features['layer2'], upsample_layer3,), 1)

print(features.shape) # torch.Size([1, 1536, 28, 28])

### (5) 特徴ベクトルのリシェイプ
features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
print(features.shape) # torch.Size([784, 1536])

### (6) k-center greedy
from sklearn.random_projection import SparseRandomProjection
import random

def k_center_greedy(x, sampling_ratio, seed=None):
    """K-Center Greedyアルゴリズム

    Args:
        x : 行列（[データ数, 特徴ベクトル次元数]の2次元）.
        sampling_ratio : コアセットサンプリング比率.
        seed: 乱数シード.

    Returns:
        サンプリングした行列とサンプリング後の要素数のタプル
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

    # 予定サンプリング数だけ実行
    min_distance = None
    for _ in range(n_sample - 1):
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

# k-center greedyでパッチ特徴ベクトルを間引き
memory_bank, _ = k_center_greedy(features, sampling_ratio=0.1)

print(f"k_center_greedy: {features.shape} -> {memory_bank.shape}")
# k_center_greedy: torch.Size([784, 1536]) -> torch.Size([78, 1536])

### (7) バリデーションデータの特徴ベクトル抽出
# バリデーションデータのパッチ特徴ベクトルを抽出
# （やっていることは学習時と同じ）

# 画像ファイルパスは任意のものをしてしてください
test_paths = [
    "data/images/wood/test/OK/IMG_3790_0102.png",
    "data/images/wood/test/NG/IMG_3790_0200.png",
]

# ラベル（0:正常 1:異常）
labels = [0, 1]
labels = torch.tensor([labels], dtype=torch.long).T

features = []
for path in test_paths:
    im = Image.open(path).convert('RGB')
    im = transform(im)
    x = torch.unsqueeze(im, 0)

    with torch.no_grad():
        f = extractor(x)

    f = { layer: pooling(feature) for layer, feature in f.items() }

    upsample_layer3 = F.interpolate(f['layer3'], size=f['layer2'].shape[-2:], mode="nearest")
    f = torch.cat((f['layer2'], upsample_layer3,), 1)

    f = f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])
    features.append(f)


### (8) 標準パラメータの算出
from torchmetrics import PrecisionRecallCurve

# 近傍Top-K
n_neighbors = 9

# 入力画像のサイズ
input_image_size = (512, 512)

# 標準化パラメータ算出のための準備
min_value = torch.tensor(float("inf"))
max_value = torch.tensor(float("-inf"))
precision_recall_curve = PrecisionRecallCurve(num_classes=1)

for target, label in zip(features, labels):
    # メモリバンクと入力画像特徴のユークリッド距離を総当りで求める
    distances = torch.cdist(target, memory_bank, p=2.0)

    # 求めた総当りのユークリッド距離の中からサンプルごとに最小n_neighbors個を取得
    patch_scores, _ = distances.topk(k=n_neighbors, largest=False, dim=1)

    print("distance top_k: ", patch_scores.shape) # distance top_k:  torch.Size([784, 9])

    # ==== 異常マップ
    #  パッチごとの最近傍の距離を[N, C, H, W]にリシェイプ
    anomaly_map = patch_scores[:, 0].reshape((1, 1, 28, 28))

    #  入力画像サイズにリサイズ
    anomaly_map = F.interpolate(anomaly_map, size=input_image_size, mode='nearest')

    print("anomaly_map: ", anomaly_map.shape) # anomaly_map:  torch.Size([1, 1, 512, 512])
    
    # ==== 異常スコア
    # 1. 一番距離の遠い要素を取得（近傍法で算出したTop-k個の距離が内包）
    max_scores_index = torch.argmax(patch_scores[:, 0])
    max_scores = torch.index_select(patch_scores, 0, max_scores_index) # [1, Top-N]

    # 2. 最小top-k個の距離から、それらのバラツキを考慮して異常スコアを算出する
    #       weights = 1 - max_scoreの最大値のsoftmax
    weights = 1 - (torch.max(torch.exp(max_scores)) / torch.sum(torch.exp(max_scores)))
    anomaly_score = weights * torch.max(patch_scores[:, 0])

    print("anomaly_score: ", anomaly_score) # anomaly_score:  tensor(1.8167) ※値は入力画像により異なる

    # 標準化パラメータのmin, max値を更新
    min_value = torch.min(min_value, torch.min(anomaly_map))
    max_value = torch.max(max_value, torch.max(anomaly_map))

    # PRカーブを更新
    precision_recall_curve(anomaly_score.unsqueeze(0).cpu(), label)

# 複数のしきい値でF1-scoreを求める
precision, recall, thresoulds = precision_recall_curve.compute()
f1_score = (2 * precision * recall) / (precision + recall + 1e-10)

# 一番精度の良い（=F1-scoreの高い）しきい値を求める
print("f1_score", f1_score)
print("thresoulds", thresoulds)
best_index = torch.argmax(f1_score)
print(best_index)
if thresoulds.dim() == 0:
    thresould = thresoulds
else:
    thresould = thresoulds[best_index]
print(f"thresould: {thresould} min: {min_value} max: {max_value}")
