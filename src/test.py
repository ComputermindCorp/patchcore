from __future__ import annotations

import argparse
from pathlib import Path
import csv
import omegaconf
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np
import cv2
from sklearn import metrics
import copy

from models.patch_core import PatchCore
from common.pytorch_custom_dataset import ImagePaths
from common.benchmark import Benchmark
from models.patch_core import visualize

def write_csv(
        save_path: str,
        csv_results: list,
        cfg: omegaconf.dictconfig.DictConfig,
        net: PatchCore,
        th: float,
        precision: list[float],
        recall: list[float],
        f1_score: list[float],
        macro_precision: float,
        macro_recall: float,
        macro_f1_score: float,
        cm: list,
        encoding: str='shift_jis',
    ):
    """csvファイルを作成・保存

    Args:
        save_path (str): 保存先パス
        csv_results (list): csvに書き込むデータ
        cfg (omegaconf.dictconfig.DictConfig): 設定ファイルオブジェクト
        net (PatchCore): PatchCoreオブジェクト
        th (float): しきい値
        precision (list[float]): Precision(正常・異常各クラス)
        recall (list[float]): Recall(正常・異常各クラス)
        f1_score (list[float]): F1-score(正常・異常各クラス)
        macro_precision (float): Precision（マクロ平均）
        macro_recall (float): Recall（マクロ平均）
        macro_f1_score (float): F1-score（マクロ平均）
        cm (list): コンフュージョンマトリックス
        encoding (str, optional): csvファイルのテキストエンコーディング. Defaults to 'shift_jis'.
    """

    with open(save_path, 'w', encoding=encoding) as f:
        writer = csv.writer(f, lineterminator='\n')
        
        # モデル情報
        data = []
        data.append(["patch core"])
        data.append(["weights file", cfg.weights_path])
        data.append(["device", net.device])
        data.append(["backborn", net.backborn_id])
        data.append(["coreset_sampling_ratio", net.coreset_sampling_ratio])
        data.append(["num_neighbors", net.num_neighbors])
        layers = ["Layers"]
        layers.extend(net.layers)
        data.append(layers)
        data.append([])

        # テスト条件
        data.append(["threshould", th])
        data.append([])

        # 精度指標
        data.append(["", "", "", "", "", "", "", "", "precision", "recall", "f1_score"])
        data.append(["", "", "", "", "", "", 0, "Normal", precision[0], recall[0], f1_score[0]])
        data.append(["", "", "", "", "", "", 1, "Abnormal", precision[1], recall[1], f1_score[1]])
        data.append(["", "", "", "", "", "", "", "macro-mean", macro_precision, macro_recall, macro_f1_score])
        data.append([])

        data.append(["", "", "", "", "", "", "confusion matrix"])
        data.append(["", "", "", "", "", "", "", "Predict"])
        data.append(["", "", "", "", "", "", "", "Normal", "Abnormal"])
        data.append(["", "", "", "", "", "Label", "Normal", cm[0][0], cm[0][1]])
        data.append(["", "", "", "", "", "", "Abnormal", cm[1][0], cm[1][1]])        
        data.append([])

        # 処理速度
        data.append(["benchmark [ms]"])
        names = [
            "get features",
            "average pooling",
            "concat features",
            "calcu distance",
            "create anomaly map",
            "normalize anomaly map",
            "total",
        ]
        for name, pred_bench in zip(names, net.bench.values()):
            data.append(["", name, pred_bench.newest_mean_time * 1000])

        data.append([])

        data.append(["total time [ms]"])
        total_time = net.bench["[predict] total"].newest_mean_time
        data.append(["", "total time", total_time * 1000])
        data.append([])

        # 書き込み
        writer.writerows(data)

        # 画像ごとの結果（ヘッダー）
        header = ["No", "image"]

        header.append("Label")
        header.append("Predict")
        header.append("Result")

        # 書き込み
        writer.writerow(header)

        # 画像ごとの結果
        writer.writerows(csv_results)

def test(
        cfg: omegaconf.dictconfig.DictConfig,
        visible_bench: bool=False,
    ):
    """テストメイン処理

    Args:
        cfg (omegaconf.dictconfig.DictConfig): 設定情報
        enable_bench (bool, optional): 処理速度ベンチマークの表示有無. Defaults to False.
    """
    # 結果出力パス設定
    if cfg.output_root_path is None or cfg.output_root_path == "":
        output_root_path = None
    else:
        output_root_path = Path(cfg.output_root_path)
        output_root_path.mkdir(exist_ok=True, parents=True)

    # PatchCoreモデル
    net = PatchCore.load_weights(cfg.weights_path, cfg.device)

    # ベンチマーク設定
    net._enable_bench()

    if visible_bench:
        net._show_bench()

    # データローダー
    # dataset
    test_dataset = ImagePaths.create_from_root_paths(
        cfg.test_data_paths,
        label_list=cfg.labels,
        transform = net.get_transform(),
        resize=net.get_resize(),
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 初期設定
    results = []
    corrects = np.zeros(len(test_loader), np.int32)
    preds = np.zeros(len(test_loader), np.int32)

    all_preds = []
    all_labes = []

    csv_results = []

    # 結果ファイルパス設定
    if output_root_path is None:
        output_heatmap_root_path = None
        csv_save_path = None
    else:
        # 結果CSVファイル
        csv_save_path = output_root_path / f"{Path(cfg.weights_path).stem}.csv"

        # 結果画像ファイル
        output_heatmap_root_path = output_root_path / f"heatmap_{Path(cfg.weights_path).stem}"
        output_heatmap_root_path.mkdir(parents=True, exist_ok=True)

        if cfg.heatmap.ng_dir:
            ng_output_path = output_heatmap_root_path / "NG"
            ng_output_path.mkdir(exist_ok=True, parents=True)

    # しきい値
    th = 0.5 if cfg.th is None else cfg.th

    # テスト
    for i, (x, label, paths) in enumerate(test_loader):
        anomaly_score, anomaly_map_org, pred = net.predict(x, th=th)
        anomaly_map = copy.deepcopy(anomaly_map_org)

        label = label.tolist()

        all_preds.append(pred)
        all_labes.extend(label)

        # 結果出力
        if output_root_path is not None:
            # csv
            csv_result = [i+1, Path(paths[0]).name]
            csv_result.extend(label)
            csv_result.append(pred)

            result = (pred == label[0])
            csv_result.append("o" if result else "x")

            csv_results.append(csv_result)

            # heatmap画像生成・保存
            if output_heatmap_root_path is not None:
                heatmap_save_path = output_heatmap_root_path / f"{Path(paths[0]).stem}_heatmap.png"
                heatmap_add_save_path = output_heatmap_root_path / f"{Path(paths[0]).stem}.png"
                heatmap_color_bar = output_heatmap_root_path / "color_bar.png"

                im_org = cv2.imread(paths[0])

                im_heatmap = visualize.create_heatmap_image(anomaly_map, org_size=im_org.shape)

                im_add = visualize.add_image(im_heatmap, im_org, alpha=0.5)

                cv2.imwrite(str(heatmap_save_path), im_heatmap)
                cv2.imwrite(str(heatmap_add_save_path), im_add)
                visualize.create_color_bar_image(save_path=str(heatmap_color_bar))

                if cfg.heatmap.ng_dir and result == False:                    
                    im_ng_heatmap = visualize.create_heatmap_image(anomaly_map, org_size=im_org.shape)
                    im_ng_add = visualize.add_image(im_heatmap, im_org, alpha=0.5)
                    cv2.imwrite(str(ng_output_path / heatmap_save_path.name), im_ng_heatmap)
                    cv2.imwrite(str(ng_output_path / heatmap_add_save_path.name), im_ng_add)

    # 精度指標
    precision = metrics.precision_score(all_labes, all_preds, average=None)
    recall = metrics.recall_score(all_labes, all_preds, average=None)
    f1_score = metrics.f1_score(all_labes, all_preds, average=None)

    macro_precision = metrics.precision_score(all_labes, all_preds, average='macro')
    macro_recall = metrics.recall_score(all_labes, all_preds, average='macro')
    macro_f1_score = metrics.f1_score(all_labes, all_preds, average='macro')

    cm = metrics.confusion_matrix(all_labes, all_preds)

    # 結果表示
    print(f"model: {cfg.weights_path}")
    print(f"\t\tprecision\trecall\t\tf1_score")
    print(f"0:Normal\t{precision[0]:.4f}\t\t{recall[0]:.4f}\t\t{f1_score[0]:.4f}")
    print(f"1:Abnormal\t{precision[1]:.4f}\t\t{recall[1]:.4f}\t\t{f1_score[1]:.4f}")
    print(f"macro-mean\t{macro_precision:.4f}\t\t{macro_recall:.4f}\t\t{macro_f1_score:.4f}")
    print()

    print("Confusion matrix")
    print(cm)

    # CSVファイル書き込み
    if csv_save_path is not None:
        write_csv(
            csv_save_path,
            csv_results,
            cfg,
            net,
            th,
            precision,
            recall,
            f1_score,
            macro_precision,
            macro_recall,
            macro_f1_score,
            cm,
        )
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='config path')
    parser.add_argument('--show-bench', action='store_true', help='enabel benchmark')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    test(cfg, args.show_bench)
