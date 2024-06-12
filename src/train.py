from __future__ import annotations

import argparse
from pathlib import Path
import tqdm
import omegaconf
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from common.pytorch_custom_dataset import ImagePathes
from models.patch_core import PatchCore
from models.patch_core import test_module

def check_input_pathes(pathes: list[str], data_name: str):
    """ファイルパスリストのチェック

    Args:
        pathes (list[str]): 対象ディレクトリパスのリスト
        data_name (str): 各ディレクトリパスの名称

    Raises:
        ValueError: ファイルパスが1つもない場合にスロー
        FileNotFoundError: ファイルが存在しない場合にスロー
    """
    if len(pathes) < 1:
        raise ValueError(f"no {data_name}.")

    for path in pathes:
        print(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"file or directory not found. {path}")

def train(cfg: omegaconf.dictconfig.DictConfig):
    """学習メイン処理

    Args:
        cfg (omegaconf.dictconfig.DictConfig): 設定情報
    """
    # 入力ファイルチェック
    check_input_pathes(cfg.train.data_pathes, "train data")
    check_input_pathes(cfg.val.data_pathes, "validation data")

    # 重み保存ファイルパスの設定
    if cfg.auto_save_weights_path:
        # 自動命名の場合
        if cfg.save_weights_path_suffix:
            # ファイル名にsuffixを付与する場合
            save_weights_filename = f"{cfg.backborn_id}_size{cfg.input_size[0]}_param_{cfg.coreset_sampling_ratio}_{cfg.num_neighbors}_{cfg.save_weights_path_suffix}.pth"
        else:
            save_weights_filename = f"{cfg.backborn_id}_size{cfg.input_size[0]}_param_{cfg.coreset_sampling_ratio}_{cfg.num_neighbors}.pth"
    else:
        # 指定したファイル名で設定する場合
        save_weights_filename = cfg.save_weights_filename

    save_weights_root_path = Path(cfg.save_weights_root_path)
    save_weights_root_path.mkdir(exist_ok=True, parents=True)
    save_weights_path = save_weights_root_path / save_weights_filename

    # PatchCoreモデル
    net = PatchCore(
        device=cfg.device,
        input_size=cfg.input_size,
        backborn_id=cfg.backborn_id,
        coreset_sampling_ratio=cfg.coreset_sampling_ratio,
        num_neighbors=cfg.num_neighbors,
    )

    # データローダー（Trainデータ）
    train_dataset = ImagePathes.create_from_root_pathes(
        cfg.train.data_pathes,
        label_list=None,
        transform = net.get_transform(),
        resize=net.get_resize(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False
    )

    # データローダー（Valデータ）
    val_dataset = ImagePathes.create_from_root_pathes(
        cfg.val.data_pathes,
        label_list=cfg.val.labels,
        transform = net.get_transform(),
        resize=net.get_resize(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )

    # データローダー（Testデータ）
    print(cfg.test.data_pathes)
    if cfg.test.enable:
        if cfg.test.data_pathes is not None and len(cfg.test.data_pathes) > 0:
            test_dataset = ImagePathes.create_from_root_pathes(
                cfg.test.data_pathes,
                label_list=cfg.test.labels,
                transform = net.get_transform(),
                resize=net.get_resize(),
            )
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        else:
            test_loader = None
    else:
        test_loader = None

    # 学習（メモリバンク作成）
    net.train_init()

    print(f"train data: {len(train_dataset)}")
    for x in tqdm.tqdm(train_loader):
        net.train_step(x)

    # サブサンプリング
    print("sub sampling...")
    net.train_epoch_end()

    # バリデーション（標準化パラメータの決定）
    net.validation_init()

    for x, label, _ in tqdm.tqdm(val_loader):
        net.validation_step(x, label)

    metrics, params = net.validation_epoch_end()

    print(metrics)
    print(params)

    net.save_weights(save_weights_path)

    # テスト
    if test_loader is not None:
        test_module.test(
            test_loader,
            save_weights_path,
            cfg.device,
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', help='config path')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config_path)

    train(cfg)
