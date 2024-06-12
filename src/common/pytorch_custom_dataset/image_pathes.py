from __future__ import annotations
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
from typing import Union
import glob
import os
import imghdr
from pathlib import Path
import csv

class ImagePathes(Dataset):
    """画像データセット
    """
    def __init__(self,
        image_pathes: list[str],
        labels: Union[list[int], None]=None,
        transform: Union[torchvision.transforms.Compose, None]=None,
        resize: Union[torchvision.transforms.Compose, None]=None,
        ):
        """コンストラクタ

        Args:
            image_pathes (list[str]): 画像パスリスト
            labels (Union[list[int, None], optional): ラベルリスト. Defaults to None.
            transform (Union[torchvision.transforms.Compose, None], optional): 前処理オブジェクト. Defaults to None.
            resize (Union[torchvision.transforms.Compose, None], optional): リサイズ処理オブジェクト. Defaults to None.
        """
        self.image_pathes = image_pathes
        self.labels = labels
        self.transform = transform
        self.resize = resize

    @staticmethod
    def create_from_root_pathes(
        path_list: list[list[str]],
        label_list: list[int] | None=None,
        transform: Union[torchvision.transforms.Compose, None]=None,
        resize: Union[torchvision.transforms.Compose, None]=None,
        ) -> ImagePathes:
        """パスからImagePathesオブジェクトを生成する

        Args:
            path_list (list[str]): ルートパスまたは画像パスファイルのリストのリスト
            label_list (list[int], optional): ラベルのリスト（Noneでラベルを付与しない）. Defaults to None.
            transform (Union[torchvision.transforms.Compose, None], optional): 前処理オブジェクト. Defaults to None.
            resize (Union[torchvision.transforms.Compose, None], optional): リサイズ処理オブジェクト. Defaults to None.

        Returns:
            ImagePathes: ImagePathesオブジェクト

        Notes:
            引数path_listは、画像ファイルあるディレクトリパスとなる
            引数label_listで各ディレクトリパスのクラスを指定する

            ex.
            path_list -> ["/data/class00", "/data/class01_1", "data/class01_2"]
            label_list -> [0, 1, 1]
        """
        all_image_path = []
        all_labels = []
        for i, path in enumerate(path_list):
            path = Path(path)
            if path.is_dir():
                # ディレクトリパスが指定された場合
                image_pathes = ImagePathes._get_image_pathes(path)
                all_image_path += image_pathes

                if label_list:
                    all_labels += [label_list[i]] * len(image_pathes)
            else:
                # ファイルパスファイルのパスが指定された場合
                with open(path) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        print(str(path.parent / row[0]))
                        all_image_path.append(str(path.parent / row[0]))
                        if label_list:
                            all_labels.append(label_list[i])

        if len(all_labels) == 0:
            all_labels = None

        return ImagePathes(all_image_path, all_labels, transform, resize)
            
    @staticmethod
    def _get_image_pathes(path: str, sort=True) -> list[str]:
        """指定パスにある画像ファイルをリスト化する

        Args:
            path (str): 対象パス
            sort (bool): ソートするか否か. Defaults to True.

        Returns:
            list[str]: 画像ファイルパスリスト
        """
        image_pathes = []
        pathes = glob.glob(os.path.join(path, '*.*'))
        for path in pathes:
            if imghdr.what(path) is not None:
                image_pathes.append(path)

        if sort:
            image_pathes = sorted(image_pathes)

        return image_pathes

    def __len__(self) -> int:
        """データセット長を取得

        Returns:
            int: データセット帳
        """
        return len(self.image_pathes)

    def __getitem__(self, i: int) -> torch.Tensor | tuple[torch.Tensor, int, str]:
        """データを取得

        Args:
            i (int): データインデックス

        Returns:
            torch.Tensor | tuple[torch.Tensor, int, str]: 画像行列、または画像行列とラベルとファイルパスのタプル
        """
        # 画像読み込み
        im = Image.open(self.image_pathes[i]).convert('RGB')

        # 前処理
        ## リサイズ
        if self.resize:
            im =self.resize(im)

        ## 前処理
        if self.transform:
            im = self.transform(im)

        if self.labels:
            return im, self.labels[i], self.image_pathes[i]
        else:
            return im
