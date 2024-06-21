from __future__ import annotations

import numpy as np
import cv2
from copy import deepcopy
import torch
import colorsys

def color_map() -> np.array:
    """カラーマップ生成

    Returns:
        np.array: カラーマップ
    """
    b = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.0, 1/6, 103)]
    g = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(1/6, 1/3, 25)]
    r = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.5, 2/3, 128)]

    rgb = np.asarray((np.array(b + g + r) * 255), dtype=np.uint8)

    return rgb

def color_map2() -> np.array:
    """カラーマップ生成(未使用)

    Returns:
        np.array: カラーマップ
    """
    
    rgb = [colorsys.hsv_to_rgb(h, 1.0, 1.0) for h in np.linspace(0.0, 0.6666666666, 256)]

    rgb = np.asarray((np.array(rgb) * 255), dtype=np.uint8)

    return rgb

def create_color_bar_image(color_map_func=color_map, save_path="color_bar.png", w=5, h=80):
    """ヒートマップのカラーバー画像を生成・保存

    Args:
        color_map_func: カラーバー生成関数. Defaults to color_map.
        save_path (str, optional): 画像保存パス. Defaults to "color_bar.png".
        w (int, optional): カラーバーの1カラーの幅（px). Defaults to 5.
        h (int, optional): カラーバーの1カラーの高さ（px). Defaults to 80.
    """
    cm = color_map_func()

    bar = np.zeros((h, 256*w, 3), dtype=np.uint8)
    for i in range(256):
        bar[:, i*w:i*w+w] = cm[i]

    cv2.imwrite(save_path, bar) 

def create_heatmap_image(
        anomaly_map: torch.Tensor,
        org_size: tuple[int, int] | None=None,
    ) -> np.array:
    """ヒートマップ画像生成

    Args:
        anomaly_map (torch.Tensor): 異常マップ
        org_size (tuple[int, int] | None, optional): 入力元画像のサイズ. Defaults to None.

    Returns:
        np.array: ヒートマップ画像
    """
    anomaly_map = anomaly_map.detach().cpu().numpy()

    map = anomaly_map[0][0]

    # 指定サイズにリサイズ
    if org_size is not None:
        map = cv2.resize(map, (org_size[1], org_size[0]))

    map = (map * 255).astype(np.uint8)

    new_map = np.take(color_map(), map, axis=0)

    return new_map

def add_image(
        im_heatmap: np.array,
        im_org: np.array,
        alpha=0.3,
    ) -> np.array:
    """ヒートマップ画像と元画像を合成する

    Args:
        im_heatmap (np.array): ヒートマップ画像
        im_org (np.array): 元画像
        alpha (float, optional): アルファ値. Defaults to 0.3.

    Returns:
        np.array: 合成画像
    """
    im_heatmap = cv2.resize(im_heatmap, (im_org.shape[1], im_org.shape[0]), interpolation=cv2.INTER_CUBIC)
    im_mask = _create_mask(im_heatmap)
    im_org = cv2.bitwise_or(im_org, im_org, mask=im_mask)

    im_add = cv2.addWeighted(src1=im_heatmap, alpha=alpha, src2=im_org, beta=1-alpha, gamma=0)

    return im_add

def _create_mask(im: np.array) -> np.array:
    """マスク画像の生成

    Args:
        im (np.array): 入力画像

    Returns:
        np.array: マスク画像
    """
    im_mask = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_mask = cv2.bitwise_not(im_mask)
    white = np.full(im_mask.shape, 255, dtype=np.uint8)
    im_mask = cv2.bitwise_or(white, white, mask=im_mask)

    return im_mask
