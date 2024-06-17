import torch
import argparse
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from tqdm.contrib import tenumerate
import math

from models.patch_core import PatchCore

def get_size(n):
    w = 2 ** math.ceil(math.log(math.sqrt(n), 2))
    h = n // w
    return w, h

def visualize(input_path, output_dir, weights_path):
    input_path = Path(input_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    
    net = PatchCore.load_weights(weights_path)

    print(f"weights file: {weights_path}")
    print(f"coreset_sampling_ratio: {net.coreset_sampling_ratio}")
    print(f"num_neighbors: {net.num_neighbors}")
    print(f"memory_bank size: {net.memory_bank.shape}")
    print(f"n_train: {net.n_train}")

    im = Image.open(input_path)
    x = net.get_transform()(im)
    x = net.get_resize()(x)
    x = torch.unsqueeze(x, 0)
    x = x.to(net.device)

    features = net.backborn.get_features(x)
    ap_features = { layer: torch.nn.AvgPool2d(3, 1, 1)(feature) for layer, feature in features.items() }

    for layer_name, features in ap_features.items():
        #output_sub_dir = output_dir / layer_name
        #output_sub_dir.mkdir(parents=True, exist_ok=True)

        w, h = get_size(len(features[0]))
        concat_features = np.zeros((features[0].shape[1] * h, features[0].shape[2] * w), np.uint8)

        for i, f in tenumerate(features[0], desc=layer_name):
            f = f.detach().cpu().numpy()
            f = f * 255
            f = f.astype(np.uint8)
            #im_result = Image.fromarray(f)
            #im_result.save(output_sub_dir / f"{layer_name}_{i:04}.png")

            y = (i // w) * f.shape[0]
            x = (i % w) * f.shape[1] 
            concat_features[y:y+f.shape[0], x:x+f.shape[1]] = f

        Image.fromarray(concat_features).save(output_dir / f"{layer_name}.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', help='input path')
    parser.add_argument('weights_path', help='weights path')
    parser.add_argument('-o', '--output_dir', default='output', help='output directory path')
    parser.add_argument('--bench', action='store_true', help='enabel benchmark')
    args = parser.parse_args()

    visualize(args.input_path, args.output_dir, args.weights_path)
