import torch
import argparse

from models.patch_core import PatchCore

weights_path = "data/weights/wide_resnet50_size224_param_0.1_9.pth"

net = PatchCore.load_weights(weights_path)

print(f"weights file: {weights_path}")
print(f"coreset_sampling_ratio: {net.coreset_sampling_ratio}")
print(f"num_neighbors: {net.num_neighbors}")
print(f"memory_bank size: {net.memory_bank.shape}")
print(f"n_train: {net.n_train}")
