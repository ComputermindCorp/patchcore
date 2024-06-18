import torch
import matplotlib.pyplot as plt
from pathlib import Path

from models.patch_core  import sampler

save_path = "data/kcenter_greedy.png"
save_path = Path(save_path)
save_path.parent.mkdir(parents=True, exist_ok=True)

n = 10000
ratio = 0.1

torch.manual_seed(0)
data = torch.rand(n, 100)
plt.scatter(data[0], data[1])

data_subsample, n_subsample =  sampler.k_center_greedy(data, ratio, seed=1, progress=True)
plt.scatter(data_subsample[0], data_subsample[1], c="r")

print(f"k-center greedy {data.shape} -> {data_subsample.shape}")

plt.savefig(save_path)
