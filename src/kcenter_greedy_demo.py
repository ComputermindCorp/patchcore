import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.random_projection import SparseRandomProjection

from models.patch_core  import sampler

save_path = "data/kecenter_greedy_demo/kcenter_greedy_result.png"
marker_size = 5
m_color = "lightblue"
mc_color = "red"

save_path = Path(save_path)
save_path.parent.mkdir(parents=True, exist_ok=True)

n = 1000
ratio = 0.1

# matplot setting
fig = plt.figure(figsize=(6.0, 3.0))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

# k-center greedy
torch.manual_seed(0)
data = torch.rand(n, 2)
data_subsample, n_subsample =  sampler.k_center_greedy(data, ratio, random_projection=False, seed=1, progress=True)

ax1.scatter(data[:, 0], data[:, 1], s=marker_size, c=m_color, label="M")
ax1.scatter(data_subsample[:, 0], data_subsample[:, 1], s=marker_size, c=mc_color, label="Mc")
ax1.set_title("k-center greedy")
ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
ax1.tick_params(labelleft=False, labelbottom=False)

print(f"k-center greedy {data.shape} -> {data_subsample.shape}")


# random
n_subsample = int(len(data) * ratio)
random_indexes = torch.randint(len(data), [n_subsample])
data_subsample = data[random_indexes, :]

print(f"random {data.shape} -> {data_subsample.shape}")

ax2.scatter(data[:, 0], data[:, 1], s=marker_size, c=m_color, label="M")
ax2.scatter(data_subsample[:, 0], data_subsample[:, 1], s=marker_size, c=mc_color, label="Mc")
ax2.set_title("random")
ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2)
ax2.tick_params(labelleft=False, labelbottom=False)

# グラフを保存
plt.savefig(save_path)
