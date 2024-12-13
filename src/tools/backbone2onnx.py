import torch
from pathlib import Path

from models.patch_core import PatchCore

opset_version = 12

arch_id_list = [
    "resnet50",
    "resnet18",
    "wide_resnet50",
]

output_dir = "onnx_output"
output_dir = Path(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

input = torch.randn(1, 3, 224, 224)
for arch_id in arch_id_list:
    net = PatchCore(
        "cpu",
        (224, 224),
        arch_id,
    )

    torch.onnx.export(
        net.backborn.extractor,
        input,
        output_dir / f"backbone_{arch_id}.onnx", 
        verbose=True,
        output_names=net.backborn.layers,
        opset_version=opset_version,
        )
