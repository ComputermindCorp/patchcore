from .wide_resnet50 import WideResNet50
from .resnet50 import ResNet50
from .resnet18 import ResNet18
from .resnet18_onnx import ResNet18_ONNX
from .resnet18_quantization_onnx import ResNet18_quantization_ONNX

backborn_list = {
    "wide_resnet50": WideResNet50,
    "resnet50": ResNet50,
    "resnet18": ResNet18,
    "resnet18_onnx": ResNet18_ONNX,
    "resnet18_quantization_onnx": ResNet18_quantization_ONNX
}
