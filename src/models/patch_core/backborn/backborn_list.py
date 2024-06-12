from .wide_resnet50 import WideResNet50
from .resnet50 import ResNet50
from .resnet18 import ResNet18

backborn_list = {
    "wide_resnet50": WideResNet50,
    "resnet50": ResNet50,
    "resnet18": ResNet18,
}
