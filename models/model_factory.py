from .cnn import CustomCNNModel
from .densenet121 import DenseNet121Model
from .efficientnet_b0 import EfficientNet_B0Model
from .googlenet import GoogLeNetModel
from .mobilenetv2 import MobileNetV2Model
from .mobilenetv3 import MobileNetV3SmallModel
from .resnet18 import ResNet18Model
from .shufflenetv2x1_5 import ShuffleNetV2x1_5Model
from .squeezenet import SqueezeNetModel
from .mnasnet0_75 import MNASNet0_75Model


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class ModelFactory:
    
    @staticmethod
    def create(model_name, trial=None):
        if model_name == 'cnn':
            return CustomCNNModel(trial)
        elif model_name == 'densenet121':
            return DenseNet121Model(trial)
        elif model_name == 'efficientnetb0':
            return EfficientNet_B0Model(trial)
        elif model_name == 'googlenet':
            return GoogLeNetModel(trial)
        elif model_name == 'mobilenetv2':
            return MobileNetV2Model(trial)
        elif model_name == 'mobilenetv3small':
            return MobileNetV3SmallModel(trial)
        elif model_name == 'resnet18':
            return ResNet18Model(trial)
        elif model_name == 'shufflenetv2':
            return ShuffleNetV2x1_5Model(trial)
        elif model_name == 'squeezenet':
            return SqueezeNetModel(trial)
        elif model_name == 'mnasnet0_75':
            return MNASNet0_75Model(trial)
