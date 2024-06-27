from torchvision.models import mobilenet_v3_small
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class MobileNetV3SmallModel(BaseModel):

    def get_model(self):
        return mobilenet_v3_small(parameters=None, num_classes=10)