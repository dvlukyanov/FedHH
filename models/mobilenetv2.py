from torchvision.models import mobilenet_v2
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class MobileNetV2Model(BaseModel):

    def get_model(self):
        return mobilenet_v2(parameters=None, num_classes=10)