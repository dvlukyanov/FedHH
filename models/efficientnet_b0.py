from torchvision.models import efficientnet_b0
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class EfficientNet_B0Model(BaseModel):

    def get_model(self):
        return efficientnet_b0(parameters=None, num_classes=10)