from torchvision.models import resnet18
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class ResNet18Model(BaseModel):

    def get_model(self):
        return resnet18(weights=None, num_classes=10)