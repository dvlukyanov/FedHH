from torchvision.models import mnasnet0_75
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class MNASNet0_75Model(BaseModel):

    def get_model(self):
        return mnasnet0_75(parameters=None, num_classes=10)