import torch.optim as optim
from torchvision.models import googlenet
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class GoogLeNetModel(BaseModel):

    def get_model(self):
        return googlenet(parameters=None, num_classes=10)

    def get_tuning_optimizer(self, model):
        # TODO
        pass