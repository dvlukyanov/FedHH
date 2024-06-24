import torch.optim as optim
from torchvision.models import densenet121
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class DenseNet121Model(BaseModel):

    def get_model(self):
        return densenet121(parameters=None, num_classes=10)

    def get_tuning_optimizer(self, model):
        # TODO
        pass