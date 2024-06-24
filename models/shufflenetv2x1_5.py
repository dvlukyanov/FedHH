import torch.optim as optim
from torchvision.models import shufflenet_v2_x1_5
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class ShuffleNetV2x1_5Model(BaseModel):

    def get_model(self):
        return shufflenet_v2_x1_5(parameters=None, num_classes=10)

    def get_tuning_optimizer(self, model):
        # TODO
        pass