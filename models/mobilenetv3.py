import torch.optim as optim
from torchvision.models import mobilenet_v3_small
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class ResNet18Model(BaseModel):

    def get_model(self):
        model = mobilenet_v3_small(pretrained=False, num_classes=10)

    def get_tuning_optimizer(self, model):
        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) if self.trial else 1e-3
        return optim.Adam(model.parameters(), lr=learning_rate)