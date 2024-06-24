import torch
import torch.optim as optim
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class SqueezeNetModel(BaseModel):

    def get_model(self):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
        model.eval()
        return model
    
    def get_tuning_optimizer(self, model):
        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) if self.trial else 1e-3
        return optim.Adam(model.parameters(), lr=learning_rate)