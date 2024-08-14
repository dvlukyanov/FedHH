import torch
from torchvision import transforms
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
        return super().get_optimizer(model)
    
    def get_logits(self, outputs):
        return super().get_logits(outputs)