from torchvision.models import resnet18
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class ResNet18Model(BaseModel):

    def _get_model(self):
        return resnet18(weights=None, num_classes=10)
    
    def get_tuning_optimizer(self, model):
        return super().get_optimizer(model)
    
    def get_logits(self, outputs):
        return super().get_logits(outputs)