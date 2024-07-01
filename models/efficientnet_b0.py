from torchvision.models import efficientnet_b0
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class EfficientNet_B0Model(BaseModel):

    def get_model(self):
        return efficientnet_b0(weights=None, num_classes=10)
    
    def get_tuning_optimizer(self, model):
        return super().get_tuning_optimizer(model)
    
    def get_logits(self, outputs):
        return super().get_logits(outputs)