from torchvision.models import densenet121
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class DenseNet121Model(BaseModel):

    def _get_model(self):
        return densenet121(weights=None, num_classes=10)
    
    def get_tuning_optimizer(self, model):
        return super().get_optimizer(model)
    
    def get_logits(self, outputs):
        return super().get_logits(outputs)