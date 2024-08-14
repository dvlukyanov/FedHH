from torchvision.models import shufflenet_v2_x1_5
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class ShuffleNetV2x1_5Model(BaseModel):

    def get_model(self):
        return shufflenet_v2_x1_5(weights=None, num_classes=10)
    
    def get_tuning_optimizer(self, model):
        return super().get_optimizer(model)
    
    def get_logits(self, outputs):
        return super().get_logits(outputs)