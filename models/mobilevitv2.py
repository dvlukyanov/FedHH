from transformers import MobileViTV2Config, MobileViTV2Model
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class MobileViTV2(BaseModel):

    def get_model(self):
        configuration = MobileViTV2Config()
        return MobileViTV2Model(configuration)
    
    def get_tuning_optimizer(self, model):
        # TODO https://huggingface.co/docs/transformers/en/model_doc/mobilevitv2
        pass