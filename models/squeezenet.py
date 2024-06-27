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
    
    def get_resize_transform(self):
        return transforms.Resize(256)
    
    def get_normalize_transform(self):
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])