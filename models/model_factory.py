

__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class ModelFactory:
    
    def create_model(self, model_name):
        if model_name == 'cnn':
            return CustomCNNModel