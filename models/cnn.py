import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov, Huaye Li'
__email__ = 'dmitry@dmitrylukyanov.com, huayel@g.clemson.edu'
__license__ = 'MIT'


class CustomCNNModel(BaseModel):
    def get_model(self):
        class CustomCNN(nn.Module):
            def __init__(self, conv_layers, fc_layers, dropout):
                super(CustomCNN, self).__init__()

                self.conv_layers = nn.ModuleList()
                in_channels = 3
                for out_channels in conv_layers:
                    self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
                    in_channels = out_channels

                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

                # Assuming input image size is 256x256
                conv_output_size = 256 // (2 ** len(conv_layers))
                num_flat_features = conv_layers[-1] * conv_output_size * conv_output_size

                self.fc_layers = nn.ModuleList()
                in_features = num_flat_features
                for out_features in fc_layers:
                    self.fc_layers.append(nn.Linear(in_features, out_features))
                    in_features = out_features

                self.dropout = nn.Dropout(dropout)
                self.output_layer = nn.Linear(fc_layers[-1], 10)

            def forward(self, x):
                for conv in self.conv_layers:
                    x = self.pool(F.relu(conv(x)))
                x = x.view(-1, self.num_flat_features(x))
                for fc in self.fc_layers:
                    x = F.relu(fc(x))
                    x = self.dropout(x)
                x = self.output_layer(x)
                return x

            def num_flat_features(self, x):
                size = x.size()[1:]  # all dimensions except the batch dimension
                num_features = 1
                for s in size:
                    num_features *= s
                return num_features

        if self.trial:
            conv_layers = []
            num_conv_layers = self.trial.suggest_int('num_conv_layers', 1, 6)
            for i in range(num_conv_layers):
                conv_layers.append(self.trial.suggest_int(f'conv{i + 1}_out_channels', 16, 256))

            fc_layers = []
            num_fc_layers = self.trial.suggest_int('num_fc_layers', 1, 3)
            for i in range(num_fc_layers):
                fc_layers.append(self.trial.suggest_int(f'fc{i + 1}_size', 64, 4096))

            dropout = self.trial.suggest_float('dropout', 0.25, 0.5)
        else:
            conv_layers = [16,32,64,128,256,256]
            fc_layers = [1024,512,256]
            dropout = 0.3

        model = CustomCNN(conv_layers, fc_layers, dropout)
        return model
    
    def get_tuning_optimizer(self, model):
        return super().get_tuning_optimizer(model)
    
    def get_logits(self, outputs):
        return super().get_logits(outputs)