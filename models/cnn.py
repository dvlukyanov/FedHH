import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .base_model import BaseModel


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class CustomCNNModel(BaseModel):
    def get_model(self):
        class CustomCNN(nn.Module):
            def __init__(self, conv1_out_channels, conv2_out_channels, dropout):
                super(CustomCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, conv1_out_channels, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, kernel_size=3, stride=1, padding=1)
                self.fc1 = nn.Linear(conv2_out_channels * 8 * 8, 256)
                self.fc2 = nn.Linear(256, 10)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                x = x.view(-1, self.num_flat_features(x))
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x

            def num_flat_features(self, x):
                size = x.size()[1:]  # all dimensions except the batch dimension
                num_features = 1
                for s in size:
                    num_features *= s
                return num_features

        if self.trial:
            conv1_out_channels = self.trial.suggest_int('conv1_out_channels', 16, 64)
            conv2_out_channels = self.trial.suggest_int('conv2_out_channels', 32, 128)
            dropout = self.trial.suggest_float('dropout', 0.25, 0.5)
        else:
            conv1_out_channels = 16
            conv2_out_channels = 64
            dropout = 0.3
        model = CustomCNN(conv1_out_channels, conv2_out_channels, dropout)
        return model

    def get_tuning_optimizer(self, model):
        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) if self.trial else 1e-3
        return optim.Adam(model.parameters(), lr=learning_rate)