import torch
import torch.optim as optim


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class BaseModel:
    def __init__(self, trial=None):
        self.trial = trial

    def get_model(self):
        raise NotImplementedError
    
    def get_resize_transform(self):
        raise NotImplementedError
    
    def get_normalize_transform(self):
        raise NotImplementedError

    def get_tuning_optimizer(self, model):
        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True) if self.trial else 1e-3
        return optim.Adam(model.parameters(), lr=learning_rate)

    def train_model(self, model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate_model(self, model, valid_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return running_loss / len(valid_loader), accuracy