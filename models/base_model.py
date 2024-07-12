import torch
import torch.optim as optim


__author__ = 'Dmitry Lukyanov, Huaye Li'
__email__ = 'dmitry@dmitrylukyanov.com, huayel@g.clemson.edu'
__license__ = 'MIT'


class BaseModel:
    def __init__(self, trial=None):
        self.trial = trial

    def get_model(self):
        raise NotImplementedError

    def get_tuning_optimizer(self, model):
        if self.trial:
            learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            beta1 = self.trial.suggest_float('beta1', 0.8, 0.99)
            beta2 = self.trial.suggest_float('beta2', 0.9, 0.999)
            eps = self.trial.suggest_float('eps', 1e-8, 1e-6, log=True)
            weight_decay = self.trial.suggest_float('weight_decay', 0, 1e-2)
        else:
            learning_rate = 1e-3
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            weight_decay = 0
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay
        )
    
    def get_tuning_scheduler(self, optimizer):
        if self.trial:
            scheduler_type = self.trial.suggest_categorical('scheduler_type', ['StepLR', 'ExponentialLR', 'CosineAnnealingLR'])
            if scheduler_type == 'StepLR':
                step_size = self.trial.suggest_int('step_size', 1, 10)
                gamma = self.trial.suggest_float('gamma', 0.1, 0.9)
                scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            elif scheduler_type == 'ExponentialLR':
                gamma = self.trial.suggest_float('gamma', 0.85, 0.99)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            elif scheduler_type == 'CosineAnnealingLR':
                T_max = self.trial.suggest_int('T_max', 10, 50) if self.trial else 20
                eta_min = self.trial.suggest_float('eta_min', 0, 1e-2) if self.trial else 0
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        else:
            scheduler = None
        return scheduler

    def get_logits(self, outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs
        return outputs.logits

    def train_model(self, model, train_loader, criterion, optimizer, scheduler, device):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = self.get_logits(outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if scheduler:
            scheduler.step()
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
                outputs = self.get_logits(outputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        return running_loss / len(valid_loader), accuracy
