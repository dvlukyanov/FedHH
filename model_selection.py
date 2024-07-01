import sys
import logging
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.model_factory import ModelFactory
from data.dataset import CustomImageDataset


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


def objective(trial):
    batch_size = 64

    dataset = CustomImageDataset(img_dir='data/tuning/images', labels_file='data/tuning/labels.csv')
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model_instance = ModelFactory.create_model('googlenet', trial)

    model = model_instance.get_model()
    optimizer = model_instance.get_tuning_optimizer(model)
    scheduler = model_instance.get_tuning_scheduler(optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    n_epochs = 10
    for epoch in range(n_epochs):
        train_loss = model_instance.train_model(model, train_loader, criterion, optimizer, scheduler, device)
        valid_loss, accuracy = model_instance.validate_model(model, valid_loader, criterion, device)
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy


SEED = 0

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.DEBUG)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED), pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=200)

print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))