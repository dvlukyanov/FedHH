import os
import sys
import logging
import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from models.model_factory import ModelFactory
from data.dataset import CustomImageDataset


__author__ = 'Dmitry Lukyanov, Huaye Li'
__email__ = 'dmitry@dmitrylukyanov.com, huayel@g.clemson.edu'
__license__ = 'MIT'


torch.autograd.profiler.profile(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.set_detect_anomaly(False, check_nan=True)
#torch.set_float32_matmul_precision('high')


BATCH_SIZE = 64
DATA_LOADER_WORKERS = 2


print(f'CUDA devices: {torch.cuda.device_count()}')


def objective(trial, model_name, img_dir, labels_file):
    batch_size = BATCH_SIZE * torch.cuda.device_count()

    dataset = CustomImageDataset(img_dir=img_dir, labels_file=labels_file)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=DATA_LOADER_WORKERS*torch.cuda.device_count(), prefetch_factor=DATA_LOADER_WORKERS*4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=DATA_LOADER_WORKERS*torch.cuda.device_count(), prefetch_factor=DATA_LOADER_WORKERS*4, pin_memory=True)

    model_instance = ModelFactory.create_model(model_name, trial)

    model = model_instance.get_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)
    optimizer = model_instance.get_tuning_optimizer(model)
    scheduler = model_instance.get_tuning_scheduler(optimizer)

    criterion = nn.CrossEntropyLoss()

    n_epochs = 10
    for epoch in range(n_epochs):
        train_loss = model_instance.train_model(model, train_loader, criterion, optimizer, scheduler, device)
        valid_loss, accuracy = model_instance.validate_model(model, valid_loader, criterion, device)
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='Model for tuning')
parser.add_argument('--img_dir', type=str, required=True, help='Directory path for images')
parser.add_argument('--labels_file', type=str, required=True, help='Path to the labels file')
parser.add_argument('--seed', type=int, required=True, help='Seed')
args = parser.parse_args()

print(args)

optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
optuna.logging.set_verbosity(optuna.logging.DEBUG)
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=args.seed), pruner=optuna.pruners.HyperbandPruner())
study.optimize(lambda trial: objective(trial, args.model, args.img_dir, args.labels_file), n_trials=200)

print('Best trial:')
trial = study.best_trial
print('  Value: {}'.format(trial.value))
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))
