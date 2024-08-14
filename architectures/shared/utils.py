import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config
from models.model_factory import ModelFactory

def split_data(df, folds):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    fold_indices = {i: [] for i in range(folds)}
    for i, (_, test_index) in enumerate(skf.split(df['filename'], df['label'])):
        fold_indices[i] = test_index
    fold_dfs = {i: df.iloc[fold_indices[i]].reset_index(drop=True) for i in range(folds)}
    print(f"Data is split into {len(fold_dfs)} partitions: " + ", ".join([f"{len(fold_dfs[i])}" for i in range(len(fold_dfs))]))
    return fold_dfs


def load_model(model_type, folder, model_name):
    model_path = os.path.join(folder, model_name)
    model = ModelFactory.create(model_type)
    model.get_model().load_state_dict(torch.load(model_path))
    model.get_model().eval()
    print(f"Model {model_name} ({model_type}) is loaded")
    return model


def save_model(model, model_type, folder, model_name):
    model_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f'Model {model_path} ({model_type}) is saved')