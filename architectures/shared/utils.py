import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from architectures.shared.config import Config
from models.model_factory import ModelFactory

def split_data(folds):
    df = pd.read_csv(Config()['storage']['data']['labels'])
    split_indices = {i: {cls: [] for cls in df['label'].unique()} for i in range(folds)}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=Config()['seed'])
    for i, (_, test_index) in enumerate(skf.split(df['filename'], df['label'])):
        for cls in df['label'].unique():
            label_indices = df[df['label'] == cls].index
            split_indices[i][cls] = np.intersect1d(label_indices, test_index).tolist()
    fold_indices = {i: [] for i in range(folds)}
    for i in range(folds):
        all_indices = []
        for cls in df['label'].unique():
            all_indices.extend(split_indices[i][cls])
        fold_indices[i] = list(set(all_indices))
    print(f"Data is split into {len(fold_indices)} partitions: " + ", ".join([f"{len(fold_indices[i])}" for i in range(len(fold_indices))]))
    return fold_indices


def load_model(model_type, folder, model_name):
    model_path = os.path.join(folder, model_name)
    model = ModelFactory.create(model_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model {model_name} ({model_type}) is loaded")
    return model


def save_model(model, model_type, folder, model_name):
    model_path = os.path.join(folder, model_name)
    torch.save(model.state_dict(), model_path)
    print(f'Model {model_path} ({model_type}) is saved')