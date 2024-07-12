import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import torch


__author__ = 'Dmitry Lukyanov'
__email__ = 'dmitry@dmitrylukyanov.com'
__license__ = 'MIT'


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, labels_file):
        self.img_dir = img_dir
        labels = pd.read_csv(labels_file)
        self.file_names = labels['filename'].tolist()
        labels_unique = sorted(set(labels['label']))
        keys = {label: idx for idx, label in enumerate(labels_unique)}
        labels_onehot = torch.zeros(size=(len(labels), len(labels_unique)))
        for idx, label in enumerate(labels['label']):
            labels_onehot[idx][keys[label]] = 1
        self.img_labels = labels_onehot

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.img_dir, str(self.file_names[idx]))
            image = Image.open(img_path).convert("RGB")
            image = self.__get_transform()(image)
            label = self.img_labels[idx].argmax().item()
            return image, label
        except Exception as e:
            print(f"Skipping file {self.file_names[idx]} due to error: {e}")
            return None
        
    def __iter__(self):
        for i in range(len(self)):
            item = self[i]
            if item is not None:
                yield item
    
    def __get_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])