import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class SignalDataset(Dataset):
    def __init__(self, data_file, labels_file):
        self.data = np.load(data_file, mmap_mode='r')
        self.label = np.load(labels_file, mmap_mode='r')

        print(f"Shape of total dataset: {self.data.shape}")
        print(f"Shape of total labels: {self.label.shape}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y
