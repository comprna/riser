import torch
from torch.utils.data.dataset import Dataset

class SignalDataset(Dataset):
    def __init__(self, positive_file, negative_file):
        n_x = torch.load(negative_file)
        p_x = torch.load(positive_file)

        print(f"Shape of negative tensor: \t{n_x.shape}")
        print(f"Shape of positive tensor: \t{p_x.shape}")

        n_y = torch.zeros(n_x.shape[0], dtype=torch.long)
        p_y = torch.ones(p_x.shape[0], dtype=torch.long)

        self.data  = torch.cat((n_x, p_x))
        self.label = torch.cat((n_y, p_y))

        print(f"Shape of total dataset: {self.data.shape}")
        print(f"Shape of total labels: {self.label.shape}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y
