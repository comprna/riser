import torch
from torch.utils.data.dataset import Dataset

class SignalDataset(Dataset):
    def __init__(self, coding_file, noncoding_file):
        n_x = torch.load(noncoding_file)
        c_x = torch.load(coding_file)

        print(f"Shape of noncoding tensor: \t{n_x.shape}")
        print(f"Shape of coding tensor: \t{c_x.shape}")

        n_y = torch.zeros(n_x.shape[0], dtype=torch.long)
        c_y = torch.ones(c_x.shape[0], dtype=torch.long)

        self.data  = torch.cat((n_x, c_x))
        self.label = torch.cat((n_y, c_y))

        print(f"Shape of total dataset: {self.data.shape}")
        print(f"Shape of total labels: {self.label.shape}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y
