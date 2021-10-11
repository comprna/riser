import torch
from torch.utils.data.dataset import Dataset

class SignalDataset(Dataset):
    def __init__(self, coding_file, noncoding_file):
        c_x = torch.load(coding_file)
        n_x = torch.load(noncoding_file)

        print(f"Shape of coding tensor: \t{c_x.shape}")
        print(f"Shape of noncoding tensor: \t{n_x.shape}")

        c_y = torch.zeros(c_x.shape[0])
        n_y = torch.ones(n_x.shape[0])

        self.data  = torch.cat((c_x, n_x))
        self.label = torch.cat((c_y, n_y))

        print(f"Shape of total dataset: {self.data.shape}")
        print(f"Shape of total labels: {self.label.shape}")

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y
