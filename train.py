import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class SignalDataset(Dataset):
    def __init__(self, coding_file, non_coding_file, device):
        c_np = np.load(coding_file, allow_pickle=True)
        n_np = np.load(non_coding_file, allow_pickle=True)

        c_x = torch.from_numpy(c_np).to(torch.device(device))
        n_x = torch.from_numpy(n_np).to(torch.device(device))

        c_y = torch.zeros(c_x.shape[0], device=torch.device(device))
        n_y = torch.ones(n_x.shape[0], device=torch.device(device))

        self.data  = torch.cat((c_x, n_x))
        self.label = torch.cat((c_y, n_y))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y


def main():

    # Determine whether to use CPU or GPU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # Create dataset

    coding_file = "hek293_test_coding_9036.npy"
    non_coding_file = "hek293_test_other_9036.npy"

    train_data = SignalDataset(coding_file, non_coding_file, device)


    # Create data loaders

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    for x, y in train_loader:
        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")




if __name__ == "__main__":
    main()
 