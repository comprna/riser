from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


class SignalDataset(Dataset):
    def __init__(self, coding_file, noncoding_file, device):
        c_x = torch.load(coding_file).to(device)
        n_x = torch.load(noncoding_file).to(device)

        c_y = torch.zeros(c_x.shape[0], device=device)
        n_y = torch.ones(n_x.shape[0], device=device)

        self.data  = torch.cat((c_x, n_x))
        self.label = torch.cat((c_y, n_y))

        print(self.data.shape)
        print(self.label.shape)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()


def main():

    # Determine whether to use CPU or GPU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Create dataset

    coding_file = "data.pt"
    noncoding_file = "data.pt"

    train_data = SignalDataset(coding_file, noncoding_file, device)

    # Create data loaders

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    for x, y in train_loader:
        print(f"Shape of x: {x.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")

        x = x.to('cpu')
        x_d = x[0]

        plt.plot(x_d)
        plt.show()




if __name__ == "__main__":
    main()
 