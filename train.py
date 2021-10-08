from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
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


# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
    
#         # Initial layer

#             # 20 channels


#         # 4 residual layers

#             # Each layer has 2 Bottleneck units

#                 # 1 x 1 conv
#                 # BN
#                 # ReLU
#                 # 3 x 3 conv
#                 # BN
#                 # ReLU
#                 # 1 x 1 conv
#                 # BN

#                 # Stride of 2

#                 # Sum output and input of unit
            
#             # Layer 1: 20 channels
#             # Layer 2: 30 channels
#             # Layer 3: 45 channels
#             # Layer 4: 67 channels
        

#         # Fully connected layer + softmax

#             # Mean pooling

#             # Fully connected layer

#             # Softmax activation



#     def forward(self, x):
#         # TODO


def main():

    # Determine whether to use CPU or GPU

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using {device} device")

    # Create dataset

    data_dir = "/g/data/xc17/Eyras/alex/working/rna-classifier/5_MakeDataset"
    train_cfile = f"{data_dir}/train_coding.pt"
    train_nfile = f"{data_dir}/train_noncoding.pt"
    test_cfile = f"{data_dir}/test_coding.pt"
    test_nfile = f"{data_dir}/test_noncoding.pt"

    train_data = SignalDataset(train_cfile, train_nfile, device)
    test_data = SignalDataset(test_cfile, test_nfile, device)

    # Create data loaders

    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # for x, y in train_loader:
    #     print(f"Shape of x: {x.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")

    #     x = x.to('cpu')
    #     x_d = x[0]

    #     plt.plot(x_d)
    #     plt.show()




if __name__ == "__main__":
    main()
 