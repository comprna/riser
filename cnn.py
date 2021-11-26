import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from utilities import get_config


class ConvNet(nn.Module):
    def __init__(self, c):
        super().__init__()

        # Convolutional layers
        layers = []
        for i in range(c.n_layers):
            if i == 0:
                layers.append(self._make_layer(1, c.layer_channels[i], c.layer_kernels[i]))
            else:
                layers.append(self._make_layer(c.layer_channels[i-1], c.layer_channels[i], c.layer_kernels[i]))
        self.conv_layers = nn.ModuleList(layers)

        # Classifier
        

        self.linear1 = nn.Linear()
        self.linear2 = nn.Linear()
        self.linear3 = nn.Linear()

        self.activation = nn.ReLU(inplace=True)
        
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.activation(x)

        x = self.linear3(x)

        return x

    def _make_layer(self, in_channels, out_channels, kernel_size, last=False):
        layers = [
            nn.Conv1D(in_channels, out_channels, kernel_size),
            nn.MaxPool1D(kernel_size=2, stride=2)
        ]

        if last == False:
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)


def main():
    config = get_config('config.yaml').cnn

    model = ConvNet(config)

    summary(model, input_size=(64, 1, 9036)) # (batch_size, dimension, seq_length)


if __name__ == "__main__":
    main()
    