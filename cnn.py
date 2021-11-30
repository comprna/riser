from torch import nn
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
        if c.classifier == 'fc':
            self.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(37654, 4096), # TODO: Hardcoded
                nn.Linear(4096, c.n_classes)
            )
        elif c.classifier == 'gap_fc':
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
                nn.Linear(c.layer_channels[-1], c.n_classes)
            )
        elif c.classifier == 'gap':
            self.classifier = nn.Sequential(
                nn.Conv1d(c.layer_channels[-1], c.n_classes, 1),
                nn.AdaptiveAvgPool1d(1)
            )
        else:
            print("Typo in config file: Classifier = {c.classifier}")

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)

        x = self.classifier(x)

        return x

    def _make_layer(self, in_channels, out_channels, kernel_size, last=False):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ]

        # Activate all but the last hidden layer
        if last == False:
            layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)


def main():
    config = get_config('config.yaml')
    model = ConvNet(config.cnn)
    summary(model, input_size=(64, 1, 9036)) # (batch_size, dimension, seq_length)


if __name__ == "__main__":
    main()
