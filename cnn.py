from torch import nn
from torchinfo import summary

from utilities import get_config


class ConvNet(nn.Module):
    def __init__(self, c):
        super().__init__()

        # Convolutional layers
        layers = []
        for i in range(c.n_layers):
            # First layer takes 1D time-series input while the rest
            # take channel outputs from previous layer
            in_channels = 1 if i == 0 else c.layer_channels[i-1]
            layers.append(self._make_layer(in_channels, c.layer_channels[i], c.layer_kernels[i]))
        self.layers = nn.ModuleList(layers)

        # Classifier
        if c.classifier == 'fc':
            self.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(67*562, 4096), # TODO: Hardcoded
                nn.ReLU(inplace=True),
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
        x = x.unsqueeze(1) # Add dimension to represent 1D input
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)

        # NB: Softmax not needed since it is incorporated into
        # torch implementation of CrossEntropyLoss. Can send the raw
        # logits there.

        return x

    def _make_layer(self, in_channels, out_channels, kernel_size):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)


def main():
    config = get_config('config.yaml')
    model = ConvNet(config.cnn)
    summary(model, input_size=(64, 9036))


if __name__ == "__main__":
    main()
