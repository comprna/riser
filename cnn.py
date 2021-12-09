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
            in_channels = 1 if i == 0 else c.channels[i-1]
            layers.append(self._make_layer(in_channels, c.channels[i], c.kernels[i], c.depth))
        self.layers = nn.ModuleList(layers)

        # Classifier
        if c.classifier == 'fc':
            self.classifier = nn.Sequential(
                nn.Flatten(1),
                nn.Linear(67*753, 4096), # TODO: Hardcoded
                nn.ReLU(inplace=True),
                nn.Linear(4096, c.n_classes) # TODO: Hardcoded
            )
        elif c.classifier == 'gap_fc':
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(1),
                nn.Linear(c.channels[-1], c.n_classes)
            )
        elif c.classifier == 'gap':
            self.classifier = nn.Sequential(
                nn.Conv1d(c.channels[-1], c.n_classes, 1),
                nn.AdaptiveAvgPool1d(1)
            )
        else:
            print("Typo in config file: Classifier = {c.classifier}")
            exit()

    def forward(self, x):
        x = x.unsqueeze(1)
        for layer in self.layers:
            x = layer(x)
        x = self.classifier(x)
        if len(x.shape) == 3 and x.shape[2] == 1:
            x = x.squeeze() # Remove extra dimension added by avg pool
        return x

    def _make_layer(self, in_channels, out_channels, kernel_size, depth):
        layers = []
        for i in range(depth):
            layers.append(nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size,
                                    stride=1,
                                    padding='same'))
            layers.append(nn.ReLU(inplace=True))
            # Only the first block takes input from the previous layer
            if i == 0:
                in_channels = out_channels
        layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)


def main():
    config = get_config('config-cnn.yaml')
    model = ConvNet(config.cnn)
    summary(model, input_size=(config.batch_size, 12048))


if __name__ == "__main__":
    main()
