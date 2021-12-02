from torch import nn
from torchinfo import summary

from utilities import get_config


class ConvRecNet(nn.Module):
    def __init__(self, c):
        super().__init__()

        # Convolutional layers
        conv_layers = []
        for i in range(c.n_conv_layers):
            # First layer takes 1D time-series input
            in_channels = 1 if i == 0 else c.channels[i-1]
            layer = self._make_conv_layer(in_channels, c.channels[i], c.kernels[i])
            conv_layers.append(layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        # Recurrent layers
        rec_layers = []
        for i in range(c.n_rec_layers):
            # First layer takes last conv layer output
            output_dim = c.hidden * 2 if c.bidirectional else c.hidden
            input_dim = c.channels[-1] if i == 0 else output_dim
            rec_layers.append(nn.LSTM(input_size=input_dim,
                                      hidden_size=c.hidden,
                                      num_layers=c.n_rec_layers,
                                      batch_first=True,
                                      dropout=c.dropout,
                                      bidirectional=c.bidirectional))
        self.rec_layers = nn.ModuleList(rec_layers)

        # Activation
        self.activation = nn.ReLU(inplace=True)

        # Classifier
        self.linear = nn.Linear(c.hidden, c.n_classes)


    def forward(self, x):
        x = x.unsqueeze(1) # Add dimension to represent 1D input
        for layer in self.conv_layers:
            x = layer(x)

        # CNN output is (B,C,L) but LSTM input needs to be (B,L,C)
        x = x.permute(0, 2, 1)

        for layer in self.rec_layers:
            x, _ = layer(x)
            x = self.activation(x)

        # TODO: Deal with bidirectional output
        # TODO: Implement GRU

        x = self.linear(x[:, -1, :]) # Hidden states for the last timestep

        return x

    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)


def main():
    config = get_config('config.yaml')
    model = ConvRecNet(config.cnn_rnn)
    summary(model, input_size=(64, 9036))


if __name__ == "__main__":
    main()
