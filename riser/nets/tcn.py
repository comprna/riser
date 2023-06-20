from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchinfo import summary

# from utilities import get_config


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dilation, padding, dropout=0.2):
        super().__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU(inplace=True)

        # Causal convolutional blocks
        self.blocks = nn.Sequential(
            self.conv_block(in_channels, out_channels, kernel, dilation, padding, dropout),
            self.conv_block(out_channels, out_channels, kernel, dilation, padding, dropout),
        )

        # Match dimensions of block's input and output for summation
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1)

        # Initialise weights and biases
        self._init_weights()

    def conv_block(self, in_channels, out_channels, kernel, dilation, padding, dropout):
        layers = [
            weight_norm(nn.Conv1d(in_channels, out_channels, kernel, stride=1, padding=padding, dilation=dilation)),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.shortcut(x) if self.should_apply_shortcut else x
        out = self.blocks(x)
        out = self.activation(out + residual)
        return out

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels


class TCN(nn.Module):
    def __init__(self, c):
        super().__init__()

        # Feature extractor layers
        layers = []
        for i in range(c.n_layers):
            dilation = c.dilation ** i
            in_channels = c.in_channels if i == 0 else c.n_filters
            out_channels = c.n_filters
            layers += [TemporalBlock(in_channels,
                                     out_channels,
                                     c.kernel,
                                     dilation=dilation,
                                     padding=(c.kernel-1) * dilation,
                                     dropout=c.dropout)]
        self.layers = nn.Sequential(*layers)

        # Classifier
        self.linear = nn.Linear(c.n_filters, c.n_classes)

        print(f"Receptive field: {self.get_receptive_field(c.kernel, c.n_layers, c.dilation)}")

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.layers(x)
        x = self.linear(x[:,:,-1]) # Receptive field of last value covers entire input
        return x

    def get_receptive_field(self, kernel, n_layers, dilation):
        return 1 + 2 * sum([dilation**i * (kernel-1) for i in range(n_layers)])

# def main():
#     config = get_config('local_data/configs/train-tcn-13.yaml')
#     model = TCN(config.tcn)
#     summary(model, input_size=(32, 12048))


# if __name__ == "__main__":
#     main()
