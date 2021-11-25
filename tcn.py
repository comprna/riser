import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchinfo import summary


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


# TODO: Move inside class
def conv_block(in_chan, out_chan, kernel_size, stride, dilation, padding, dropout):
    layers = [
        weight_norm(nn.Conv1d(in_chan, out_chan, kernel_size, stride=stride, padding=padding, dilation=dilation)),
        Chomp1d(padding),
        nn.ReLU(),
        nn.Dropout(dropout)
    ]
    return nn.Sequential(*layers)


class TemporalBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # Parameters
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.activation = nn.ReLU(inplace=True)

        # Causal convolutional blocks
        self.blocks = nn.Sequential(
            conv_block(in_chan, out_chan, kernel_size, stride, dilation, padding, dropout),
            conv_block(out_chan, out_chan, kernel_size, stride, dilation, padding, dropout),
        )
        
        # Match dimensions of block's input and output for summation
        self.shortcut = nn.Conv1d(in_chan, out_chan, 1)

        # Initialise weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, x):
        residual = self.shortcut(x) if self.should_apply_shortcut else x
        out = self.blocks(x)
        out += residual
        out = self.activation(out)
        return out
    
    @property
    def should_apply_shortcut(self):
        return self.in_chan != self.out_chan


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

        self.linear = nn.Linear(num_channels[-1], 2) # TODO: Parameterise output_size

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x[:,:,-1]) # TODO: Why the third dimension??
        x = F.log_softmax(x, dim=1) # TODO: What is this for?
        return x


if __name__ == "__main__":
    input_channels = 1
    n_hidden_units_per_layer = 25
    n_levels = 4
    channel_sizes = [n_hidden_units_per_layer] * n_levels
    model = TemporalConvNet(input_channels, channel_sizes)
    summary(model, input_size=(64, 1, 9036)) # (batch_size, dimension, seq_length)