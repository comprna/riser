import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchinfo import summary

from utilities import get_config


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# TODO: Inherit ResidualBlock??
class TemporalBlock(nn.Module):
    def __init__(self, in_chan, out_chan, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        # Parameters
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.activation = nn.ReLU(inplace=True)

        # Causal convolutional blocks
        self.blocks = nn.Sequential(
            self.conv_block(in_chan, out_chan, kernel_size, stride, dilation, padding, dropout),
            self.conv_block(out_chan, out_chan, kernel_size, stride, dilation, padding, dropout),
        )
        
        # Match dimensions of block's input and output for summation
        self.shortcut = nn.Conv1d(in_chan, out_chan, 1)

        # Initialise weights and biases
        self._init_weights()

    # TODO: Pass config as param
    def conv_block(self, in_chan, out_chan, kernel_size, stride, dilation, padding, dropout):
        layers = [
            weight_norm(nn.Conv1d(in_chan, out_chan, kernel_size, stride=stride, padding=padding, dilation=dilation)),
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
    
    # TODO: Move to TCN class
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
    
    @property
    def should_apply_shortcut(self):
        return self.in_chan != self.out_chan


class TCN(nn.Module):
    # TODO: Pass config
    # Layer channels = # channels output from each hidden layer
    def __init__(self, c):
        super().__init__()
        
        # Feature extractor layers
        layers = []
        for i in range(c.n_layers):
            dilation = 2 ** i
            in_chan = c.in_chan if i == 0 else c.n_filters
            out_chan = c.n_filters
            layers += [TemporalBlock(in_chan, out_chan, c.kernel_size, stride=1, dilation=dilation,
                                     padding=(c.kernel_size-1) * dilation, dropout=c.dropout)]
        self.layers = nn.Sequential(*layers)

        # Classifier
        self.linear = nn.Linear(c.n_filters, c.n_classes)


    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x[:,:,-1]) # TODO: Why the third dimension??
        x = F.log_softmax(x, dim=1) # TODO: What is this for?
        return x


    # TODO: Function to calculate receptive field.  See https://github.com/locuslab/TCN/issues/44

def main():
    config = get_config('config.yaml')

    model = TCN(config.tcn)

    summary(model, input_size=(64, 1, 9036)) # (batch_size, dimension, seq_length)


if __name__ == "__main__":
    main()
    