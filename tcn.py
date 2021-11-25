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
    def __init__(self, input_chan, layer_chan, n_classes, kernel_size=2, dropout=0.2):
        super().__init__()
        
        # Feature extractor layers
        layers = []
        n_layers = len(layer_chan)
        for i in range(n_layers):
            dilation_size = 2 ** i
            in_chan = input_chan if i == 0 else layer_chan[i-1]
            out_chan = layer_chan[i]
            layers += [TemporalBlock(in_chan, out_chan, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.layers = nn.Sequential(*layers)

        # Classifier
        self.linear = nn.Linear(layer_chan[-1], n_classes)


    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x[:,:,-1]) # TODO: Why the third dimension??
        x = F.log_softmax(x, dim=1) # TODO: What is this for?
        return x


def main():
    input_chan = 1
    n_filters_per_layer = 25
    n_levels = 4
    layer_chan = [n_filters_per_layer] * n_levels
    model = TCN(input_chan, layer_chan, 2)
    summary(model, input_size=(64, 1, 9036)) # (batch_size, dimension, seq_length)


if __name__ == "__main__":
    main()
    