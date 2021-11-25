from torch import nn
from torchinfo import summary

from utilities import get_config


class ResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()

        # Parameters
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.stride = stride
        self.activation = nn.ReLU(inplace=True)

        # Convolutional blocks
        self.blocks = nn.Identity()

        # Match dimensions of block's input and output for summation
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_chan)
        )

    def conv_block(self, in_chan, out_chan, kernel_size, last=False, **kwargs):
        layers = [
            nn.Conv1d(in_chan, out_chan, kernel_size, **kwargs),
            nn.BatchNorm1d(out_chan),
        ]
        if last == False:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = self.shortcut(x) if self.should_apply_shortcut else x
        out = self.blocks(x)
        out = self.activation(out + residual)
        return out

    @property
    def should_apply_shortcut(self):
        return self.in_chan != self.out_chan or self.stride != 1


class BasicBlock(ResidualBlock):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__(in_chan, out_chan, stride)

        self.blocks = nn.Sequential(
            self.conv_block(in_chan, out_chan, 3, stride=stride, bias=False),
            self.conv_block(out_chan, out_chan, 3, last=True)
        )


class BottleneckBlock(ResidualBlock):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__(in_chan, out_chan, stride)

        self.blocks = nn.Sequential(
            self.conv_block(in_chan, in_chan, 1, bias=False),
            self.conv_block(in_chan, in_chan, 3, stride=stride, padding=1, bias=False), # Downsample here as per line 107 https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
            self.conv_block(in_chan, out_chan, 1, last=True, bias=False)
        )


class ResNet(nn.Module):
    def __init__(self, c):
        super(ResNet, self).__init__()
        self.in_chan = c.layer_channels[0]

        # Feature extractor layer
        self.conv_block = nn.Sequential(
            nn.Conv1d(1, self.in_chan, c.kernel, padding=c.padding, stride=c.stride),
            nn.BatchNorm1d(self.in_chan),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=1, stride=2)
        )

        # Residual layers
        block = BottleneckBlock if c.block == 'bottleneck' else BasicBlock
        layers = []
        for i in range(c.n_layers):
            if i == 0:
                layers.append(self._make_layer(block, c.layer_channels[i], c.layer_blocks[i]))
            else:
                layers.append(self._make_layer(block, c.layer_channels[i], c.layer_blocks[i], stride=2))
        self.layers = nn.ModuleList(layers)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.decoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(c.layer_channels[-1], 2) # TODO: Parameterise output_size
        )

        # Initialise weights and biases
        self._init_weights()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = self.decoder(x)

        return x

    def _make_layer(self, block, channels, blocks, stride=1):
        # First residual block in layer may downsample
        layers = [block(self.in_chan, channels, stride)]
        # TODO: Rename layers to blocks

        # In channels for next layer will be this layer's out channels
        self.in_chan = channels

        # Remaining residual blocks in layer
        for _ in range(1, blocks):
            layers.append(block(self.in_chan, channels))

        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def main():
    config = get_config('config.yaml')

    assert config.n_layers == len(config.layer_blocks)
    assert config.n_layers == len(config.layer_channels)

    model = ResNet(config)
    summary(model, input_size=(64, 9036))


if __name__ == "__main__":
    main()
