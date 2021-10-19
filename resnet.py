import torch
from torch import nn
from torchinfo import summary


def conv_block(in_chan, out_chan, kernel_size, last=False, **kwargs):
    layers = [
        nn.Conv1d(in_chan, out_chan, kernel_size, **kwargs),
        nn.BatchNorm1d(out_chan),
    ]
    if last == False:
        layers.append(nn.ReLU(inplace=True)) # TODO: is inplace necessary?

    return nn.Sequential(*layers)


class BottleneckBlock(nn.Module):
    expansion = 1.5 # TODO: Why is this here?
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()

        # Parameters
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.stride = stride
        self.activate = nn.ReLU(inplace=True)
    
        # Convolutional blocks
        self.blocks = nn.Sequential(
            conv_block(in_chan, in_chan, 1, bias=False),
            conv_block(in_chan, in_chan, 3, stride=stride, padding=1, bias=False),
            conv_block(in_chan, out_chan, 1, last=True, bias=False)
        )

        # Match dimensions of input and block output for summation
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_chan)
        )

    def forward(self, x):
        residual = self.shortcut(x) if self.should_apply_shortcut else x
        out = self.blocks(x)
        out += residual
        out = self.activate(out)
        return out

    @property
    def should_apply_shortcut(self):
        return self.in_chan != self.out_chan or self.stride != 1


class ResNet(nn.Module):
    def __init__(self, block, layer_sizes):
        super(ResNet, self).__init__()
        self.in_chan = 20

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, self.in_chan, 19, padding=5, stride=3),
            nn.BatchNorm1d(self.in_chan),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=1, stride=2)
        )

        self.layer1 = self._make_layer(block, 20, layer_sizes[0])
        self.layer2 = self._make_layer(block, 30, layer_sizes[1], stride=2)
        self.layer3 = self._make_layer(block, 45, layer_sizes[2], stride=2)
        self.layer4 = self._make_layer(block, 67, layer_sizes[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(67, 2)
        )

        # Initialise weights and biases
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, channels, blocks, stride=1):
        # First residual block in layer may downsample
        layers = [block(self.in_chan, channels, stride)]

        # In channels for next layer will be this layer's out channels
        self.in_chan = channels

        # Remaining residual blocks in layer
        for _ in range(1, blocks):
            layers.append(block(self.in_chan, channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    model = ResNet(BottleneckBlock, [2,2,2,2])
    summary(model, input_size=(64, 9036))