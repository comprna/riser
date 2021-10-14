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

        self.in_chan = in_chan
        self.out_chan = out_chan
        self.stride = stride # TODO: Rename downsampling

        self.blocks = nn.Sequential(
            conv_block(in_chan, in_chan, 1, bias=False),
            conv_block(in_chan, in_chan, 3, stride=stride, padding=1, bias=False),
            conv_block(in_chan, out_chan, 1, last=True, bias=False)
        )
        self.activate = nn.ReLU(inplace=True)
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
    def __init__(self, block, layers, num_classes=2):
        super(ResNet, self).__init__()
        self.chan1 = 20

        self.conv_block = nn.Sequential(
            nn.Conv1d(1, 20, 19, padding=5, stride=3),
            nn.BatchNorm1d(self.chan1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, padding=1, stride=2)
        )

        self.layer1 = self._make_layer(block, 20, layers[0])
        self.layer2 = self._make_layer(block, 30, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 45, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 67, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.decoder = nn.Sequential(
            nn.Flatten(1),   # TODO: This should match torch.flatten(x, 1)
            nn.Linear(67, 2)
        )

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(self, block, channels, blocks, stride=1):
        layers = []
        layers.append(block(self.chan1, channels, stride))
        if stride != 1 or self.chan1 != channels:
          self.chan1 = channels # TODO: Rename or remove self.chan1
        for _ in range(1, blocks):
            layers.append(block(self.chan1, channels))

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


# class ResNet1D(nn.Module):
#     def __init__(self):
#         super().__init__()

        # Initial layer

            # 20 channels


        # 4 residual layers

            # Each layer has 2 Bottleneck units

                # 1 x 1 conv
                # BN
                # ReLU
                # 3 x 3 conv
                # BN
                # ReLU
                # 1 x 1 conv
                # BN

                # Stride of 2

                # Sum output and input of unit
            
            # Layer 1: 20 channels
            # Layer 2: 30 channels
            # Layer 3: 45 channels
            # Layer 4: 67 channels
        

        # Fully connected layer + softmax

            # Mean pooling

            # Fully connected layer

            # Softmax activation

    # def forward(self, x):
    #     # TODO


if __name__ == "__main__":
    model = ResNet(BottleneckBlock, [2,2,2,2])
    summary(model, input_size=(64, 9036))