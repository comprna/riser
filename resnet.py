import torch
from torch import nn
from torchinfo import summary


class BottleneckBlock(nn.Module):
	expansion = 1.5
	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super().__init__()
		self.conv_block1 = nn.Sequential(
			nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
			nn.BatchNorm1d(in_channels),
			nn.ReLU(inplace=True)  # TODO: is inplace necessary?
		)

		self.conv_block2 = nn.Sequential(
			nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=False),
			nn.BatchNorm1d(in_channels),
			nn.ReLU(inplace=True)  # TODO: is inplace necessary?
		)

		self.conv_block3 = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
			nn.BatchNorm1d(out_channels)
		)

		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		identity = x

		out = self.conv_block1(x)
		out = self.conv_block2(out)
		out = self.conv_block3(out)

		if self.downsample is not None:
			identity = self.downsample(x)

		out += identity
		out = self.relu(out)

		return out


class ResNet(nn.Module):
	def __init__(self, block, layers, num_classes=2):
		super(ResNet, self).__init__()
		self.chan1 = 20

		self.conv_block1 = nn.Sequential(
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
		downsample = None
		if stride != 1 or self.chan1 != channels: # stride != 1 means need to downsample identity, chan1 != channels means need to downsample channels of identity
			downsample = nn.Sequential(
				nn.Conv1d(self.chan1, channels, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm1d(channels)
			)

		layers = []
		layers.append(block(self.chan1, channels, stride, downsample))
		if stride != 1 or self.chan1 != channels:
		  self.chan1 = channels
		for _ in range(1, blocks):
			layers.append(block(self.chan1, channels))

		return nn.Sequential(*layers)

	def _forward_impl(self, x):
		x = x.unsqueeze(1)
		x = self.conv_block1(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = self.decoder(x)

		return x

	def forward(self, x):
	  return self._forward_impl(x)


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