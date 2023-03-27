import torch
from torch import nn
import torch.nn.functional as F
import torchvision

# basic resdidual block of ResNet
# This is generic in the sense, it could be used for downsampling of features.
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=[1, 1], padding=['same', 'same'], kernel_size=[3, 3], bias=[False, False], downsample=None):
        """
        A basic residual block of ResNet
        Parameters
        ----------
            in_channels: Number of channels that the input have
            out_channels: Number of channels that the output have
            stride: strides in convolutional layers
            downsample: A callable to be applied before addition of residual mapping
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size[0], stride=stride[0], 
            padding=padding[0], bias=bias[0]
        )

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size[1], stride=stride[1], 
            padding=padding[1], bias=bias[1]
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        # Apply downsample function if defined
        if(self.downsample is not None):
            residual = downsample(residual)

        out = F.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))
        
        # Skip connection
        out = out + residual
        out = F.relu(out)
        return out