# Originally from model.py from Contrastive Learning Framework

import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution, MaxAvgPool
from monai.networks.nets import ResNet

# Model that will be trained, including a Convolutional Neural Network(backbone) and an MLP Projection Head

class CompleteNet(nn.Module):
    def __init__(self, backbone):
        super(CompleteNet, self).__init__()
        self.backbone = backbone # This is the CNN/Resnet
        self.fc1 = nn.Linear(64, 80) # Converting into linear layer
        self.fc2 = nn.Linear(80, 40)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv1 = Convolution(
                spatial_dims=3,
                in_channels=1,
                out_channels=64,
                kernel_size = (5,5,5),
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                dropout=0.1
                )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = Convolution(
                spatial_dims=3,
                in_channels=64,
                out_channels=64,
                kernel_size = (5,5,5),
                adn_ordering="NDA",
                act=("prelu", {"init": 0.2}),
                dropout=0.1
                )
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.global_pool(x)
        return x

# resnet = ResNet(block='basic', layers=18, block_inplanes=1, spatial_dims=3, n_input_channels=1, num_classes=64)
