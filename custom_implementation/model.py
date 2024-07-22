import torch.nn as nn
import torch.nn.functional as F

# Model that will be trained, including a Convolutional Neural Network(backbone) and an MLP Projection Head

class ConvNet(nn.Module):
    def __init__(self, backbone):
        super(ConvNet, self).__init__()
        self.backbone = backbone # This is the CNN
        self.fc1 = nn.Linear(64, 80) # Converting into linear layer
        self.fc2 = nn.Linear(80, 40)
    
    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class OriginalCNN(nn.Module):
    def __init__(self):
        super(OriginalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 5) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = nn.Conv2d(64, 64, 5) 
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        return x

