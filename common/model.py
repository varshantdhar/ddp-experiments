# Placeholder for shared CNN model definition 

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # First convolution: outputs 16 feature maps of size 32x32 (input is 3x32x32)
        # ReLU applies non-linearity
        # MaxPool2d with kernel_size=2, stride=2 reduces each spatial dimension by half:
        #   - For input of size [B, 16, 32, 32], output is [B, 16, 16, 16]
        #   - It does this by taking the maximum value in each 2x2 window, effectively downsampling the feature map
        x = self.pool(F.relu(self.conv1(x)))  # [B, 16, 16, 16]
        # Second convolution and pooling: further extract features and downsample
        x = self.pool(F.relu(self.conv2(x)))  # [B, 32, 8, 8]
        # Flatten the output from [B, 32, 8, 8] to [B, 32*8*8] so it can be fed into the fully connected layer
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x 