from torchvision.models import (
    ResNet34_Weights,
    resnet34,
    MobileNet_V2_Weights,
    mobilenet_v2,
)
import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np
from torchvision import transforms

mobile = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)


class MobileNet(nn.Module):
    def __init__(self, resnet=mobile, n_classes=18):
        super().__init__()
        self.resnet = resnet
        self.last_layer = torch.nn.Linear(1000, n_classes)
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        x = self.resnet(x)
        return self.sigm(self.last_layer(x))
        # return self.last_layer(x)
        
        
if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = MobileNet(n_classes = 18)
    print(model(x).shape)
    print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    print(model(x).max())