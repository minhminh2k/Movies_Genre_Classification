from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet34,
    resnet50,
    resnet101,
)
import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np
from torchvision import transforms

RNet101 = resnet101(weights=ResNet101_Weights.DEFAULT)


class Resnet101(nn.Module):
    def __init__(self, resnet=RNet101, n_classes=18):
        super().__init__()
        self.resnet = resnet
        self.last_layer = torch.nn.Linear(1000, n_classes)
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        x = self.resnet(x)
        return self.sigm(self.last_layer(x))
    
        
if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = Resnet101(n_classes = 18)
    print(model(x).shape)
    print(model(x).shape)  # 'torch.Size([1, 18])
    print(model(x).max())