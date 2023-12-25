from torchvision.models import (
    ResNet34_Weights,
    ResNet101_Weights,
    ResNet50_Weights,
    resnet34,
    resnet50,
    resnet101,
)
import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np

res50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

class Resnet50(nn.Module):
    def __init__(self, resnext=res50, n_classes=18):
        super().__init__()
        resnet = resnext
        resnet.fc = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
    
        
if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = Resnet50(n_classes = 18)
    print(model(x).shape)
    print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    print(model(x).max())