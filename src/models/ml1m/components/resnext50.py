import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np

resnext50 = models.resnext50_32x4d(weights=models.ResNeXt50_32X4D_Weights.DEFAULT)

class Resnext50(nn.Module):
    def __init__(self, resnext=resnext50, n_classes=18):
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
    model = Resnext50(n_classes = 18)
    print(model(x).shape)
    print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    print(model(x).max())