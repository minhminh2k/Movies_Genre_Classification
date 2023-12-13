from torchvision.models import (
    VGG16_Weights,
    vgg16,
)
import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np
from torchvision import transforms

V16 = vgg16(weights=VGG16_Weights.DEFAULT)


class VGG16(nn.Module):
    def __init__(self, vgg=V16, n_classes=18):
        super().__init__()
        resnet = vgg
        resnet.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=1000, out_features=n_classes, bias=True)
        )
        # self.last_layer = torch.nn.Linear(1000, n_classes)
        self.base_model = resnet
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        return self.sigm(self.base_model(x))
    
        
if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = VGG16(n_classes = 18)
    print(model(x).shape)
    print(model(x).shape)  # 'torch.Size([1, 18])
    print(model(x).max())