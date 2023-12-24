from torchvision.models import (
    VisionTransformer,
    ViT_B_16_Weights,
    ViT_B_32_Weights,
    ViT_L_16_Weights,
    ViT_L_32_Weights,
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
)
import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np
from torchvision import transforms

ViT = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)


class VisionTrans(nn.Module):
    def __init__(self, net=ViT, n_classes=18):
        super().__init__()
        self.net = net
        self.last_layer_1 = torch.nn.Linear(1000, 256)
        self.last_layer_2 = torch.nn.Linear(256, 64)
        self.last_layer_3 = torch.nn.Linear(64, n_classes)
        self.sigm = nn.Sigmoid()
 
    def forward(self, x):
        x = self.net(x)
        x = self.last_layer_1(x)
        x = self.last_layer_2(x)
        return self.sigm(self.last_layer_3(x))
        # return self.last_layer(x)
        
        
if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    model = VisionTrans(n_classes = 18)
    print(model(x).shape)
    print(model(x).min())  # 'torch.Size([1, 1, 256, 256])
    print(model(x).max())