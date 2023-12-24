from torchvision.models import (
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNeXt50_32X4D_Weights,
    VGG16_Weights,
    resnet34,
    resnet50,
    resnet101,
    resnext50_32x4d,
    vgg16,
    VisionTransformer,
    ViT_L_32_Weights,
    vit_l_32,
    MobileNet_V2_Weights,
    mobilenet_v2,
)
import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np
from torchvision import transforms

class CombineModel(nn.Module):
    def __init__(self, n_classes, len_vocab=3072, embedding_dimension=3898, hidden_size=64, n_length=4, arch="resnet101"):
        super(CombineModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.arch = arch
        # image
        '''
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 64 * 64, hidden_size)

        '''
        # Pretrained for image
        if self.arch == "resnet34":
            self.rn = resnet34(weights=ResNet34_Weights.DEFAULT)
            print("Using Resnet34")
        elif self.arch == "resnet101":
            self.rn = resnet101(weights=ResNet101_Weights.DEFAULT)
            print("Using Resnet101")
        elif self.arch == "resnext50":
            self.rn = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            print("Using Resnext50")
        elif self.arch == "vgg16":
            self.rn = vgg16(weights=VGG16_Weights.DEFAULT)
            print("Using VGG16")
        elif self.arch == "vit":
            self.rn = vit_l_32(weights=ViT_L_32_Weights.DEFAULT)
            print("Using Vision Transformer")
        elif self.arch == "mobile":
            self.rn = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            print("Using Mobile Net")
            
        self.last_layer = torch.nn.Linear(1000, hidden_size * 2)
        
        self.sigm = nn.Sigmoid()
        
        self.flatten = nn.Flatten()

        # Text
        self.fc2 = nn.Linear(n_length * len_vocab, hidden_size)

        # Combine
        self.fc3 = nn.Linear(hidden_size * 3, n_classes)

    def forward(self, text_tens, img_tens):
        text_feat = self.fc2(self.flatten(text_tens))
        '''
        img_feat = self.conv1(img_tensor)
        img_feat = self.maxpool(img_feat)
        img_feat = self.conv2(img_feat)
        img_feat = self.fc1(self.flatten(img_feat))
        '''
        img_feat = self.rn(img_tens)
        img_feat = self.last_layer(img_feat)
        
        out = self.fc3(torch.concat([text_feat, img_feat], dim=1))
        return self.sigm(out)
    
if __name__ == "__main__":
    y = torch.rand((1, 3, 256, 256))
    x = torch.rand((1, 4, 3072))
    model = CombineModel(n_classes = 18, len_vocab=3072, embedding_dimension=3898, hidden_size=64, n_length=4, arch="resnet101")
    print(model(x, y).shape) # torch.Size([1, 18])
    print(model(x, y).min())
    print(model(x, y).max())
    print(model(x, y))
    
    