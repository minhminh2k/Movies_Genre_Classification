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
)
import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np
from torchvision import transforms

class CNNModel(nn.Module):
    def __init__(self, n_classes=18, len_vocab=3072, embedding_dimension=3898, hidden_size=64, n_length=4):
        super(CNNModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Image
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(64 * 64 * 64, 512)
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(512, hidden_size * 2)
        
        self.sigm = nn.Sigmoid()
        
        self.flatten = nn.Flatten()

        # Text
        self.fc3 = nn.Linear(n_length * len_vocab, 512)
        self.fc4 = nn.Linear(512, hidden_size)
        # Combine
        self.fc5 = nn.Linear(hidden_size * 3, n_classes)

    def forward(self, text_tens, img_tens):
        text_feat = self.fc3(self.flatten(text_tens))
        text_feat = self.fc4(text_feat)
        
        img_feat = self.relu1(self.conv1(img_tens))
        img_feat = self.relu2(self.conv2(img_feat))
        img_feat = self.dropout(self.max_pool1(img_feat))
        
        img_feat = self.relu3(self.conv3(img_feat))
        img_feat = self.relu4(self.conv4(img_feat))
        img_feat = self.dropout(self.max_pool2(img_feat))
        
        img_feat = self.dropout(self.relu5(self.fc1(self.flatten(img_feat))))
        img_feat = self.fc2(img_feat)
        
        out = self.fc5(torch.concat([text_feat, img_feat], dim=1))
        return self.sigm(out)
    
if __name__ == "__main__":
    y = torch.rand((1, 3, 256, 256))
    x = torch.rand((1, 4, 3072))
    model = CNNModel(n_classes = 18, len_vocab=3072, embedding_dimension=3898, hidden_size=64, n_length=4)
    print(model(x, y).shape) # torch.Size([1, 18])
    print(model(x, y).min())
    print(model(x, y).max())
    print(model(x, y))
    
    