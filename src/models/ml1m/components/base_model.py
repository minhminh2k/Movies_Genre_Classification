import torch
from torch import nn
from torchvision import models
import pandas as pd
import numpy as np
from torchvision import transforms

class BaseModel(nn.Module):
    def __init__(self, n_classes=18, len_vocab =3544, embedding_dimension=3898, hidden_size=64, n_length=7):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size

        # img
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 64 * 64, hidden_size)

        self.sigm = nn.Sigmoid()
        

        # text
        self.fc2 = nn.Linear(n_length * len_vocab, hidden_size)

        self.fc3 = nn.Linear(hidden_size*2, n_classes)

    def forward(self, text_tens, img_tens):
        text_feat = self.fc2(self.flatten(text_tens))

        img_feat = self.conv1(img_tens)
        img_feat = self.maxpool(img_feat)
        img_feat = self.conv2(img_feat)
        img_feat = self.fc1(self.flatten(img_feat))

        out = self.fc3(torch.concat([text_feat, img_feat], dim=1))
        return self.sigm(out)
    
if __name__ == "__main__":
    y = torch.rand((1, 3, 256, 256))
    x = torch.rand((1, 7, 3544))
    model = BaseModel(n_classes = 18, len_vocab=3544, embedding_dimension=3898, hidden_size=64, n_length=7)
    print(model(x, y).shape) # torch.Size([1, 18])
    print(model(x, y).min())
    print(model(x, y).max())
    print(model(x, y))
    
    