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
from transformers import AutoModelForSequenceClassification

class CombineModel_Bert(nn.Module):
    def __init__(self, n_classes, hidden_size=64, n_length=4, arch="resnet101"):
        super(CombineModel_Bert, self).__init__()
        self.hidden_size = hidden_size
        
        # pretrained architecture
        self.arch = arch
        
        if self.arch == "resnet34":
            self.rn = resnet34(weights=ResNet34_Weights.DEFAULT)
            print("Using Resnet34")
        elif self.arch == "resnet101":
            self.rn = resnet101(weights=ResNet101_Weights.DEFAULT)
            print("Using Resnet101")
        elif self.arch == "resnet50":
            self.rn = resnet50(weights=ResNet50_Weights.DEFAULT)
            print("Using Resnet50")
        
        # Freeze
        for param in self.rn.parameters():
            param.requires_grad = False

        # Unfreeze
        for param in self.rn.fc.parameters():
            param.requires_grad = True
            
        # Pretrained title model
        self.bert = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=hidden_size)   
        
        # Freeze 
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze
        for param in self.bert.classifier.parameters():
            param.requires_grad = True 
        
        self.last_layer = torch.nn.Linear(1000, hidden_size)
        
        self.sigm = nn.Sigmoid()
        
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, text_tens, img_tens):
        text_feat = self.bert(**text_tens).logits
        
        img_feat = self.rn(img_tens)
        img_feat = self.last_layer(img_feat)
        
        out = self.fc(torch.concat([text_feat, img_feat], dim=1))
        return self.sigm(out)
    
if __name__ == "__main__":
    pass
    