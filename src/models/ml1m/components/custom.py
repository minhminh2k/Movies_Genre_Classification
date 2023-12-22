import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self, n_classes=18):
        super(CustomModel, self).__init__()

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
        self.fc2 = nn.Linear(512, 128)
        
        self.sigm = nn.Sigmoid()
        
        self.flatten = nn.Flatten()

        self.fc3 = nn.Linear(128, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout(self.max_pool1(self.relu2(self.conv2(self.relu1(self.conv1(x))))))
        x = self.dropout(self.max_pool2(self.relu4(self.conv4(self.relu3(self.conv3(x))))))
        x = self.dropout(self.relu5(self.fc1(self.flatten(x))))
        x = self.fc3(self.fc2(x))
        x = self.sigmoid(x)
        return x

if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    model = CustomModel(n_classes = 18)
    print(model(x).shape) # [1, 18]
    print(model(x).max()) # [1, 18]
    print(model(x).min()) # [1, 18]