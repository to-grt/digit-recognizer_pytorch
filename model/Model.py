import torch
import torch.nn as nn
import torch.nn.functional as f


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Conv2d(1, 64, (3, 3), padding='same')
        self.fc2 = nn.Conv2d(64, 64, (3, 3), padding='same')
        self.maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.fc3 = nn.Conv2d(64, 32, (3, 3), padding='same')
        self.fc4 = nn.Conv2d(32, 32, (3, 3), padding='same')
        self.maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.fc5 = nn.Conv2d(32, 16, (3, 3), padding='same')
        self.fc6 = nn.Conv2d(16, 16, (3, 3), padding='same')
        self.maxpool3 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.linear1 = nn.Linear(in_features=16*3*3, out_features=10)

        self.flatten = nn.Flatten()

    def forward(self, x):

        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = f.relu(self.fc5(x))
        x = f.relu(self.fc6(x))
        x = self.maxpool3(x)
        x = self.dropout3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        return x
