import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):

    INPUT_SIZE = (28, 28)
    NB_CHANNELS = 1
    OUTPUT_SIZE = 10
    NB_EPOCHS = 10
    BATCH_SIZE = 8
    VALIDATION_SPLIT = 0.2

    def __init__(self, device):
        super(Model, self).__init__()

        self.relu1 = nn.ReLU()
        self.softmax1 = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(in_channels=self.NB_CHANNELS, out_channels=32,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=7*7*64, out_features=10)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

        self = self.to(torch.float64).to(device)

    def forward(self, feed):
        feed = self.conv1(feed)
        feed = self.relu1(feed)
        feed = self.conv2(feed)
        feed = self.relu1(feed)
        feed = self.pool1(feed)
        feed = self.conv3(feed)
        feed = self.relu1(feed)
        feed = self.conv4(feed)
        feed = self.relu1(feed)
        feed = self.pool2(feed)
        feed = self.flatten(feed)
        feed = self.fc1(feed)
        feed = self.softmax1(feed)
        return feed
