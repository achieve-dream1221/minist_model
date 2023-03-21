#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 14:44
# @Author  : achieve_dream
# @File    : model.py
# @Software: Pycharm
import torch
import torch.nn as nn


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc3 = nn.Linear(in_features=16 * 6 * 6, out_features=120)
        self.fc4 = nn.Linear(in_features=120, out_features=84)
        self.fc5 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)


if __name__ == '__main__':
    model = LeNet()
    ret = model(torch.randn(1, 1, 32, 32))
    print(ret.shape)
    print(ret)
