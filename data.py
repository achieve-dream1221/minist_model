#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 12:16
# @Author  : achieve_dream
# @File    : data.py
# @Software: Pycharm
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

__all__ = [
    "data_train_loader",
    "data_test_loader"
]
train_batch_size = 64  # 指定DataLoader在训练集中每批加载的样本数量
test_batch_size = 128  # 指定DataLoader在测试集中每批加载的样本数量
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.1307], [0.3081])
])

data_train = MNIST("data", download=True, train=True, transform=transform)

data_test = MNIST("data", download=True, train=False, transform=transform)

data_train_loader = DataLoader(data_train, batch_size=train_batch_size, shuffle=True)
data_test_loader = DataLoader(data_test, batch_size=test_batch_size, shuffle=False)

# import matplotlib.pyplot as plt

# figure = plt.figure()
# num_of_images = 60
#
# for imgs, targets in data_train_loader:
#     break
#
# for index in range(num_of_images):
#     plt.subplot(6, 10, index + 1)
#     plt.axis('off')
#     img = imgs[index, ...]
#     plt.imshow(img.numpy().squeeze(), cmap='gray_r')
# plt.show()
