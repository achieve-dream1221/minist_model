#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 14:57
# @Author  : achieve_dream
# @File    : train.py
# @Software: Pycharm
import torch
import torch.nn as nn
from model import LeNet
from data import data_train_loader

if __name__ == '__main__':
    # 定义LeNet模型
    model = LeNet()
    # 切换到训练状态
    model.train()
    # 学习率
    lr = 0.01
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    # 随机梯度下降优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for epoch in range(2):
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(data_train_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(f"epoch: {epoch + 1}/{5}", batch_idx,
                  len(data_train_loader),
                  f"Loss: {train_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}%({correct}/{total})")
    save_info = {
        # "optimizer": optimizer.state_dict(),
        "model": model.state_dict()
    }
    torch.save(save_info, "model.pth")
