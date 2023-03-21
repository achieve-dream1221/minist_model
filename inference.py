#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 15:16
# @Author  : achieve_dream
# @File    : inference.py
# @Software: Pycharm
import torch
import torch.nn as nn
from model import LeNet
from data import data_test_loader

if __name__ == '__main__':
    # 加载训练好的模型
    save_info = torch.load(f="model.pth")
    # 实例化模型
    model = LeNet()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 加载模型的参数
    model.load_state_dict(save_info['model'])
    # 切换为测试状态
    model.eval()

    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():  # 关闭计算题
        for batch_idx, (inputs, targets) in enumerate(data_test_loader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx,
                  len(data_test_loader),
                  f"Loss: {test_loss / (batch_idx + 1):.3f} | Acc: {100. * correct / total:.3f}%({correct}/{total})")
