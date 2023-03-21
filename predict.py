#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/20 15:59
# @Author  : achieve_dream
# @File    : predict.py
# @Software: Pycharm
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import torch
from model import LeNet

if __name__ == '__main__':
    # 加载训练好的模型
    save_info = torch.load(f="model2.pth")
    # 实例化模型
    model = LeNet()
    # 损失函数
    # 加载模型的参数
    model.load_state_dict(save_info['model'])
    # 切换为测试状态
    model.eval()
    # 转换图片
    image = Image.open('data/img.png').convert('L')
    image = ImageOps.invert(image)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    transforms.ToPILImage()(image.squeeze(0)).save('7.png')
    with torch.no_grad():
        output = model(image)
        # print(output.squeeze(0))
        prediction = output.argmax(dim=1, keepdim=True)
        print("预测结果:", prediction.item() + 1)
