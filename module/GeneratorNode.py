# -*- codeing = utf-8 -*-
# @Time :2021/7/21 3:00
# @Author :Onion
# @File :GeneratorNode.py
# @Software :PyCharm
import torch.nn as nn
from utils import kernel_initializer

class EncorderNode(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super(EncorderNode, self).__init__()
        self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
        # affine： 布尔值，当设为true，给该层添加可学习的仿射变换参数
        # 在CycleGAN的代码里有的结果乘以scale后加上了offset说明是添加了仿射参数的
        self.instancenorm = nn.InstanceNorm2d(out_dim, affine=True)
        # inplace设置为true表示覆盖之前的值，用来减少内存消耗
        self.relu = nn.ReLU(inplace=True)
        self.conv2d.weight = kernel_initializer(self.conv2d.weight)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        return x


class DecorderNode(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding):
        super(DecorderNode, self).__init__()
        self.transconv2d = nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride, padding, bias=False)
        self.instancenorm = nn.InstanceNorm2d(out_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.transconv2d.weight = kernel_initializer(self.transconv2d.weight)

    def forward(self, x):
        x = self.transconv2d(x)
        x = self.instancenorm(x)
        x = self.relu(x)
        return x
