#-*- codeing = utf-8 -*-
#@Time :2021/7/21 3:04
#@Author :Onion
#@File :ResNetBlock.py
#@Software :PyCharm
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import kernel_initializer
from utils import padding

class ResNetBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1):
        super(ResNetBlock, self).__init__()
        self.dim = dim
        self.ks = kernel_size
        self.s = stride
        self.p = (kernel_size - 1) // 2
        self.Conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, bias=False),
            nn.InstanceNorm2d(dim, affine=True),
            nn.ReLU(inplace=True)
        )
        self.Conv1[0].weight = kernel_initializer(self.Conv1[0].weight)

        self.Conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, stride, bias=False),
            nn.InstanceNorm2d(dim, affine=True)
        )
        self.Conv2[0].weight = kernel_initializer(self.Conv2[0].weight)

    def __call__(self, x):
        y = padding(x, self.p)
        # (bs,256,18,23)
        y = self.Conv1(y)
        # (bs,256,16,21)
        y = padding(y, self.p)
        # (bs,256,18,23)
        y = self.Conv2(y)
        # (vs,256,16,21)
        out = torch.relu(x + y)
        return out
