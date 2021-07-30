#-*- codeing = utf-8 -*-
#@Time :2021/7/21 3:00
#@Author :Onion
#@File :Generator.py
#@Software :PyCharm
import torch.nn as nn
import torch
from model.GeneratorNode import EncorderNode,DecorderNode
from model.ResNetBlock import ResNetBlock
from utils import kernel_initializer
from utils import padding

class Generator(nn.Module):
    def __init__(self, out_dim=64):
        super(Generator, self).__init__()
        self.Encorder = nn.Sequential(
            EncorderNode(in_dim=1, out_dim=out_dim, kernel_size=7, stride=1, padding=0),
            EncorderNode(in_dim=out_dim, out_dim=2 * out_dim, kernel_size=3, stride=2, padding=1),
            EncorderNode(in_dim=2 * out_dim, out_dim=4 * out_dim, kernel_size=3, stride=2, padding=1),
        )
        self.ResNet = nn.ModuleList(
            [ResNetBlock(4 * out_dim, 3, 1) for i in range(10)])


        self.Decorder = nn.Sequential(
            DecorderNode(in_dim=4 * out_dim, out_dim=2 * out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
            DecorderNode(in_dim=2 * out_dim, out_dim=out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        self.output = nn.Conv2d(in_channels=out_dim,out_channels=1,kernel_size=7,stride=1,bias=False)

        self.output.weight = kernel_initializer(self.output.weight)

    def forward(self, x):
        x = padding(x)
        x = self.Encorder(x)
        for i in range(10):
            x = self.ResNet[i](x)
        x = self.Decorder(x)
        x = padding(x)
        x = self.output(x)
        return torch.sigmoid(x)