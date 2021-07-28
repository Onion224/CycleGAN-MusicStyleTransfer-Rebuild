#-*- codeing = utf-8 -*-
#@Time :2021/7/21 3:00
#@Author :Onion
#@File :DiscriminatorNode.py
#@Software :PyCharm
import torch.nn as nn
from utils import kernel_initializer
class DiscriminatorNode(nn.Module):
    def __init__(self, out_dim=64):
        super(DiscriminatorNode,self).__init__()
        self.conv2D1 = nn.Conv2d(in_channels=1,out_channels=out_dim,kernel_size=7,stride=2,padding=3,bias=False)
        self.conv2D2 = nn.Conv2d(in_channels=out_dim,out_channels=4*out_dim,kernel_size=7,stride=2,padding=3,bias=False)
        self.conv2D3 = nn.Conv2d(in_channels=4*out_dim,out_channels=1,kernel_size=7,padding=3,bias=False)
        self.conv2D1.weight = kernel_initializer(self.conv2D1.weight)
        self.conv2D2.weight = kernel_initializer(self.conv2D2.weight)
        self.conv2D3.weight = kernel_initializer(self.conv2D3.weight)
        self.instancenorm = nn.InstanceNorm2d(4*out_dim,affine=True)
        self.leakyRelu = nn.LeakyReLU(negative_slope=0.2,inplace=True)

    def forward(self,x):
        x = self.conv2D1(x)
        x = self.leakyRelu(x)
        x = self.conv2D2(x)
        x = self.instancenorm(x)
        x = self.conv2D3(x)

        return x