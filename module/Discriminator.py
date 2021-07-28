#-*- codeing = utf-8 -*-
#@Time :2021/7/21 3:00
#@Author :Onion
#@File :Discriminator.py
#@Software :PyCharm
import torch.nn as nn
from ..module.DiscriminatorNode import DiscriminatorNode
class Discriminator(nn.Module):
    def __init__(self,out_dim=64):
        super(Discriminator,self).__init__()
        self.Discriminator = DiscriminatorNode(out_dim)

    def forward(self,x):
        x = self.Discriminator(x)
        return x