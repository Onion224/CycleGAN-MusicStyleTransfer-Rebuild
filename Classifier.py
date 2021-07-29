#-*- codeing = utf-8 -*-
#@Time :2021/7/21 3:04
#@Author :Onion
#@File :Classifier.py
#@Software :PyCharm
import torch.nn as nn
from utils import kernel_initializer

class Classifier(nn.Module):
    def __init__(self, args, dim=64):
        super(Classifier, self).__init__()
        self.cla = nn.Sequential(
            nn.Conv2d(1, dim, (1, 12), (1, 12), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,64,64,7)
            nn.Conv2d(dim, 2 * dim, (4, 1), (4, 1), bias=False),
            nn.InstanceNorm2d(2 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,128,16,7)
            nn.Conv2d(2 * dim, 4 * dim, (2, 1), (2, 1), bias=False),
            nn.InstanceNorm2d(4 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,256,8,7)
            nn.Conv2d(4 * dim, 8 * dim, (8, 1), (8, 1), bias=False),
            nn.InstanceNorm2d(8 * dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # (bs,512,1,7)
            nn.Conv2d(8 * dim, 2, (1, 7), (1, 7), bias=False)
            # (bs,2,1,1)
        )
        self.softmax = nn.Softmax(dim=1)

        for i in [0, 2, 5, 8, 11]:
            self.cla[i].weight = kernel_initializer(self.cla[i].weight)
        for i in [3, 6, 9]:
            self.cla[i].weigth = kernel_initializer(
                self.cla[i].weight, 1., 0.02)

    def forward(self, x):
        x = self.cla(x)
        # x.squeeze(-1).squeeze(-1)
        x = self.softmax(x)
        return x.squeeze(-1).squeeze(-1)

