#-*- codeing = utf-8 -*-
#@Time :2021/7/26 1:08
#@Author :Onion
#@File :losses.py
#@Software :PyCharm
# 如果使用损失函数训练生成网络（和Alpha-GAN网络中的编码器），那么，应该使用哪种损失函数来训练判别器呢？
# 判别器的任务是区分实际数据分布和生成数据分布，使用监督的方式训练判别器比较容易，如二元交叉熵。由于判别器是生成器的损失韩式，这就意味着，判别器的二进制交叉熵损失函数产生的梯度也可以用来更新生成器。
# 结论
# CycleGAN风格迁移本质上属于回归问题,因此选用的函数一般基于均方差损失函数(MSE)
import torch
from torch.nn import functional as F


# L1损失函数(MAE):torch.nn.L1Loss()
def mae_criterion(pred, target):
    criterion = torch.nn.L1Loss(reduction='mean')
    return criterion(pred, target)

# L1损失函数(MAE):F.l1_loss(pred,target),cycle_loss则是将两个相加
def cycle_loss(real_a, cycle_a, real_b, cycle_b):

    l1_loss = torch.nn.L1Loss()

    return l1_loss(cycle_a, real_a) + l1_loss(cycle_b, real_b)


