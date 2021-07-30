# -*- codeing = utf-8 -*-
# @Time :2021/7/20 0:19
# @Author :Onion
# @File :CycleGAN.py
# @Software :PyCharm
from model.Generator import Generator
from model.Discriminator import Discriminator
from utils import *
from losses import cycle_loss
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CycleGAN(nn.Module):
    def __init__(self, args):
        super(CycleGAN, self).__init__()
        # super(CycleGAN, self).__init__()
        self.generatorA2B = Generator(64)
        self.generatorB2A = Generator(64)
        # 判断从从B生成的fake_A和fake_B生成回来的XA是真还是假
        self.discriminatorA = Discriminator(64)
        # 判断从从A生成的fake_B和fake_A生成回来的XB是真还是假
        self.discriminatorB = Discriminator(64)
        # 加了噪声的discriminatorA和discriminatorB
        self.discriminatorA_all = Discriminator(64)
        self.discriminatorB_all = Discriminator(64)
        self.input_nc = args.input_nc
        self.output_nc = args.output_nc
        self.time_step = args.time_step
        self.pitch_range = args.pitch_range
        self.sigma = args.sigma
        self.gamma = args.gamma
        self.lamb = args.lamb
        self.batch_size = args.batch_size
        self.sampler = Sampler(args.sample_size)
        self.args = args
        self.mse_criterion = torch.nn.MSELoss()

    def forward(self, data, optimizer_Disc, optimizer_Gen, epoch, idx, train_num, counter):

        real_A, real_B, real_mixed = data['bar_A'], data['bar_B'], data['bar_mixed']
        # 将numpy类型转为tensor
        # (batch_size * input_nc * time_step * pitch_range)
        real_A = torch.FloatTensor(real_A).to(device)
        real_B = torch.FloatTensor(real_B).to(device)
        real_mixed = torch.FloatTensor(real_mixed).to(device)

        # real_A = real_A.detach().cpu().numpy()
        # real_B = real_B.detach().cpu().numpy()

        # A2B
        fake_B = self.generatorA2B(real_A)
        cycle_A = self.generatorB2A(fake_B)

        # 测试一下
        # real_A = real_A.detach().cpu().numpy()
        # real_B = real_B.detach().cpu().numpy()

        # B2A
        fake_A = self.generatorB2A(real_B)
        cycle_B = self.generatorA2B(fake_A)

        # 测试一下
        # real_A = real_A.detach().cpu().numpy()
        # real_B = real_B.detach().cpu().numpy()

        [sample_fake_A, sample_fake_B] = self.sampler([fake_A, fake_B])
        # 这个函数(kernel_initializer)有问题,改变了real_A的值

        # gauss_noise = kernel_initializer(real_A, mean=0, std=self.sigma)

        gauss_noise = np.abs(np.random.normal(0, self.sigma, [self.batch_size, self.input_nc, self.time_step, self.pitch_range])).astype(np.float32)
        gauss_noise = torch.FloatTensor(gauss_noise).to(device)

        # 测试一下
        # real_A = real_A.detach().cpu().numpy()
        # real_B = real_B.detach().cpu().numpy()

        DA_real = self.discriminatorA(real_A + gauss_noise)
        DA_fake = self.discriminatorA(fake_A + gauss_noise)
        DB_real = self.discriminatorB(real_B + gauss_noise)
        DB_fake = self.discriminatorB(fake_B + gauss_noise)

        # 测试一下
        # real_A = real_A.detach().cpu().numpy()
        # real_B = real_B.detach().cpu().numpy()

        DA_fake_sample = self.discriminatorA(sample_fake_A + gauss_noise)
        DB_fake_sample = self.discriminatorB(sample_fake_B + gauss_noise)

        DA_real_all = self.discriminatorA_all(real_mixed + gauss_noise)
        DB_real_all = self.discriminatorB_all(real_mixed + gauss_noise)

        DA_fake_sample_all = self.discriminatorA_all(sample_fake_A + gauss_noise)
        DB_fake_sample_all = self.discriminatorB_all(sample_fake_B + gauss_noise)

        # 测试一下
        # real_A = real_A.detach().cpu().numpy()
        # real_B = real_B.detach().cpu().numpy()

        # Generator loss
        optimizer_Gen.zero_grad()

        Cycle_loss = cycle_loss(real_A, cycle_A, real_B, cycle_B)
        generator_A2B_loss = self.mse_criterion(DB_fake, torch.ones_like(DB_fake))
        generator_B2A_loss = self.mse_criterion(DA_fake, torch.ones_like(DA_fake))

        G_loss = generator_A2B_loss + generator_B2A_loss - Cycle_loss
        # 训练生成器
        # 梯度置零
        G_loss.backward(retain_graph=True)
        optimizer_Gen.step()


        # Discriminator loss
        # 梯度置零
        optimizer_Disc.zero_grad()
        DA_real_loss = self.mse_criterion(DA_real, torch.ones_like(DA_real))
        DA_fake_loss = self.mse_criterion(DA_fake_sample, torch.zeros_like(DB_fake_sample))
        DA_loss = (DA_real_loss + DA_fake_loss) / 2

        DB_real_loss = self.mse_criterion(DB_real, torch.ones_like(DB_real))
        DB_fake_loss = self.mse_criterion(DB_fake_sample, torch.zeros_like(DB_fake_sample))
        DB_loss = (DB_real_loss + DB_fake_loss) / 2

        d_loss = DA_loss + DB_loss
        # ALL
        DA_real_all_loss = self.mse_criterion(DA_real_all, torch.ones_like(DA_real_all))
        DA_fake_all_loss = self.mse_criterion(DA_fake_sample_all, torch.zeros_like(DA_fake_sample_all))
        DA_all_loss = (DA_real_all_loss + DA_fake_all_loss) / 2

        DB_real_all_loss = self.mse_criterion(DB_real_all, torch.ones_like(DB_real_all))
        DB_fake_all_loss = self.mse_criterion(DB_fake_sample, torch.zeros_like(DB_fake_sample_all))
        DB_all_loss = (DB_real_all_loss + DB_fake_all_loss) / 2

        d_all_loss = DA_all_loss + DB_all_loss

        D_loss = d_loss + self.gamma * d_all_loss
        # 训练判别器

        D_loss.backward(retain_graph=True)
        optimizer_Disc.step()


        # generate samples during training to track the learning process
        # 测试一下
        data_sample = [real_A, fake_B, cycle_A,
                       real_B, fake_A, cycle_B]

        for i in range(len(data_sample)):

            data_sample[i] = data_sample[i].detach().cpu().numpy()
        # 将训练期间产生的sample输出为mid

        samples = [to_binary(data_sample[0],0.5), to_binary(data_sample[1],0.5), to_binary(data_sample[2],0.5),
                   to_binary(data_sample[3],0.5), to_binary(data_sample[4],0.5), to_binary(data_sample[5],0.5)]

        print('=================================================================')
        print("[Epoch: %3d/%3d, batch: %5d/%5d, ite: %d] G_loss : %3f, D_loss : %3f " %
              (epoch + 1, self.args.epochs, (idx + 1) * self.args.batch_size, train_num, counter, G_loss, D_loss))

        return samples

