#-*- codeing = utf-8 -*-
#@Time :2021/7/30 16:07
#@Author :Onion
#@File :famous_songs.py
#@Software :PyCharm
import os
import glob as glob
import torch
import argparse
import numpy as np
from model.CycleGAN import CycleGAN
# 测试集的路径
from dataloader.dataloader import MusicDataSet,ToTensor
from torch.utils.data import DataLoader
from utils import save_midis,to_binary
from model.Generator import Generator
# 定义参数(参考MuseGAN模型)这里为了方便直接把CycleGAN.py里的参数复制了过来,并不是都需要用到
parser = argparse.ArgumentParser(description='Parameters')
parser.add_argument('--domainA', dest='domainA', default='Classical', help='Genre A')
parser.add_argument('--domainB', dest='domainB', default='Jazz', help='Genre B')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=5, help='music in batch')
parser.add_argument('--epochs', dest='epochs', type=int, default=50, help='number of epoch')
parser.add_argument('--pitch_range', dest='pitch_range', type=int, default=84, help='pitch range of pianoroll')
parser.add_argument('--time_step', dest='time_step', type=int, default=64, help='time steps of pianoroll')
parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='initial learning rate for adam')
# parser.add_argument('--which_direction', dest='which_direction', default='AtoB', help='AtoB or BtoA')
parser.add_argument('--mode', dest='mode', default='train', help='train, test')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./samples', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./datasets/test', help='Genre A that want to transfer to B')
parser.add_argument('--transfer_B_dir', dest='transfer_B_dir',default='./test', help='Genre B that transfer from Genre B')
parser.add_argument('--model_dir', dest='model_dir', default='./checkpoint', help='saved_model that need to be loded')
parser.add_argument('--log_dir', dest='log_dir', default='./log', help='logs are saved here')
# parser.add_argument('--module', dest='module', default='full', help='three different models, base, partial, full')
parser.add_argument('--type', dest='type', default='cyclegan', help='cyclegan or classifier')
parser.add_argument('--transfer', dest='transfer', default='./transfer', help='put the music that want to transfer')
parser.add_argument('--tempo', dest='tempo', type=int, default=100, help='tempo is the speed at each time step')
parser.add_argument('--number_of_track', dest='number_of_track', type=int, default=5, help='number of tracks')
parser.add_argument('--beat', dest='beat', type=int, default=4, help='temporal resolution to multitrack')
parser.add_argument('--sigma', dest='sigma', type=float, default=0.01,help='sigma of gaussian noise of classifier and discriminator')
parser.add_argument('--gamma', dest='gamma', type=float, default=1.0, help='weight of extra discriminators')
parser.add_argument('--lamb', dest='lamb', type=float, default=10.0, help='weight on L1 term in objective')
parser.add_argument('--sample_size', dest='sample_size', type=int, default=50, help='max size of sample pool, 0 means do not use sample pool')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input data channels')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output data channels')
# 初始化参数
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

song_npy_name = 'YMCA.npy'
song_name = 'YMCA'
song = np.load('./datasets/famous_songs/{}'.format(song_npy_name)).transpose((2, 0, 1))

song = torch.tensor(song)

# 加载模型
print('==> loading existing model')
model = CycleGAN(args)
model_info = torch.load(os.path.join(args.checkpoint_dir, '{}2{}checkpoint.pth.tar'.format(args.domainA, args.domainB)))
model.load_state_dict(model_info['state_dict'])
model.to(device)
model.eval()

model.test()
print('==>Succeed loaded existing model')

print("Trasnfering song from {} to {}".format(args.domainA,args.domainB))

transfer_song = model.generatorA2B(song).detach().cpu().numpy()

transfer_song = to_binary(transfer_song, 0.5).transpose((0, 2, 3, 1))

save_midis(transfer_song, './transfer/{}.mid'.format(song_name))


