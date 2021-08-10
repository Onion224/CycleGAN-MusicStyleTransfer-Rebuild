#-*- codeing = utf-8 -*-
#@Time :2021/7/28 20:22
#@Author :Onion
#@File :test.py
#@Software :PyCharm
import os
import glob as glob
import torch
import argparse
from model.CycleGAN import CycleGAN
# 测试集的路径
from dataloader.dataloader import MusicDataSet,ToTensor
from torch.utils.data import DataLoader
from utils import save_midis,to_binary
from model.Generator import Generator
# 测试CycleGAN

# 定义参数(参考MuseGAN模型)
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

# transfer_B_dir用来存放A转换为B之后的mid文件
if not os.path.exists(args.transfer_B_dir):
    os.makedirs(args.transfer_B_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_A_files = glob.glob('{}/{}/*.*'.format(args.test_dir, args.domainA))
test_A_len = len(test_A_files)

print('--------------------')
print('Test:', test_A_len)
print('--------------------')

mixed_all_dir = os.path.join(os.getcwd(), 'datasets', 'MixedAll' + os.sep)

music_dataset = MusicDataSet(
    A_dir_list=test_A_files,
    B_dir_list=None,
    # 测试模式下mixed_dir使用test_A_files
    Mixed_dir_list=test_A_files,
    transform=ToTensor()
)

music_dataloader = DataLoader(
    dataset=music_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# 加载模型
model = CycleGAN(args)
model_info = torch.load(os.path.join(args.checkpoint_dir, '{}2{}checkpoint.pth.tar'.format(args.domainA, args.domainB)))
model.load_state_dict(model_info['state_dict'])
model.to(device)
model.eval()

# netG_A2B = Generator().to(device)
# netG_B2A = Generator().to(device)
# netG_A2B.load_state_dict(model_info['GeneratorA2B'])
# netG_B2A.load_state_dict(model_info['GeneratorB2A'])
#
# netG_A2B.eval()
# netG_B2A.eval()

print('--------start testing--------')

for idx, data in enumerate(music_dataloader):

    real_A = data['bar_A']
    # # 将numpy类型转为tensor
    # # (batch_size * input_nc * time_step * pitch_range)

    real_A = torch.FloatTensor(real_A).to(device)
    # fake_B = netG_A2B(real_A)
    # cycle_A = netG_B2A(fake_B)
    fake_B = model.generatorA2B(real_A)
    cycle_A = model.generatorB2A(fake_B)

    data_sample = [real_A, fake_B, cycle_A]
    for i in range(len(data_sample)):
        data_sample[i] = data_sample[i].detach().cpu().numpy()

    samples = [to_binary(data_sample[0],0.5), to_binary(data_sample[1],0.5), to_binary(data_sample[2],0.5)]

    # 转换为numpy格式保存
    real_A = samples[0].transpose((0, 2, 3, 1))
    fake_B = samples[1].transpose((0, 2, 3, 1))
    cycle_A = samples[2].transpose((0, 2, 3, 1))

    name = test_A_files[idx].split('/')[-1].split('.')[0]

    transfer_B_dir = os.path.join(args.transfer_B_dir)

    save_midis(real_A, './{}/{}_origin.mid'.format(args.transfer_B_dir, name))
    save_midis(fake_B, './{}/{}_transfer.mid'.format(args.transfer_B_dir, name))
    save_midis(cycle_A, './{}/{}_cycle.mid'.format(args.transfer_B_dir, name))


# 测试Classifier