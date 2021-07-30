#-*- codeing = utf-8 -*-
#@Time :2021/7/28 20:22
#@Author :Onion
#@File :test.py
#@Software :PyCharm
import os
import glob as glob
import torch
from model.CycleGAN import CycleGAN
# 测试集的路径
from main import args
from dataloader import MusicDataSet, ToTensor
from torch.utils.data import DataLoader
from utils import save_midis
# 测试CycleGAN

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
    Mixed_dir_list=mixed_all_dir,
    transform=ToTensor()
)

music_dataloader = DataLoader(
    dataset=music_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# 加载模型
model = CycleGAN(args=args)
model_info = torch.load(os.path.join(args.checkpoint_dir, '{}2{}checkpoint.pth.tar'.format(args.domainA, args.domainB)))
model.load_state_dict(model_info['state_dict'])

model.to(device=device)

model.eval()

print('--------start testing--------')

for idx, data in enumerate(music_dataloader):

    sample = model(data)

    # 转换为numpy格式保存
    real_A = sample[0].permute(0, 2, 3, 1).detach().cpu().numpy()
    fake_B = sample[1].permute(0, 2, 3, 1).detach().cpu().numpy()
    cycle_A = sample[2].permute(0, 2, 3, 1).detach().cpu().numpy()

    name = test_A_files[idx].split('/')[-1].split('.')[0]

    save_midis(real_A, args.transfer_B_dir + name + '_origin.mid')
    save_midis(fake_B, args.transfer_B_dir + name + '_transfer.mid')
    save_midis(cycle_A, args.transfer_B_dir + name + '_cycle.mid')


# 测试Classifier