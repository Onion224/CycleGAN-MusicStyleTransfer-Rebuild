# -*- codeing = utf-8 -*-
# @Time :2021/7/20 0:19
# @Author :Onion
# @File :main.py
# @Software :PyCharm
import os
import argparse
import glob
import torch
from model.CycleGAN import CycleGAN
from model.Classifier import Classifier
from torch import optim
from torch.utils.data import DataLoader
from utils import sample_model, save_checkpoint
from dataloader import MusicDataSet, ToTensor

# 只训练full类型的模型,其余的没有考虑,并且只做了A2B,没有做B2A

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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 数据目录
data_dir = os.path.join(os.getcwd(), 'datasets' + os.sep)
train_A_dir = os.path.join('train', 'Classical' + os.sep)
train_B_dir = os.path.join('train', 'Jazz' + os.sep)
mixed_all_dir = os.path.join('MixedAll' + os.sep)
# checkpoint_dir = r'checkpoint/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_A_dir_name_list = glob.glob(data_dir + train_A_dir + '*.*')
train_B_dir_name_list = glob.glob(data_dir + train_B_dir + '*.*')
mixed_all_dir_name_list = glob.glob(data_dir + mixed_all_dir + '*.*')
train_num = len(train_A_dir_name_list)

print(len(train_A_dir_name_list))
print(len(train_B_dir_name_list))

# 数据集加载器初始化
music_dataset = MusicDataSet(
    A_dir_list=train_A_dir_name_list,
    B_dir_list=train_B_dir_name_list,
    Mixed_dir_list=mixed_all_dir_name_list,
    transform=ToTensor()
)

music_dataloader = DataLoader(
    dataset=music_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0
)

# 模型初始化
if args.type == 'cyclegan':
    model = CycleGAN(args=args)
else:
    model = Classifier(args=args)

# 将模型设置到GPU上训练
if torch.cuda.is_available():
    model.to(device)

# 加载已经存在的模型
if os.path.exists(os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar')):
    # load existing model
    print('==> loading existing model')
    # 加载CycleGAN模型
    if args.type == 'cyclegan':
        model_info = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth.tar'))
        model.load_state_dict(model_info['state_dict'])
        # optimizer = torch.optim.Adam(model.parameters())
        optimizer_Gen = torch.optim.Adam(list(model.generatorA2B.parameters()) + list(model.generatorB2A.parameters()))

        optimizer_Disc = torch.optim.Adam(list(model.discriminatorA.parameters())
                                          + list(model.discriminatorB.parameters())
                                          + list(model.discriminatorA_all.parameters())
                                          + list(model.discriminatorB_all.parameters()))

        optimizer_Gen.load_state_dict(model_info['optimizer_Gen'])
        optimizer_Disc.load_state_dict(model_info['optimizer_Disc'])
    # 加载classifier模型
    else:

        pass

else:
    # 初始化当前epoch
    cur_epoch = 0

    # 定义Generator的optimizer
    optimizer_Gen = optim.Adam(
        list(model.generatorA2B.parameters()) + list(model.generatorB2A.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
        eps=1e-08,
        weight_decay=0
    )
    # 定义Discriminator的optimizer
    optimizer_Disc = optim.Adam(
        list(model.discriminatorA.parameters())
        + list(model.discriminatorB.parameters())
        + list(model.discriminatorA_all.parameters())
        + list(model.discriminatorB_all.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
        eps=1e-08,
        weight_decay=0
    )

if __name__ == '__main__':
    # 如果不存在该目录，创建用于保存模型的目录
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    # 如果不存在该目录，创建用于保存CycleGAN训练过程中生成的cycle.mid和transfer.mid
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    # 如果不存在该目录，创建用于保存测试数据的目录
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
    # 该路径用来放想要做风格迁移的mid音乐
    if not os.path.exists(args.transfer):
        os.makedirs(args.transfer)


    print("-------start training-------")
    counter = 0
    # 分别训练CycleGAN和分类器
    if args.type == 'cyclegan':

        for epoch in range(0, args.epochs):

            for idx, data in enumerate(music_dataloader):
                samples = model(data, optimizer_Disc, optimizer_Gen, epoch, idx, train_num, counter)


                # 将samples中的Tensor数据转为numpy后保存为midi
                for i in range(len(samples)):
                    samples[i] = samples[i].transpose((0, 2, 3, 1))

                sample_dir = os.path.join(args.sample_dir)

                if idx % 200 == 0:

                    sample_model(samples=samples,
                                 sample_dir=args.sample_dir,
                                 epoch=epoch,
                                 idx=idx)

                counter += 1

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_Gen': optimizer_Gen.state_dict(),
                'optimizer_Disc': optimizer_Disc.state_dict()
            },
                checkpoint_dir=args.checkpoint_dir,
                filename='{}2{}checkpoint.pth.tar'.format(args.domainA , args.domainB))

    if args.type == 'classifier':

        pass
