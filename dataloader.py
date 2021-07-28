# -*- codeing = utf-8 -*-
# @Time :2021/7/20 1:51
# @Author :Onion
# @File :dataloader.py
# @Software :PyCharm
import torch
from torch.utils.data import Dataset
from utils import load_npy
import numpy as np

class MusicDataSet(Dataset):
    def __init__(self, A_dir_list, B_dir_list, Mixed_dir_list, transform=None):
        super(MusicDataSet, self).__init__()
        self.A_dir = A_dir_list
        self.B_dir = B_dir_list
        self.Mixed_dir = Mixed_dir_list
        self.transform = transform

        assert len(A_dir_list) == len(B_dir_list),'The lengths of A are different from B'

    def __len__(self):
        len_of_A = len(self.A_dir)
        len_of_B = len(self.B_dir)

        return len_of_A if len_of_A < len_of_B else len_of_B

    def __getitem__(self, idx):
        bar_A = load_npy(self.A_dir[idx])
        bar_B = load_npy(self.B_dir[idx])
        bar_Mixed = load_npy(self.Mixed_dir[idx])
        baridx = np.array([idx])
        if len(bar_A.shape) != 3:
            bar_A = np.expand_dims(bar_A, axis=2)
        if len(bar_B.shape) != 3:
            bar_B = np.expand_dims(bar_B, axis=2)

        sample = {'baridx': baridx, 'bar_A': bar_A, 'bar_B': bar_B, 'bar_mixed': bar_Mixed}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        baridx, bar_A, bar_B, bar_mixed = sample['baridx'], sample['bar_A'], sample['bar_B'], sample['bar_mixed']
        bar_A = bar_A.transpose((2, 0, 1))
        bar_B = bar_B.transpose((2, 0, 1))
        bar_mixed = bar_mixed.transpose((2, 0, 1))

        return {
            'baridx': torch.tensor(baridx),
            'bar_A': torch.tensor(bar_A, requires_grad=False),
            'bar_B': torch.tensor(bar_B, requires_grad=False),
            'bar_mixed': torch.tensor(bar_mixed, requires_grad=False)
        }