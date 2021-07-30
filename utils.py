# -*- codeing = utf-8 -*-
# @Time :2021/7/20 0:19
# @Author :Onion
# @File :utils.py
# @Software :PyCharm
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import numpy as np
import argparse
import os
import pypianoroll, pretty_midi
import shutil
# 数据预处理，对多音轨的mid做处理
parser = argparse.ArgumentParser(description='Train_Parameters')
parser.add_argument('--dataset_root', dest='dataset_root', default='datasets', help='the dataset root path')
parser.add_argument('--dataset_A_dir', dest='dataset_A_dir', default='Classical',
                    help='path of the dataset of domain A')
parser.add_argument('--dataset_B_dir', dest='dataset_B_dir', default='Jazz', help='path of the dataset of domain B')

args = parser.parse_args()
# is_best用来选择保存最佳模型
def save_checkpoint(state, checkpoint_dir, is_best = False, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(checkpoint_dir,filename))
    if is_best:
        shutil.copyfile(os.path.join(checkpoint_dir, filename), os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def save_multitrack_to_mid(save_dir, multitrack_list):
    return None


# CycleGAN原先论文中的保存midi的方式
def save_numpy_to_midi(piano_rolls, program_nums=None, is_drum=None, filename='test.mid', velocity=100,
                       tempo=120.0, beat_resolution=24):
    if len(piano_rolls) != len(program_nums) or len(piano_rolls) != len(is_drum):
        print("Error: piano_rolls and program_nums have different sizes...")
        return False
    if not program_nums:
        program_nums = [0, 0, 0]
    if not is_drum:
        is_drum = [False, False, False]
    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    # Iterate through all the input instruments
    for idx in range(len(piano_rolls)):
        # Create an Instrument object
        instrument = pretty_midi.Instrument(
            program=program_nums[idx], is_drum=is_drum[idx])
        # Set the piano roll to the Instrument object
        set_piano_roll_to_instrument(
            piano_rolls[idx], instrument, velocity, tempo, beat_resolution)
        # Add the instrument to the PrettyMIDI object
        midi.instruments.append(instrument)
    # Write out the MIDI data
    midi.write(filename)


def save_multitrack_to_mid(save_dir, multitrack_list):
    return None


def save_multitrack_to_npz(save_dir, multitrack_list):
    return None


def kernel_initializer(w, mean=0., std=0.02):

    return nn.init.normal_(w, mean, std)


def padding(x, p=3):
    return F.pad(x, (p, p, p, p), mode='reflect')


def load_mid_to_multitrack(mid_dir):
    # pypianoroll.read()是读midi
    # pypianoroll.load()是读npz
    multitrack = pypianoroll.read(mid_dir)

    return multitrack


def load_npy(npy_dir):

    npy = np.load(npy_dir).astype(np.float32)

    return npy


def load_npz_to_multitrack(npz_dir):
    multitrack = pypianoroll.load(npz_dir)

    return multitrack


class Sampler(object):
    def __init__(self, max_length=50):
        self.maxsize = max_length
        self.num = 0
        self.samples = []

    def __call__(self, sample):
        if self.maxsize <= 0:
            return sample
        if self.num < self.maxsize:
            self.samples.append(sample)
            self.num += 1
            return sample
        if np.random.rand() > 0.5:
            idx = int(np.random.rand() * self.maxsize)
            tmp1 = copy.copy(self.samples[idx])[0]
            self.samples[idx][0] = sample[0]
            idx = int(np.random.rand() * self.maxsize)
            tmp2 = copy.copy(self.samples[idx])[1]
            self.samples[idx][1] = sample[1]
            return [tmp1, tmp2]
        else:
            return sample


def sample_model(samples, sample_dir, epoch, idx):
    print('generating samples during learning......')

    if not os.path.exists(os.path.join(sample_dir, 'A2B')):
        os.makedirs(os.path.join(sample_dir, 'A2B'))

    save_midis(samples[0], '{}/A2B/{:02d}_{:04d}_origin.mid'.format(sample_dir, epoch, idx))
    save_midis(samples[1], '{}/A2B/{:02d}_{:04d}_transfer.mid'.format(sample_dir, epoch, idx))
    save_midis(samples[2], '{}/A2B/{:02d}_{:04d}_cycle.mid'.format(sample_dir, epoch, idx))
    save_midis(samples[3], './{}/B2A/{:02d}_{:04d}_origin.mid'.format(sample_dir, epoch, idx))
    save_midis(samples[4], './{}/B2A/{:02d}_{:04d}_transfer.mid'.format(sample_dir, epoch, idx))
    save_midis(samples[5], './{}/B2A/{:02d}_{:04d}_cycle.mid'.format(sample_dir, epoch, idx))


def save_midis(bars, file_path, tempo=80.0):
    padded_bars = np.concatenate((np.zeros((bars.shape[0], bars.shape[1], 24, bars.shape[3])),
                                  bars,
                                  np.zeros((bars.shape[0], bars.shape[1], 20, bars.shape[3]))),
                                 axis=2)

    padded_bars = padded_bars.reshape(-1, 64,
                                      padded_bars.shape[2], padded_bars.shape[3])
    padded_bars_list = []
    for ch_idx in range(padded_bars.shape[3]):
        padded_bars_list.append(padded_bars[:, :, :, ch_idx].reshape(padded_bars.shape[0],
                                                                     padded_bars.shape[1],
                                                                     padded_bars.shape[2]))
    # this is for multi-track version
    # write_midi.write_piano_rolls_to_midi(padded_bars_list, program_nums=[33, 0, 25, 49, 0],
    #                                      is_drum=[False, True, False, False, False], filename=file_path, tempo=80.0)

    # this is for single-track version
    save_numpy_to_midi(piano_rolls=padded_bars_list,
                       program_nums=[0],
                       is_drum=[False],
                       filename=file_path,
                       tempo=tempo,
                       beat_resolution=4)

# CycleGAN原论文中保存的方式
def set_piano_roll_to_instrument(piano_roll, instrument, velocity=100, tempo=120.0, beat_resolution=16):
    # Calculate time per pixel
    tpp = 60.0 / tempo / float(beat_resolution)
    threshold = 60.0 / tempo / 4
    phrase_end_time = 60.0 / tempo * 4 * piano_roll.shape[0]
    # Create piano_roll_search that captures note onsets and offsets
    piano_roll = piano_roll.reshape((piano_roll.shape[0] * piano_roll.shape[1], piano_roll.shape[2]))
    piano_roll_diff = np.concatenate((np.zeros((1, 128), dtype=int), piano_roll, np.zeros((1, 128), dtype=int)))
    piano_roll_search = np.diff(piano_roll_diff.astype(int), axis=0)
    # Iterate through all possible(128) pitches

    for note_num in range(128):
        # Search for notes
        start_idx = (piano_roll_search[:, note_num] > 0).nonzero()
        start_time = list(tpp * (start_idx[0].astype(float)))
        # print('start_time:', start_time)
        # print(len(start_time))
        end_idx = (piano_roll_search[:, note_num] < 0).nonzero()
        end_time = list(tpp * (end_idx[0].astype(float)))
        # print('end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('duration each note:', duration)
        # print(len(duration))

        temp_start_time = [i for i in start_time]
        temp_end_time = [i for i in end_time]

        for i in range(len(start_time)):
            # print(start_time)
            if start_time[i] in temp_start_time and i != len(start_time) - 1:
                # print('i and start_time:', i, start_time[i])
                t = []
                current_idx = temp_start_time.index(start_time[i])
                for j in range(current_idx + 1, len(temp_start_time)):
                    # print(j, temp_start_time[j])
                    if temp_start_time[j] < start_time[i] + threshold and temp_end_time[j] <= start_time[i] + threshold:
                        # print('popped start time:', temp_start_time[j])
                        t.append(j)
                        # print('popped temp_start_time:', t)
                for _ in t:
                    temp_start_time.pop(t[0])
                    temp_end_time.pop(t[0])
                # print('popped temp_start_time:', temp_start_time)

        start_time = temp_start_time
        # print('After checking, start_time:', start_time)
        # print(len(start_time))
        end_time = temp_end_time
        # print('After checking, end_time:', end_time)
        # print(len(end_time))
        duration = [pair[1] - pair[0] for pair in zip(start_time, end_time)]
        # print('After checking, duration each note:', duration)
        # print(len(duration))

        if len(end_time) < len(start_time):
            d = len(start_time) - len(end_time)
            start_time = start_time[:-d]
        # Iterate through all the searched notes
        for idx in range(len(start_time)):
            if duration[idx] >= threshold:
                # Create an Note object with corresponding note number, start time and end time
                note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx], end=end_time[idx])
                # Add the note to the Instrument object
                instrument.notes.append(note)
            else:
                if start_time[idx] + threshold <= phrase_end_time:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=start_time[idx] + threshold)
                else:
                    # Create an Note object with corresponding note number, start time and end time
                    note = pretty_midi.Note(velocity=velocity, pitch=note_num, start=start_time[idx],
                                            end=phrase_end_time)
                # Add the note to the Instrument object
                instrument.notes.append(note)
    # Sort the notes by their start time
    instrument.notes.sort(key=lambda note: note.start)
    # print(max([i.end for i in instrument.notes]))
    # print('tpp, threshold, phrases_end_time:', tpp, threshold, phrase_end_time)

def to_binary(bar, threshold=0.0):
    """Turn velocity value into boolean"""
    track_is_max = np.equal(bar, np.max(bar, axis=-1, keepdims=True))
    track_pass_threshold = (bar > threshold)
    out_track = np.logical_and(track_is_max, track_pass_threshold)
    return out_track