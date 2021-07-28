# -*- codeing = utf-8 -*-
# @Time :2021/7/20 0:20
# @Author :Onion
# @File :test.py
# @Software :PyCharm
import torch
# 将MIDI文件转换为适当的格式，我们需要完成以下步骤：
# 读取MIDI文件以生成一个Multitrack对象。
# 确定要使用的四条乐器音轨。
# 检索各音轨中的 pianoroll 矩阵，并将其调整为正确的格式。
# 将Pianoroll对象与给定的乐器相匹配，并将其存储在.npy文件中。
import pypianoroll
import pretty_midi
import music21
import numpy as np
# init with beat resolution of 4
# music_tracks = pypianoroll.Multitrack(resolution=4)
# multitrack = pypianoroll.Multitrack(resolution=4, name="0eb9b102beab4ac575947a23f06e4fce")

# pretty_midi格式的数据可以通过program属性获取到乐器
x = pretty_midi.PrettyMIDI("./datasets/train/Classical/ABC - Frank Zappa.mid")



category_list = {'Drums': [], 'Piano': [], 'Guitar': [], 'Bass': [], 'Strings': []}
program_dict = {'Drums': 0, 'Piano': 0, 'Guitar': 24, 'Bass': 32, 'Strings': 48}


multitrack = pypianoroll.read("./datasets/train/Classical/ABC - Frank Zappa.mid")
multitrack.resolution = 24

resolution = multitrack.resolution
# 尝试更换乐器,idx为multitrack中每一个音轨的索引
for idx,track in enumerate(multitrack.tracks):
    if track.is_drum:
        continue
    elif track.program == 10:
        track.program = 24
    elif track.program == 12:
        track.program = 32
    elif track.program == 88:
        track.program = 48

# 更换速度(绝对速度)
tempo = 100


# 每个小节有4拍(这里采用四分音符为1拍)
bar_resolution = 4 * multitrack.resolution

# 截取原mid的长度
tempo_array = np.full((4 * bar_resolution, 1), tempo)

# 截取原来mid的downbeat
downbeat = multitrack.downbeat[0:len(tempo_array)]


# idx为multitrack中每一个音轨的索引
for idx,track in enumerate(multitrack.tracks):
    if track.is_drum:
        category_list['Drums'].append(idx)
    elif track.program // 8 == 0:
        category_list['Piano'].append(idx)
    elif track.program // 8 == 3:
        category_list['Guitar'].append(idx)
    elif track.program // 8 == 4:
        category_list['Bass'].append(idx)
    elif track.program // 8 == 6:
        category_list['Strings'].append(idx)



# 提取指定的category_list里的音轨
tracks = []
for key in category_list:   #Bass\Drums\Guitar\Piano\Strings
    if category_list[key]:
        print(category_list[key])
        tracklist = category_list[key]
        for trackidx in (tracklist):
            tracks.append(multitrack[trackidx])
            print(multitrack[trackidx].pianoroll.shape)

# 将改变后的Multitrack保存为新的MIDI文件
# pypianoroll.write(path: str, multitrack: Multitrack)

# resolution:Time steps per quarter note(每一个四分音符包含的时间步数，在MuseGAN中使用的是24).
# 一个小节中的 第一个拍点,或可说是被强调的拍点,称为强拍(downbeat)。

newMultitrack = pypianoroll.Multitrack(name='NewMultitrack',
                                       resolution=resolution,
                                       tempo=tempo_array,
                                       downbeat=downbeat,
                                       tracks=tracks)


pypianoroll.write('NewMultitrack.mid',multitrack=newMultitrack)

# pypianoroll.plot_multitrack(multitrack)
