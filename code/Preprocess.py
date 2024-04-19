import glob
import os

import scipy.io as scio

file_data = "D:\\Study\\capstone\\SEED-VIG\\EEG_Feature_2Hz\\"
file_label = "D:\\Study\\capstone\\SEED-VIG\\perclos_labels\\"


def get_two_character_number(filename):
    if len(filename) >= 2:
        if filename[1] == '_':
            return int(filename[0])
        else:
            return int(filename[:2])
    else:
        return 0

EEG2Hz_data=[]
EEG2Hz_label=[]
# def getfiles(dir1,dir2):
fdata = os.listdir(file_data)
flabel = os.listdir(file_label)
fdata = sorted(fdata, key=get_two_character_number)
flabel = sorted(flabel, key=get_two_character_number)
for i in range(23):
    EEG2Hz_data.append(scio.loadmat(file_data+fdata[i]))
    EEG2Hz_label.append(scio.loadmat(file_label+flabel[i]))
    # print(filenames)


# getfiles(file_data,file_label)


# data = scio.loadmat('D:\\Study\\capstone\\SEED-VIG\\EEG_Feature_2Hz\\2_20151106_noon.mat')
# label = scio.loadmat('D:\\Study\\capstone\\SEED-VIG\\perclos_labels\\2_20151106_noon.mat')
