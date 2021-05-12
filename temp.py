# coding=utf-8
import cv2
import numpy as np
import datetime
import os
from tqdm import tqdm
from glob import glob

np.set_printoptions(threshold=np.inf)

# start_time = datetime.datetime.now()
#
# tran_tab = str.maketrans('- :.', '____')
# start_time = str(start_time).translate(tran_tab)
# print(start_time)

# intab = "aeiou"
# outtab = "12345"
# trantab = maketrans(intab, outtab)

# str = "this is string example....wow!!!";
# print
# str.translate(trantab);
#
# file_list = []
# for index in range(15):
# file_list.extend(glob('./data/celeb_ori_label/' + str(index) + '/*.png'))
file_list = glob('./data/celeb_ori_label/' + '*.png')
print(len(file_list))

file_list = tqdm(file_list)
for file in file_list:
    os.remove(file)
    file_list.set_description("删除中")
