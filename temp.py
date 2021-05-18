# coding=utf-8
import cv2
import numpy as np
import datetime
import os
from tqdm import tqdm
from glob import glob

np.set_printoptions(threshold=np.inf)

img1 = cv2.imread('./data/temp/celeb_ori_img/0.jpg')
rows, cols, channels = img1.shape
label1 = cv2.imread('./data/temp/celeb_edge/00000_edge.png', 0)

label1 = cv2.resize(label1, dsize=(rows, cols), interpolation=cv2.INTER_CUBIC)

for row in range(rows):
    for col in range(cols):
        if label1[row, col] > 128:
            value = label1[row, col]
            img1[row, col, :] = [value, value, value]


cv2.imwrite('./data/temp/merge.jpg', img1)