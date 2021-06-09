# coding=utf-8
import cv2
import numpy as np
import datetime
import os
from tqdm import tqdm
from glob import glob
from data_utils.label_utils import draw_contour_pupil
from data_utils.utils import clean_val_file

np.set_printoptions(threshold=np.inf)


val_file_path = './data/val/img/'
val_file_list = glob(val_file_path + '*.jpg')
print(len(val_file_list))
clean_val_file(val_file_path)
val_file_list = glob(val_file_path + '*.jpg')
print(len(val_file_list))
