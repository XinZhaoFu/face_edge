# coding=utf-8
import cv2
import numpy as np
import datetime
import os
from tqdm import tqdm
from glob import glob
from data_utils.label_utils import draw_contour_pupil

np.set_printoptions(threshold=np.inf)

label = cv2.imread('./data/celeb_edge/0.png', 0) / 255
label_rows , label_cols = label.shape
points_255 = np.where(label == 1)
points_0 = np.where(label == 0)
print(len(points_255[0]))
print(len(points_0[0]))
print(len(points_255[0]) + len(points_0[0]))
print(label_rows * label_cols)
