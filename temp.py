# coding=utf-8
import cv2
import numpy as np
import datetime
import os
from tqdm import tqdm
from glob import glob
from data_utils.label_utils import draw_contour_pupil

np.set_printoptions(threshold=np.inf)

label = cv2.imread('./data/predict/demo1_dice_test.png', 0)
print(label)