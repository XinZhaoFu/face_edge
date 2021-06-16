# coding=utf-8
import cv2
import numpy as np
import datetime
import os
from tqdm import tqdm
from glob import glob
from data_utils.label_utils import draw_contour_pupil
from data_utils.utils import clean_val_file
from PIL import Image
from scipy.interpolate import UnivariateSpline

np.set_printoptions(threshold=np.inf)

img = cv2.imread('./data/test/demo8.jpg')
label = cv2.imread('./data/predict/2021_06_16_09_53_21_658719_u2net_dice_demo8.png', 0)
label_pil = Image.open('./data/predict/2021_06_16_09_53_21_658719_u2net_dice_demo8.png')
img_rows, img_cols, _ = img.shape

label1 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
label2 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
label3 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_BITS)
label4 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
label5 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
label6 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_LINEAR)
label7 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_BITS2)
label8 = cv2.resize(label, dsize=(img_rows*16, img_cols*16), interpolation=cv2.INTER_CUBIC)
label8 = cv2.resize(label8, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
label9 = cv2.resize(label, dsize=(img_rows*8, img_cols*8), interpolation=cv2.INTER_CUBIC)
label9 = cv2.resize(label9, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
label10 = cv2.resize(label, dsize=(img_rows*4, img_cols*4), interpolation=cv2.INTER_CUBIC)
label10 = cv2.resize(label10, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
label11 = cv2.resize(label, dsize=(img_rows*2, img_cols*2), interpolation=cv2.INTER_CUBIC)
label11 = cv2.resize(label11, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
label12 = label8 * 0.25 + label9 * 0.25 + label10 * 0.25 + label11 * 0.25

for _ in range(20):
    label11 = cv2.resize(label11, dsize=(img_rows * 2, img_cols * 2), interpolation=cv2.INTER_CUBIC)
    label11 = cv2.resize(label11, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)

for _ in range(10):
    label8 = cv2.resize(label8, dsize=(img_rows * 16, img_cols * 16), interpolation=cv2.INTER_CUBIC)
    label8 = cv2.resize(label8, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)

cv2.imwrite('./data/temp/temp/demo8_1.png', label1)
cv2.imwrite('./data/temp/temp/demo8_2.png', label2)
cv2.imwrite('./data/temp/temp/demo8_3.png', label3)
cv2.imwrite('./data/temp/temp/demo8_4.png', label4)
cv2.imwrite('./data/temp/temp/demo8_5.png', label5)
cv2.imwrite('./data/temp/temp/demo8_6.png', label6)
cv2.imwrite('./data/temp/temp/demo8_7.png', label7)
cv2.imwrite('./data/temp/temp/demo8_8.png', label8)
cv2.imwrite('./data/temp/temp/demo8_9.png', label9)
cv2.imwrite('./data/temp/temp/demo8_10.png', label10)
cv2.imwrite('./data/temp/temp/demo8_11.png', label11)
cv2.imwrite('./data/temp/temp/demo8_12.png', label12)

label_pil = label_pil.resize((img_rows, img_cols), Image.ANTIALIAS)
label_pil.save('./data/temp/temp/demo8_pil.png')
