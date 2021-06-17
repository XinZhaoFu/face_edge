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
from data_utils.label_utils import get_point, fit_interpolation, draw_line

np.set_printoptions(threshold=np.inf)

from numpy import linspace, exp
from numpy.random import randn
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


img = cv2.imread('./data/temp/celeb_ori_img/1.jpg')

points = get_point(point_file_path='./data/temp/celeb_106points/1.txt')

# x, y = [], []
# for point_x, point_y in points[:32]:
#     if len(x) > 0 and point_x < x[-1]:
#         point_x = x[-1]
#     x.append(point_x)
#     y.append(point_y)
#
# fit_point_x, fit_point_y = fit_interpolation(x, y, 2, x[0], x[-1], 3)
# fit_point = np.empty(shape=(len(fit_point_x), 2), dtype=np.int32)
# fit_point[:, 0] = fit_point_x
# fit_point[:, 1] = fit_point_y


def get_target_points(points, index_list):
    target_points = []
    for index in index_list:
        target_points.append(points[index])

    return target_points


left_eye_brow_points = get_target_points(points, [33, 34, 35, 36, 37, 67, 66, 65, 64, 33])
right_eye_brow_points = get_target_points(points, [38, 39, 40, 41, 42, 71, 70, 69, 68, 38])
left_eye_points = get_target_points(points, [52, 53, 72, 54, 55])

img = draw_line(img, left_eye_brow_points, (255, 255, 255), 1)
img = draw_line(img, points[38:43], (255, 255, 255), 1)
img = draw_line(img, points[84:91], (255, 255, 255), 1)
img = draw_line(img, points[:33], (255, 255, 255), 1)

# for index in range(1, len(point_x_list)):
#     point1 = (int(point_x_list[index-1]), int(point_y_list[index-1]))
#     point2 = (int(point_x_list[index]), int(point_y_list[index]))
#     cv2.line(img, point1, point2, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite('./data/temp/temp/fl_1.jpg', img)

# img = cv2.imread('./data/test/demo8.jpg')
# label = cv2.imread('./data/predict/2021_06_16_09_53_21_658719_u2net_dice_demo8.png', 0)
# label_pil = Image.open('./data/predict/2021_06_16_09_53_21_658719_u2net_dice_demo8.png')
# img_rows, img_cols, _ = img.shape
#
# label_256 = cv2.resize(label, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
#
# label1 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
# label2 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_NEAREST)
# label3 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_BITS)
# label4 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC)
# label5 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_LANCZOS4)
# label6 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_LINEAR)
# label7 = cv2.resize(label, dsize=(img_rows, img_cols), interpolation=cv2.INTER_BITS2)
# label8 = cv2.resize(label, dsize=(img_rows*16, img_cols*16), interpolation=cv2.INTER_CUBIC)
# label8 = cv2.resize(label8, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
# label9 = cv2.resize(label, dsize=(img_rows*8, img_cols*8), interpolation=cv2.INTER_CUBIC)
# label9 = cv2.resize(label9, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
# label10 = cv2.resize(label, dsize=(img_rows*4, img_cols*4), interpolation=cv2.INTER_CUBIC)
# label10 = cv2.resize(label10, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
# label11 = cv2.resize(label_256, dsize=(img_rows*2, img_cols*2), interpolation=cv2.INTER_CUBIC)
# label11 = cv2.resize(label11, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
# label12 = label8 * 0.25 + label9 * 0.25 + label10 * 0.25 + label11 * 0.25
#
# for _ in range(20):
#     label11 = cv2.resize(label11, dsize=(img_rows * 2, img_cols * 2), interpolation=cv2.INTER_CUBIC)
#     label11 = cv2.resize(label11, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
#
# for _ in range(10):
#     label8 = cv2.resize(label8, dsize=(img_rows * 16, img_cols * 16), interpolation=cv2.INTER_CUBIC)
#     label8 = cv2.resize(label8, dsize=(img_rows, img_cols), interpolation=cv2.INTER_AREA)
#
# cv2.imwrite('./data/temp/temp/demo8_1.png', label1)
# cv2.imwrite('./data/temp/temp/demo8_2.png', label2)
# cv2.imwrite('./data/temp/temp/demo8_3.png', label3)
# cv2.imwrite('./data/temp/temp/demo8_4.png', label4)
# cv2.imwrite('./data/temp/temp/demo8_5.png', label5)
# cv2.imwrite('./data/temp/temp/demo8_6.png', label6)
# cv2.imwrite('./data/temp/temp/demo8_7.png', label7)
# cv2.imwrite('./data/temp/temp/demo8_8.png', label8)
# cv2.imwrite('./data/temp/temp/demo8_9.png', label9)
# cv2.imwrite('./data/temp/temp/demo8_10.png', label10)
# cv2.imwrite('./data/temp/temp/demo8_11.png', label11)
# cv2.imwrite('./data/temp/temp/demo8_12.png', label12)
#
# label_pil = label_pil.resize((img_rows, img_cols), Image.ANTIALIAS)
# label_pil.save('./data/temp/temp/demo8_pil.png')
