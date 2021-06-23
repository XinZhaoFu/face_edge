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

img_list = glob('./data/res/con_temp/' + '*.jpg')
label_list = glob('./data/res/con_temp/' + '*.png')

img_list.sort()
label_list.sort()
# print(len(img_list), len(label_list))
label_list1 = label_list[:15]
label_list2 = label_list[15:30]
label_list3 = label_list[30:]
# assert len(img_list) == len(label_list)

con_img = np.empty((512 * 15, 512 * 4, 3))

for index, (img_path, label1_path, label2_path, label3_path) in enumerate(zip(img_list, label_list1, label_list2, label_list3)):
    img = cv2.imread(img_path)
    label1 = cv2.imread(label1_path)
    label2 = cv2.imread(label2_path)
    label3 = cv2.imread(label3_path)

    img = cv2.resize(img, (512, 512))
    label1 = cv2.resize(label1, (512, 512))
    label2 = cv2.resize(label2, (512, 512))
    label3 = cv2.resize(label3, (512, 512))

    con_img[index * 512:(index + 1) * 512, :512, :] = img
    con_img[index * 512:(index + 1) * 512, 512:1024, :] = label1
    con_img[index * 512:(index + 1) * 512, 1024:1536, :] = label2
    con_img[index * 512:(index + 1) * 512, 1536:2048, :] = label3

cv2.imwrite('./data/res/con_temp/all_demo.jpg', con_img)

# img1 = cv2.imread('./data/res/temp/con_0.jpg')
# img2 = cv2.imread('./data/res/temp/con_5.jpg')
# img3 = cv2.imread('./data/res/temp/con_10.jpg')
# img4 = cv2.imread('./data/res/temp/con_15.jpg')
#
# con_img1 = np.empty((2560*2, 2048, 3))
# con_img2 = np.empty((2560*2, 2048, 3))
# all_img = np.empty((2560*4, 2048, 3))
#
# con_img1[:2560, :, :] = img1
# con_img1[2560:, :, :] = img2
# con_img2[:2560, :, :] = img3
# con_img2[2560:, :, :] = img4
# all_img[:5120, :, :] = con_img1
# all_img[5120:, :, :] = con_img2
#
# cv2.imwrite('./data/res/temp/con_demo1.jpg', con_img1)
# cv2.imwrite('./data/res/temp/con_demo2.jpg', con_img2)
# cv2.imwrite('./data/res/temp/all_demo.jpg', all_img)

# for index in range(0, 20, 5):
#     con_img = np.empty((512 * 5, 512 * 4, 3))
#     for sub_index in range(index, index+5):
#         img = cv2.imread('./data/res/temp/demo' + str(sub_index) + '.jpg')
#         sub_index = sub_index % 5
#         con_img[sub_index*512:(sub_index+1)*512, :, :] = img
#     cv2.imwrite('./data/res/temp/con_' + str(index) + '.jpg', con_img)

# for index in range(1, 20):
#     img = cv2.imread('./data/res/sample/demo' + str(index) + '.jpg')
#     img = cv2.resize(img, (512, 512))
#     label_gray = cv2.imread('./data/res/predict1/2021_06_23_11_15_19_933264_u2net_bin_02aug10000_demo' + str(index) + '.png')
#     label_red = cv2.imread('./data/res/predict1/2021_06_23_11_13_08_273202_u2net_bin_02aug10000_demo' + str(index) + '.png')
#     label_green = cv2.imread('./data/res/predict1/2021_06_23_11_18_31_989915_u2net_bin_02aug10000_demo' + str(index) + '.png')
#
#     con_img = np.empty((512, 512 * 4, 3))
#     con_img[:, 0:512, :] = img
#     con_img[:, 512:1024, :] = label_gray
#     con_img[:, 1024:1536, :] = label_red
#     con_img[:, 1536:2048, :] = label_green
#
#     cv2.imwrite('./data/res/temp/demo' + str(index) + '.jpg', con_img)

# img = cv2.imread('./data/temp/celeb_edge/0_random_filling.png', 0)
# img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_AREA)
# _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
# # img = cv2.pyrDown(img)
# print(img)
# cv2.imwrite('./data/temp/temp/0.png', img)

#
# img = cv2.imread('./data/temp/celeb_ori_img/1.jpg')
#
# points = get_point(point_file_path='./data/temp/celeb_106points/1.txt')
#
# # x, y = [], []
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
#
#
# def get_target_points(points, index_list):
#     target_points = []
#     for index in index_list:
#         target_points.append(points[index])
#
#     return target_points
#
#
# left_eye_brow_points = get_target_points(points, [33, 34, 35, 36, 37, 67, 66, 65, 64, 33])
# right_eye_brow_points = get_target_points(points, [38, 39, 40, 41, 42, 71, 70, 69, 68, 38])
# left_eye_points = get_target_points(points, [52, 53, 72, 54, 55])
#
# img = draw_line(img, left_eye_brow_points, (255, 255, 255), 1)
# img = draw_line(img, points[38:43], (255, 255, 255), 1)
# img = draw_line(img, points[84:91], (255, 255, 255), 1)
# img = draw_line(img, points[:33], (255, 255, 255), 1)

# for index in range(1, len(point_x_list)):
#     point1 = (int(point_x_list[index-1]), int(point_y_list[index-1]))
#     point2 = (int(point_x_list[index]), int(point_y_list[index]))
#     cv2.line(img, point1, point2, (255, 255, 255), 1, cv2.LINE_AA)

# cv2.imwrite('./data/temp/temp/fl_1.jpg', img)

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
