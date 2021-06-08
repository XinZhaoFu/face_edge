# coding=utf-8
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import datetime
from label_utils import get_contour_pupil_label, get_nose_label

"""
label	class
0	background
1	skin
2	left eyebrow
3	right eyebrow
4	left eye
5	right eye
6	nose
7	upper lip
8	inner mouth
9	lower lip
10	hair
"""

np.set_printoptions(threshold=np.inf)


def clean_ori_nose(label):
    (rows, cols) = np.where(label == 6)
    label[rows, cols] = 1

    return label


def get_label(label, contour_point_file_path, nose_point_file_path, img_rows, img_cols):
    label = clean_ori_nose(label)
    label = get_contour_pupil_label(label=label,
                                    contour_point_file_path=contour_point_file_path,
                                    img_rows=img_rows,
                                    img_cols=img_cols)
    # label = cv2.Canny(label, 0, 0)
    label = get_nose_label(label=label,
                           img_rows=img_rows,
                           img_cols=img_cols,
                           nose_point_file_path=nose_point_file_path,
                           draw_type=0)

    return label


def main():
    # img_path = '../data/lapa_ori_img/'
    # label_path = '../data/lapa_ori_label/'
    # save_label_path = '../data/lapa_edge/'
    # contour_point_file_path = '../data/lapa_eye_contour/'
    # nose_point_file_path = '../data/lapa_106points/'

    img_path = '../data/temp/lapa_ori_img/'
    label_path = '../data/temp/lapa_ori_label/'
    save_label_path = '../data/temp/lapa_edge/'
    contour_point_file_path = '../data/temp/lapa_eye_contour/'
    nose_point_file_path = '../data/temp/lapa_106points/'

    img_file_list = glob(img_path + '*.jpg')
    label_file_list = glob(label_path + '*.png')

    assert len(img_file_list) == len(label_file_list)
    img_file_list.sort()
    label_file_list.sort()

    for index in tqdm(range(len(img_file_list))):
        label_file_path = label_file_list[index]
        img_file_path = img_file_list[index]

        img = cv2.imread(img_file_path)
        label = cv2.imread(label_file_path, 0)
        img_rows, img_cols, _ = img.shape
        label_name = (label_file_path.split('/')[-1]).split('.')[0]
        cur_contour_point_file_path = contour_point_file_path + label_name + '.txt'
        cur_nose_point_file_path = nose_point_file_path + label_name + '.txt'

        label = get_label(label, cur_contour_point_file_path, cur_nose_point_file_path, img_rows, img_cols)
        cv2.imwrite(save_label_path + label_name + '.png', label)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])
