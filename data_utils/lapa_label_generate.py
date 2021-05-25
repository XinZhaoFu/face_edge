# coding=utf-8
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from data_utils.label_generate import get_contour_pupil_label, get_nose_label

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
    label_rows, label_cols = label.shape
    for row in range(label_rows):
        for col in range(label_cols):
            if label[row][col] == 6:
                label[row][col] = 1

    return label


def get_label(label_file_path, save_label_path, contour_point_file_path, nose_point_file_path, img_rows, img_cols):
    label_name = label_file_path.split('/')[-1]

    label = cv2.imread(label_file_path, 0)
    label = clean_ori_nose(label)
    label = get_contour_pupil_label(label=label,
                                    contour_point_file_path=contour_point_file_path,
                                    img_rows=img_rows,
                                    img_cols=img_cols)
    label = cv2.Canny(label, 0, 0)
    label = get_nose_label(label=label,
                           img_rows=img_rows,
                           img_cols=img_cols,
                           nose_point_file_path=nose_point_file_path,
                           draw_type=0)

    cv2.imwrite(save_label_path + label_name, label)


def main():
    img_path = ''
    label_path = '../data/lapa_ori_label/'
    save_label_path = '../data/lapa_edge/'
    contour_point_file_path = ''
    nose_point_file_path = ''

    img_file_list = glob(img_path + '*.jpg')
    label_file_list = glob(label_path + '*.png')
    assert len(img_file_list) == len(label_file_list)

    for index in tqdm(range(len(img_file_list))):
        label_file_path = label_file_list[index]
        img_file_path = img_file_list[index]
        get_label(label_file_path, save_label_path, contour_point_file_path, nose_point_file_path, img_rows, img_cols)


if __name__ == '__main__':
    main()
