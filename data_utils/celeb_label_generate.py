from glob import glob
import numpy as np
import cv2
import datetime
from tqdm import tqdm
from label_generate import get_contour_pupil_label, get_nose_label

"""
label class         name
0   background      
1   skin            skin
2	left eyebrow    l_brow
3	right eyebrow   r_brow
4	left eye        l_eye
5	right eye       r_eye
6	nose            nose
7	upper lip       u_lip
8	inner mouth     mouth
9	lower lip       l_lip
10	hair            hair
11  neck            neck
12  left ear        l_ear
13  right ear       r_ear
14  cloth           cloth
15  earring         ear_r
16  hat             hat
17  eye glasses     eye_g
18  neck_l          neck_l
--------------------------
19  iris            iris
"""


def get_class_code(class_label):
    dict_class_code = {'skin': 1, 'l_brow': 2, 'r_brow': 3, 'l_eye': 4, 'r_eye': 5, 'nose': 6, 'u_lip': 7,
                       'mouth': 8, 'l_lip': 9, 'hair': 10, 'neck': 11, 'l_ear': 12, 'r_ear': 13, 'cloth': 14,
                       'ear_r': 15, 'hat': 16, 'eye_g': 17, 'neck_l': 18}
    return dict_class_code[class_label]


def code_label(label, class_code):
    """
    转为数字标签
    :param label:
    :param class_code:
    :return:
    """
    return label / 255 * class_code


def concat_label(labels, class_codes, contour_point_file_path, nose_point_file_path, img_rows, img_cols, priority=None):
    """
    合并各类数字标签
    同时需要注意遮盖问题 目前的优先级是预估的 不一定是准确的
    labels 和 class_codes的索引需要对应
    :param nose_point_file_path:
    :param contour_point_file_path:
    :param img_cols:
    :param img_rows:
    :param priority:
    :param class_codes:
    :param labels:
    :return:
    """
    if priority is None:
        priority = (1, 14, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18)

    con_label = np.zeros(shape=labels[0].shape, dtype=np.uint8)
    label_rows, label_cols = labels[0].shape
    priority_labels = []

    for pri_code in priority:
        if pri_code in class_codes:
            index_class_codes = class_codes.index(pri_code)
            label = labels[index_class_codes]
            label = code_label(label, pri_code)
            priority_labels.append(label)

    # 这里效率很低 后面觉得慢的话需要优化这里
    for label in priority_labels:
        for row in range(label_rows):
            for col in range(label_cols):
                if label[row, col] != 0:
                    con_label[row, col] = label[row, col]

    con_label = get_contour_pupil_label(label=con_label,
                                        contour_point_file_path=contour_point_file_path,
                                        img_rows=img_rows,
                                        img_cols=img_cols)
    con_label = cv2.Canny(con_label, 0, 0)
    con_label = get_nose_label(label=con_label,
                               label_rows=label_rows,
                               label_cols=label_cols,
                               img_rows=img_rows,
                               img_cols=img_cols,
                               nose_point_file_path=nose_point_file_path,
                               draw_type=0)

    return con_label


def main():
    # nose可以是分割的 seg  也可以是拟合的 fit
    nose_type = 'draw'
    save_path = '../data/celeb_edge/'
    contour_point_file_path = '../data/celeb_eye_contour/'
    nose_point_file_path = '../data/celeb_106points/'
    img_path = '../data/celeb_ori_img/'
    label_path = '../data/celeb_ori_label/'
    label_file_list = glob(label_path + '*.png')
    label_file_list.sort()

    last_label_name = None
    labels = []
    class_codes = []
    flag = 1
    for index in tqdm(range(len(label_file_list))):
        label_file_path = label_file_list[index]
        label = cv2.imread(label_file_path, 0)

        label_name = (label_file_path.split('/')[-1]).split('.')[0]
        cur_label_name, label_class = label_name.split('_')[0], label_name[6:]
        if nose_type == 'draw' and label_class == 'nose':
            continue
        class_code_label = get_class_code(label_class)

        if last_label_name is None:
            last_label_name = cur_label_name
            labels.append(label)
            class_codes.append(class_code_label)

        if last_label_name == cur_label_name:
            labels.append(label)
            class_codes.append(class_code_label)

        if last_label_name != cur_label_name:
            last_label_name_easy = str(int(last_label_name))
            img = cv2.imread(img_path + last_label_name_easy + '.jpg')
            img_rows, img_cols, _ = img.shape
            last_contour_point_file_path = contour_point_file_path + last_label_name_easy + '.txt'
            last_nose_point_file_path = nose_point_file_path + last_label_name_easy + '.txt'

            res_label = concat_label(labels, class_codes, last_contour_point_file_path, last_nose_point_file_path,
                                     img_rows, img_cols)
            cv2.imwrite(save_path+last_label_name_easy+'.png', res_label)

            last_label_name = cur_label_name
            labels = [label]
            class_codes = [class_code_label]

        if index == len(label_file_list)-1:
            cur_label_name_easy = str(int(cur_label_name))
            img = cv2.imread(img_path + cur_label_name_easy + '.jpg')
            img_rows, img_cols, _ = img.shape
            cur_contour_point_file_path = contour_point_file_path + cur_label_name_easy + '.txt'
            cur_nose_point_file_path = nose_point_file_path + cur_label_name_easy + '.txt'

            res_label = concat_label(labels, class_codes, cur_contour_point_file_path, cur_nose_point_file_path,
                                     img_rows, img_cols)
            cv2.imwrite(save_path + cur_label_name_easy + '.png', res_label)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])

