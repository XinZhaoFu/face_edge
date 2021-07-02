from glob import glob
import cv2
import datetime

import numpy as np
from tqdm import tqdm
from label_utils import get_contour_pupil_label, get_augmentation, get_class_code, get_nose_label, concat_label, \
    get_lower_nose_edge
from utils import recreate_dir

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


def get_semantic_label(label_path, save_path, is_nose=True):
    """
    获取语义分割标签

    :param is_nose:
    :param label_path:
    :param save_path:
    :return:
    """
    label_file_list = glob(label_path + '*.png')
    label_file_list.sort()

    last_label_name = None
    labels = []
    class_codes = []

    for index in tqdm(range(len(label_file_list))):
        label_file_path = label_file_list[index]
        label = cv2.imread(label_file_path, 0)

        label_name = (label_file_path.split('/')[-1]).split('.')[0]
        cur_label_name, label_class = label_name.split('_')[0], label_name[6:]
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

            res_label = concat_label(labels, class_codes, is_nose=is_nose)
            cv2.imwrite(save_path + last_label_name_easy + '.png', res_label)

            last_label_name = cur_label_name
            labels = [label]
            class_codes = [class_code_label]

        if index == len(label_file_list) - 1:
            cur_label_name_easy = str(int(cur_label_name))
            res_label = concat_label(labels, class_codes, is_nose=is_nose)
            cv2.imwrite(save_path + cur_label_name_easy + '.png', res_label)


def add_contour_nose_label(img_path,
                           save_semantic_path,
                           contour_point_file_path,
                           nose_point_file_path,
                           save_label_path,
                           save_img_path,
                           is_augmentation=False,
                           is_edge=False,
                           is_lower_nose_edge=True,
                           is_seg=False):
    """
    在语义分割标签的基础上获得含有虹膜和鼻子的轮廓标签  并进行离线的数据扩增

    :param is_seg:
    :param is_lower_nose_edge:
    :param is_edge:
    :param save_img_path:
    :param img_path:
    :param save_semantic_path:
    :param contour_point_file_path:
    :param nose_point_file_path:
    :param save_label_path:
    :param is_augmentation:
    :return:
    """
    img_file_list = glob(img_path + '*.jpg')
    semantic_label_file_list = glob(save_semantic_path + '*.png')
    contour_point_file_list = glob(contour_point_file_path + '*.txt')
    nose_point_file_list = glob(nose_point_file_path + '*.txt')

    print(len(img_file_list), len(semantic_label_file_list), len(contour_point_file_list), len(nose_point_file_list))
    assert len(img_file_list) == len(semantic_label_file_list) == \
           len(contour_point_file_list) == len(nose_point_file_list)

    img_file_list.sort()
    semantic_label_file_list.sort()
    contour_point_file_list.sort()
    nose_point_file_list.sort()

    for img_file_path, semantic_label_path, contour_point_path, nose_point_path \
            in tqdm(zip(img_file_list, semantic_label_file_list, contour_point_file_list, nose_point_file_list),
                    total=len(img_file_list)):
        img_name = (img_file_path.split('/')[-1]).split('.')[0]
        label_name = (semantic_label_path.split('/')[-1]).split('.')[0]
        contour_point_name = (contour_point_path.split('/')[-1]).split('.')[0]
        nose_point_name = (nose_point_path.split('/')[-1]).split('.')[0]

        assert img_name == label_name == contour_point_name == nose_point_name

        img = cv2.imread(img_file_path)
        semantic_label = cv2.imread(semantic_label_path, 0)
        img_rows, img_cols, _ = img.shape

        con_label = np.zeros(shape=semantic_label.shape)

        if is_edge:
            con_label = get_contour_pupil_label(label=semantic_label,
                                                contour_point_file_path=contour_point_path,
                                                img_rows=img_rows,
                                                img_cols=img_cols,
                                                is_canny=True)
            con_label = get_nose_label(label=con_label,
                                       img_rows=img_rows,
                                       img_cols=img_cols,
                                       nose_point_file_path=nose_point_path,
                                       draw_type=0)

        if is_lower_nose_edge:
            contours = get_contour_pupil_label(label=semantic_label,
                                               contour_point_file_path=contour_point_path,
                                               img_rows=img_rows,
                                               img_cols=img_cols,
                                               is_canny=False)
            con_label = get_lower_nose_edge(semantic_label)
            cv2.drawContours(con_label, contours, -1, 255, 1)
        # if is_seg:

        cv2.imwrite(save_label_path + label_name + '.png', con_label)

        if is_augmentation:
            cv2.imwrite(save_img_path + img_name + '.jpg', img)
            get_augmentation(img, con_label, save_img_path, save_label_path, img_name, label_name)


def main():
    is_get_semantic_label = False
    is_augmentation = True
    is_edge = False
    is_lower_nose_edge = True
    is_seg = False
    is_all_file = False
    is_recreate = False

    if is_all_file:
        save_semantic_path = '../data/celeb_semantic_label/'
        save_label_path = '../data/celeb_edge/'
        save_img_path = '../data/celeb_aug_img/'
        contour_point_file_path = '../data/celeb_eye_contour/'
        nose_point_file_path = '../data/celeb_106points/'
        img_path = '../data/celeb_ori_img/'
        label_path = '../data/celeb_ori_label/'
    else:
        save_semantic_path = '../data/temp/celeb_semantic_label/'
        save_label_path = '../data/temp/celeb_edge/'
        save_img_path = '../data/temp/celeb_aug_img/'
        contour_point_file_path = '../data/temp/celeb_eye_contour/'
        nose_point_file_path = '../data/temp/celeb_106points/'
        img_path = '../data/temp/celeb_ori_img/'
        label_path = '../data/temp/celeb_ori_label/'

    if is_recreate:
        recreate_dir(save_label_path)
        recreate_dir(save_img_path)

    if is_get_semantic_label is True:
        print('[INFO] 重新获得语义label')
        get_semantic_label(label_path, save_semantic_path, is_nose=True)

    print('[INFO] 进行数据扩增')
    add_contour_nose_label(img_path=img_path,
                           save_semantic_path=save_semantic_path,
                           contour_point_file_path=contour_point_file_path,
                           nose_point_file_path=nose_point_file_path,
                           save_label_path=save_label_path,
                           save_img_path=save_img_path,
                           is_augmentation=is_augmentation,
                           is_edge=is_edge,
                           is_lower_nose_edge=is_lower_nose_edge,
                           is_seg=is_seg)


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print('time:\t' + str(end_time - start_time).split('.')[0])
