from glob import glob
import numpy as np
import cv2

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


def concat_label(labels, class_codes, num_label_flag, save_path, contour_path, img_path, priority=None):
    """
    合并各类数字标签
    同时需要注意遮盖问题 目前的优先级是预估的 不一定是准确的
    labels 和 class_codes的索引需要对应
    :param img_path:
    :param contour_path:
    :param save_path:
    :param num_label_flag:
    :param priority:
    :param class_codes:
    :param labels:
    :return:
    """
    if priority is None:
        priority = (1, 14, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18)

    con_label = np.zeros(shape=labels[0].shape, dtype=np.uint8)
    rows, cols = labels[0].shape
    priority_labels = []

    for pri_code in priority:
        if pri_code in class_codes:
            index_class_codes = class_codes.index(pri_code)
            label = labels[index_class_codes]
            label = code_label(label, pri_code)
            priority_labels.append(label)

    # 这里效率很低 后面觉得慢的话需要优化这里
    for label in priority_labels:
        for row in range(rows):
            for col in range(cols):
                if label[row, col] != 0:
                    con_label[row, col] = label[row, col]

    # con_label = cv2.Canny(con_label, 0, 0)
    con_label = get_nose(con_label, small_rate=0.9)
    con_label = get_iris(con_label, contour_path, str(int(num_label_flag)), img_path=img_path, is_canny=True)

    cv2.imwrite(save_path + num_label_flag + '_edge.png', con_label)
    print(save_path + num_label_flag + '_edge.png')




def main():
    save_path = '../data/temp/celeb_edge/'
    contour_path = '../data/temp/eye_contour/'
    img_path = '../data/temp/celeb_ori_img/'
    label_path = '../data/temp/ori_celeb_label/'
    label_file_list = glob(label_path + '*.png')
    label_file_list.sort()

    num_label_flag = None
    labels = []
    class_codes = []

    for label_file_path in label_file_list:
        label = cv2.imread(label_file_path, 0)

        label_name = (label_file_path.split('/')[-1]).split('.')[0]
        num_label, class_label = label_name.split('_')[0], label_name[6:]
        class_code_label = get_class_code(class_label)

        if num_label_flag is None:
            num_label_flag = num_label
            labels.append(label)
            class_codes.append(class_code_label)

        if num_label_flag == num_label:
            labels.append(label)
            class_codes.append(class_code_label)
        else:
            concat_label(labels=labels,
                         class_codes=class_codes,
                         num_label_flag=num_label_flag,
                         contour_path=contour_path,
                         save_path=save_path,
                         img_path=img_path)

            num_label_flag = num_label
            labels = [label]
            class_codes = [class_code_label]

    if len(labels) != 0:
        concat_label(labels=labels,
                     class_codes=class_codes,
                     num_label_flag=num_label_flag,
                     contour_path=contour_path,
                     save_path=save_path,
                     img_path=img_path)


if __name__ == '__main__':
    main()
