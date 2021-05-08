# coding=utf-8
import cv2
import numpy as np
from glob import glob
from skimage.measure import regionprops, label

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


def get_iris(seg_label):
    """
    读取分割标签，获得canny后的边缘标签记作edge_label，分割标签读取左右眼区域，获得轮廓，进行椭圆拟合获得椭圆中心点坐标以及长短轴直径
    以椭圆中心点坐标作为假定虹膜中心点坐标，以短轴半径微调作为假定虹膜半径
    在空npArray上绘制该假定虹膜
    :param seg_label_path:
    :return:
    """
    eyes_label = np.zeros(shape=seg_label.shape, dtype=np.uint8)
    rows, cols = seg_label.shape
    for row in range(rows):
        for col in range(cols):
            if seg_label[row, col] == 4 or seg_label[row, col] == 5:
                eyes_label[row, col] = 255

    contours, hierarchy = cv2.findContours(eyes_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    seg_label = cv2.Canny(seg_label, 0, 0)
    for contour in contours:
        if len(contour) < 5:
            continue
        (x, y), (a, b), angle = cv2.fitEllipse(contour)  # (x, y)中心点 (a, b)长短轴直径 angle中心旋转角度
        rad = int(min(a, b) * 0.5)
        if rad <= 7:
            rad += 1
        else:
            rad = int(rad * 1.25)
        seg_label = cv2.circle(seg_label, center=(int(x), int(y)), radius=rad, color=255, thickness=1)

    return seg_label


def get_nose(seg_label):
    """
    获得更小的鼻子轮廓
    :param seg_label_path:
    :return:
    """
    nose_label = np.zeros(shape=seg_label.shape, dtype=np.uint8)

    rows, cols = seg_label.shape
    # print(rows, cols)
    for row in range(rows):
        for col in range(cols):
            if seg_label[row][col] == 6:
                seg_label[row][col] = 1
                nose_label[row][col] = 6

    # 暂时没用bbox信息 感觉写起来麻烦
    ori_centroid, _ = get_centroid_bbox(nose_label)
    ori_centroid_row, ori_centroid_col = int(ori_centroid[0]), int(ori_centroid[1])
    small_rate = 0.6
    temp = np.zeros(shape=seg_label.shape, dtype=np.uint8)
    s_rows, s_cols = int(small_rate * rows), int(small_rate * cols)
    nose_label = cv2.resize(nose_label, dsize=(s_cols, s_rows), interpolation=cv2.INTER_NEAREST)
    # new_centroid_row, new_centroid_col = get_centroid(nose_label)
    new_centroid_row, new_centroid_col = int(small_rate * ori_centroid_row), int(small_rate * ori_centroid_col)
    # print(temp.shape, nose_label.shape)
    # print(ori_centroid_row, ori_centroid_col, new_centroid_row, new_centroid_col, s_rows, s_cols)
    temp[ori_centroid_row - new_centroid_row:ori_centroid_row - new_centroid_row + s_rows,
    ori_centroid_col - new_centroid_col:ori_centroid_col - new_centroid_col + s_cols] = nose_label[:, :]

    for row in range(rows):
        for col in range(cols):
            if temp[row][col] == 6:
                seg_label[row][col] = 6

    return seg_label


def get_centroid_bbox(seg_label):
    labels, labels_num = label(seg_label, background=0, return_num=True)
    assert labels_num == 1
    regions = regionprops(labels)  # 获取各个联通域的属性

    return regions[0].centroid, regions[0].bbox


def get_label(seg_label_file_path, save_label_path):
    seg_label_name = seg_label_file_path.split('/')[-1]
    print(seg_label_name)

    seg_label = cv2.imread(seg_label_file_path, 0)
    seg_label = get_nose(seg_label)
    seg_label = get_iris(seg_label)
    seg_label = cv2.resize(seg_label, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_label_path + seg_label_name, seg_label)


def main():
    label_path = '../data/lapa_ori_label/'
    save_label_path = '../data/lapa_edge/'
    label_file_list = glob(label_path + '*.png')
    print(len(label_file_list))
    for label_file_path in label_file_list:
        get_label(label_file_path, save_label_path)


if __name__ == '__main__':
    main()
