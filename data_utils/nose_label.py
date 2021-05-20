import cv2
import numpy as np
from glob import glob
from nose_fit import nose_fit

"""
鼻梁 43-46, 鼻子下沿 47-51, 鼻子外侧 78-83
43  44  45  46  47  48  49  50  51  78  79  80  81  82  83  46与49中心点  48与80中心点  50与81中心点
0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15           16          17
"""
nose_index_list = [43, 44, 45, 46, 47, 48, 49, 50, 51, 78, 79, 80, 81, 82, 83]

img_list = glob('../data/temp/celeb_ori_img/' + '*.jpg')

flag = 1
for img_path in img_list:
    nose_point_list = []
    img_name = (img_path.split('/')[-1]).split('.')[0]
    img = cv2.imread(img_path)

    txt_path = '../data/temp/106points/' + img_name + '.txt'
    with open(txt_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            line = line.strip('\n')
            point_x_f = float(line.split(',')[0])
            point_y_f = float(line.split(',')[1])

            point_x = int(point_x_f)
            point_y = int(point_y_f)

            if index in nose_index_list:
                nose_point_list.append([point_x, point_y])
    if f:
        f.close()

    """
    自增点
    """
    point1_index_list = [3, 5, 7]
    point2_index_list = [6, 11, 12]
    assert len(point1_index_list) == len(point2_index_list)
    for index in range(len(point1_index_list)):
        point1, point2 = nose_point_list[point1_index_list[index]], nose_point_list[point2_index_list[index]]
        new_point_x = int(point1[0] + point2[0]) // 2
        new_point_y = int(point1[1] + point2[1]) // 2
        nose_point_list.append([new_point_x, new_point_y])

    """
    line_point_index_list = [[0, 1], [1, 2], [2, 3], [3, 6],
                             [4, 5], [5, 6], [6, 7], [7, 8],
                             [9, 11], [11, 13], [13, 4],
                             [10, 12], [12, 14], [14, 8]]
    """

    # line_point_index_list = [[0, 1], [1, 2], [2, 3], [3, 6],
    #                          [4, 5], [5, 6], [6, 7], [7, 8],
    #                          [9, 11], [11, 13], [13, 4],
    #                          [10, 12], [12, 14], [14, 8]]

    # for line_point_index in line_point_index_list:
    #     point1 = (nose_point_list[line_point_index[0]][0], nose_point_list[line_point_index[0]][1])
    #     point2 = (nose_point_list[line_point_index[1]][0], nose_point_list[line_point_index[1]][1])
    #     cv2.line(img, point1, point2, (255, 255, 255), 2, cv2.LINE_AA)
    nose_left_point_list, nose_right_point_list, nose_lower_point_list = [], [], []
    nose_left_point_index_list = [11, 13, 4]
    nose_right_point_index_list = [12, 14, 8]
    nose_lower_point_index_list = [4, 16, 15, 17, 8]

    for nose_left_point_index in nose_left_point_index_list:
        nose_left_point_list.append(nose_point_list[nose_left_point_index])

    for nose_right_point_index in nose_right_point_index_list:
        nose_right_point_list.append(nose_point_list[nose_right_point_index])

    for nose_lower_point_index in nose_lower_point_index_list:
        nose_lower_point_list.append(nose_point_list[nose_lower_point_index])

    nose_left_fit_point, nose_right_fit_point, nose_lower_fit_point = nose_fit(nose_left_point_list,
                                                                               nose_right_point_list,
                                                                               nose_lower_point_list)

    rows, cols, _ = img.shape
    nose_fit_label = np.zeros(shape=(rows, cols), dtype=np.uint8)

    for point_index in range(1, len(nose_left_fit_point)):
        point1 = (nose_left_fit_point[point_index-1][0], nose_left_fit_point[point_index-1][1])
        point2 = (nose_left_fit_point[point_index][0], nose_left_fit_point[point_index][1])

        cv2.line(img, point1, point2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(nose_fit_label, point1, point2, 255, 2, cv2.LINE_AA)

    for point_index in range(1, len(nose_right_fit_point)):
        point1 = (nose_right_fit_point[point_index-1][0], nose_right_fit_point[point_index-1][1])
        point2 = (nose_right_fit_point[point_index][0], nose_right_fit_point[point_index][1])

        cv2.line(img, point1, point2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(nose_fit_label, point1, point2, 255, 2, cv2.LINE_AA)

    for point_index in range(1, len(nose_lower_fit_point)):
        point1 = (nose_lower_fit_point[point_index-1][0], nose_lower_fit_point[point_index-1][1])
        point2 = (nose_lower_fit_point[point_index][0], nose_lower_fit_point[point_index][1])

        cv2.line(img, point1, point2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(nose_fit_label, point1, point2, 255, 2, cv2.LINE_AA)

    cv2.imwrite('../data/temp/nose_merge/nose_all_test_' + img_name + '.jpg', img)
    cv2.imwrite('../data/temp/nose_merge/nose_label_test_' + img_name + '.png', nose_fit_label)

