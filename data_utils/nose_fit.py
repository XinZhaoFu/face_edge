import numpy as np

"""
    其实应该再写个函数的 就不会显得重复的这么多
"""


def nose_fit(nose_left_point_list, nose_right_point_list, nose_lower_point_list):
    """
    对两鼻侧的xy互换 做一个2次项的拟合
    对鼻下沿 做一个4次项的拟合
    :param nose_left_point_list:
    :param nose_right_point_list:
    :param nose_lower_point_list:
    :return:
    """
    nose_left_point_list = np.array(nose_left_point_list)
    nose_left_point_x = nose_left_point_list[:, 1]
    nose_left_point_y = nose_left_point_list[:, 0]

    nose_right_point_list = np.array(nose_right_point_list)
    nose_right_point_x = nose_right_point_list[:, 1]
    nose_right_point_y = nose_right_point_list[:, 0]

    nose_lower_point_list = np.array(nose_lower_point_list)
    nose_lower_point_x = nose_lower_point_list[:, 0]
    nose_lower_point_y = nose_lower_point_list[:, 1]

    # 计算左鼻侧
    f_left = np.polyfit(nose_left_point_x, nose_left_point_y, 2)
    p_left = np.poly1d(f_left)

    nose_left_fit_point_x = np.arange(nose_left_point_x[0], nose_left_point_x[-1], 3, dtype=np.int32)
    nose_left_fit_point_y = p_left(nose_left_fit_point_x)

    nose_left_fit_point = np.empty(shape=(len(nose_left_fit_point_x), 2), dtype=np.int32)
    nose_left_fit_point[:, 1] = nose_left_fit_point_x
    nose_left_fit_point[:, 0] = nose_left_fit_point_y

    # 计算右鼻侧
    f_right = np.polyfit(nose_right_point_x, nose_right_point_y, 2)
    p_right = np.poly1d(f_right)

    nose_right_fit_point_x = np.arange(nose_right_point_x[0], nose_right_point_x[-1], 3, dtype=np.int32)
    nose_right_fit_point_y = p_right(nose_right_fit_point_x)

    nose_right_fit_point = np.empty(shape=(len(nose_right_fit_point_x), 2), dtype=np.int32)
    nose_right_fit_point[:, 1] = nose_right_fit_point_x
    nose_right_fit_point[:, 0] = nose_right_fit_point_y

    # 计算鼻下沿
    f_lower = np.polyfit(nose_lower_point_x, nose_lower_point_y, 4)
    p_lower = np.poly1d(f_lower)

    nose_lower_point_left_x = (nose_lower_point_x[0] + nose_lower_point_x[1]) // 2
    nose_lower_point_right_x = (nose_lower_point_x[3] + nose_lower_point_x[4]) // 2
    nose_lower_fit_point_x = np.arange(nose_lower_point_left_x, nose_lower_point_right_x, 3, dtype=np.int32)
    nose_lower_fit_point_y = p_lower(nose_lower_fit_point_x)

    nose_lower_fit_point = np.empty(shape=(len(nose_lower_fit_point_x), 2), dtype=np.int32)
    nose_lower_fit_point[:, 0] = nose_lower_fit_point_x
    nose_lower_fit_point[:, 1] = nose_lower_fit_point_y

    return nose_left_fit_point, nose_right_fit_point, nose_lower_fit_point









