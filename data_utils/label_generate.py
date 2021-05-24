import numpy as np
import cv2
from glob import glob

"""
    把生成各类局部label的函数都放在了这里 目前放有鼻子、虹膜 预计未来还会有嘴巴
"""
np.set_printoptions(threshold=np.inf)


def get_nose_label(label, label_rows, label_cols, img_rows, img_cols, nose_point_file_path, draw_type=0):
    """
    以下为对应关系 前者为商汤给的106点中 鼻子相关点的索引   后者为提出所有鼻子相关点后的现索引 
    14点以后是计算的点 非精确点 (当然商汤点也并没完全精确)
    鼻梁 43-46, 鼻子下沿 47-51, 鼻子外侧 78-83
    43  44  45  46  47  48  49  50  51  78  79  80  81  82  83  46与49中心点  48与80中心点  50与81中心点
    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15(3,6)      16(5,11)     17(7,12)
    """
    # 鼻子索引 
    nose_index_list = [43, 44, 45, 46, 47, 48, 49, 50, 51, 78, 79, 80, 81, 82, 83]

    # 关键点的坐标是针对原图的 很多label的尺寸和对应原图的尺寸不是同一尺寸 需要对关键点坐标值进行调整
    point_x_resize, point_y_resize = label_rows / img_rows, label_cols / img_cols

    # 新增三点 为三对两点的中心点
    new_point1_index_list, new_point2_index_list = [3, 5, 7], [6, 11, 12]
    nose_point_list = get_nose_point(nose_point_file_path, point_x_resize, point_y_resize,
                                     new_point1_index_list, new_point2_index_list, nose_index_list)

    # 因为有的图片是有遮挡的 会存在无关键点的情况 则对label不做处理 直接返回无鼻子轮廓的label
    if nose_point_list is None:
        return label

    if draw_type:
        # 如果打算绘制非拟合鼻子(鼻子轮廓为直线型 非曲线型)   则输入鼻子关键点索引 需要分成组 每组内前后两点相连

        """
            line_point_index_list = [[0, 1, 2, 3, 6],
                                    [4, 5, 6, 7, 8],
                                    [9, 11, 13, 4],
                                    [10, 12, 14, 8]]
            该示例将绘制直线鼻梁 直线鼻左侧 直线鼻右侧 直线鼻下沿
            试了几版 截取部分直线鼻梁与鼻侧 完全舍弃鼻下沿的情况较美观一些
            所以下面的索引只有三组：
                2、3点构成的一小截鼻梁 基本上是鼻梁中部到鼻尖
                11、13、4、5点构成的半包围左鼻侧 会舍弃鼻侧上半截 同时包含一点鼻下沿
                12、14、8、7点构成的半包围右鼻侧 同上
        """
        # line_point_index_list = [[2, 3], [11, 13, 4, 5], [12, 14, 8, 7]]
        line_point_index_list = [[2, 3], [], []]
        label = draw_nose_line(label, nose_point_list, line_point_index_list, thickness=1)

    # 绘制拟合鼻子轮廓 用传入的三组点(三组分别为左鼻侧、右鼻侧、鼻下沿)分别进行多项式拟合 然后通过插值(点距为3) 获得更多点坐标 并将这些点相连
    fit_point_index_list = [[11, 13, 4], [12, 14, 8], []]
    label = draw_nose_fit(label, nose_point_list, fit_point_index_list, thickness=1)

    return label


def get_contour_pupil_label(label, contour_point_file_path, img_rows, img_cols):
    label_rows, label_cols = label.shape
    iris_label, _ = draw_contour_pupil(contour_point_file_path, img_rows, img_cols, label_rows, label_cols)

    for row in range(label_rows):
        for col in range(label_cols):
            if (label[row, col] == 4 or label[row, col] == 5) and iris_label[row, col] == 255:
                label[row, col] = 19

    return label


def nose_side_fit(point_list):
    """
    鼻侧x、y颠倒拟合二次函数 以3为点距进行插值 获得插值后的x、y值
    :param point_list:
    :return:
    """
    point_list = np.array(point_list)
    point_x = point_list[:, 1]
    point_y = point_list[:, 0]

    fit_point_x, fit_point_y = fit_interpolation(point_x, point_y, 2, point_x[0], point_x[-1], 3)

    fit_point = np.empty(shape=(len(fit_point_x), 2), dtype=np.int32)
    fit_point[:, 1] = fit_point_x
    fit_point[:, 0] = fit_point_y

    return fit_point


def nose_lower_fit(point_list):
    """
    鼻下沿拟合4次函数 以3为点距 以五点组成的四段中的边缘两段的中点分别作为插值起始点 获得x、y值
    :param point_list:
    :return:
    """
    point_list = np.array(point_list)
    point_x = point_list[:, 0]
    point_y = point_list[:, 1]

    inter_point_left_x = (point_x[0] + point_x[1]) // 2
    inter_point_right_x = (point_x[3] + point_x[4]) // 2

    fit_point_x, fit_point_y = fit_interpolation(point_x, point_y, 4, inter_point_left_x, inter_point_right_x, 3)

    fit_point = np.empty(shape=(len(fit_point_x), 2), dtype=np.int32)
    fit_point[:, 0] = fit_point_x
    fit_point[:, 1] = fit_point_y

    return fit_point


def fit_interpolation(point_x, point_y, polynomial_degree, inter_left, inter_right, inter_space):
    function = np.polyfit(point_x, point_y, polynomial_degree)
    polynomial = np.poly1d(function)

    fit_point_x = np.arange(inter_left, inter_right, inter_space, dtype=np.int32)
    fit_point_y = polynomial(fit_point_x)

    return fit_point_x, fit_point_y


def get_nose_point(point_file_path, point_x_resize, point_y_resize, new_point1_index_list, new_point2_index_list,
                   nose_index_list):
    all_point_list = get_point(point_file_path, point_x_resize, point_y_resize)
    if len(all_point_list) == 0:
        return None
    nose_point_list = extract_index_point(all_point_list, nose_index_list)

    if len(nose_point_list) == 15 and new_point1_index_list is not None and new_point2_index_list is not None:
        # 自增点  获得46与49中心点、48与80中心点、50与81中心点
        assert len(new_point1_index_list) == len(new_point2_index_list)
        for point1_index, point2_index in zip(new_point1_index_list, new_point2_index_list):
            point1, point2 = nose_point_list[point1_index], nose_point_list[point2_index]
            new_point_x = (point1[0] + point2[0]) / 2
            new_point_y = (point1[1] + point2[1]) / 2
            nose_point_list.append([new_point_x, new_point_y])

    return nose_point_list


def get_point(point_file_path, point_x_resize, point_y_resize):
    point_list = []
    with open(point_file_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            line = line.strip('\n')
            if line == 'end':
                break
            point_x_f = float(line.split(',')[0]) * point_x_resize
            point_y_f = float(line.split(',')[1]) * point_y_resize
            point_list.append([point_x_f, point_y_f])
    if f:
        f.close()

    return point_list


def draw_line(label, point_list, color, thickness):
    for index in range(1, len(point_list)):
        point1 = (int(point_list[index-1][0]), int(point_list[index-1][1]))
        point2 = (int(point_list[index][0]), int(point_list[index][1]))

        cv2.line(label, point1, point2, color, thickness, cv2.LINE_AA)

    return label


def extract_index_point(point_list, point_index_list):
    extract_point_list = []
    for point_index in point_index_list:
        extract_point_list.append(point_list[point_index])

    return extract_point_list


def draw_nose_fit(label, nose_point_list, fit_point_index_list, thickness):
    if len(fit_point_index_list[0]) != 0:
        nose_left_point_list = extract_index_point(nose_point_list, fit_point_index_list[0])
        nose_left_fit_point = nose_side_fit(nose_left_point_list)
        label = draw_line(label, nose_left_fit_point, 255, thickness)
    if len(fit_point_index_list[1]) != 0:
        nose_right_point_list = extract_index_point(nose_point_list, fit_point_index_list[1])
        nose_right_fit_point = nose_side_fit(nose_right_point_list)
        label = draw_line(label, nose_right_fit_point, 255, thickness)
    if len(fit_point_index_list[2]) != 0:
        nose_lower_point_list = extract_index_point(nose_point_list, fit_point_index_list[2])
        nose_lower_fit_point = nose_lower_fit(nose_lower_point_list)
        label = draw_line(label, nose_lower_fit_point, 255, thickness)

    return label


def draw_nose_line(label, nose_point_list, line_point_index_list, thickness):
    if len(line_point_index_list[0]) != 0:
        nose_bridge_point_list = extract_index_point(nose_point_list, line_point_index_list[0])
        label = draw_line(label, nose_bridge_point_list, 255, thickness)
    if len(line_point_index_list[1]) != 0:
        nose_left_point_list = extract_index_point(nose_point_list, line_point_index_list[1])
        label = draw_line(label, nose_left_point_list, 255, thickness)
    if len(line_point_index_list[2]) != 0:
        nose_right_point_list = extract_index_point(nose_point_list, line_point_index_list[2])
        label = draw_line(label, nose_right_point_list, 255, thickness)

    return label


def draw_contour_pupil(point_file_path, img_rows, img_cols, label_rows, label_cols):
    contour_left, contour_right, center_left, center_right = get_eye_point(point_file_path)

    iris_label = np.zeros(shape=(img_rows, img_cols), dtype=np.uint8)
    center_label = np.zeros(shape=(img_rows, img_cols), dtype=np.uint8)  # 暂不做实现  返回为空

    if contour_left is None or contour_right is None:
        return iris_label, center_label
    if contour_left.shape != (1, 19, 2) or contour_right.shape != (1, 19, 2):
        return iris_label, center_label

    cv2.fillPoly(iris_label, contour_left, 255)
    cv2.fillPoly(iris_label, contour_right, 255)
    iris_label = cv2.resize(iris_label, dsize=(label_rows, label_cols), interpolation=cv2.INTER_NEAREST)

    return iris_label, center_label


def get_eye_point(point_file_path):
    all_point_list = get_point(point_file_path, 1, 1)
    if len(all_point_list) == 0:
        return None, None, None, None

    temp_left_x_sum, temp_left_y_sum, temp_right_x_sum, temp_right_y_sum = 0, 0, 0, 0
    contour_left, contour_right = [], []

    for index in range(len(all_point_list)):
        point_x, point_y = int(all_point_list[index][0]), int(all_point_list[index][1])
        if index < 19:
            contour_left.append([point_x, point_y])
            temp_left_x_sum += point_x
            temp_left_y_sum += point_y
        else:
            contour_right.append([point_x, point_y])
            temp_right_x_sum += point_x
            temp_right_y_sum += point_y
    contour_left = np.array([contour_left], dtype=np.int32)
    contour_right = np.array([contour_right], dtype=np.int32)
    center_left = [temp_left_x_sum // 19, temp_left_y_sum // 19]
    center_right = [temp_right_x_sum // 19, temp_right_y_sum // 19]

    return contour_left, contour_right, center_left, center_right
