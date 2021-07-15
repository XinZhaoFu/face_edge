import numpy as np
import cv2
from random import randint, uniform
from skimage.measure import label as ski_label
from skimage.measure import regionprops

"""
    把生成各类局部label的函数都放在了这里 目前放有鼻子、虹膜 预计未来还会有嘴巴
    还有数据增强的函数也合并进来了
"""

np.set_printoptions(threshold=np.inf)


def get_nose_label(label, img_rows, img_cols, nose_point_file_path, draw_type=0):
    """
    基于商汤106关键点 绘制鼻子标签
    流程如下：
        1.通过商汤接口读取图片 获得106关键点 存储为图片名.txt文件  (在外部程序完成的该操作)
        2.读取关键点文件 下面注释中标明了鼻子所对应的关键点索引 以及索引点指代的位置 从文件中抽取鼻子各点的坐标
        3.将原106点鼻子的索引与新索引建立对应 以方便使用
        4.绘制鼻子轮廓 提供两种方式(1)直接对各点进行直接连线的绘制 (2)对各点进行拟合 获得拟合曲线 计算插值点 再连接插值点
        5.返回绘制好的label

    需要原图size是因为关键点坐标是基于原图的 和label的尺寸不一定一致 如果不一致需要进行调整

    以下为对应关系 前者为商汤给的106点中 鼻子相关点的索引   后者为提出所有鼻子相关点后的现索引
    14点以后是计算的点 非精确点 (当然商汤点也并没完全精确)
    鼻梁 43-46, 鼻子下沿 47-51, 鼻子外侧 78-83
    43  44  45  46  47  48  49  50  51  78  79  80  81  82  83  46与49中心点  48与80中心点  50与81中心点
    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15(3,6)      16(5,11)     17(7,12)

    :param label:
    :param img_rows:
    :param img_cols:
    :param nose_point_file_path:
    :param draw_type: 0为不绘制直线 1为绘制直线
    :return:
    """
    # 鼻子索引 
    nose_index_list = [43, 44, 45, 46, 47, 48, 49, 50, 51, 78, 79, 80, 81, 82, 83]

    # 关键点的坐标是针对原图的 很多label的尺寸和对应原图的尺寸不是同一尺寸 需要对关键点坐标值进行调整
    label_rows, label_cols = label.shape
    point_x_resize, point_y_resize = label_rows / img_rows, label_cols / img_cols

    # 新增三点 为三对两点的中心点
    new_point1_index_list, new_point2_index_list = [3, 5, 7], [6, 11, 12]
    nose_point_list = get_nose_point(nose_point_file_path, point_x_resize, point_y_resize,
                                     new_point1_index_list, new_point2_index_list, nose_index_list)

    # 因为有的图片是有遮挡的 会存在无关键点的情况 则对label不做处理 直接返回无鼻子轮廓的label
    if nose_point_list is None:
        return label

    if draw_type:
        """
        如果打算绘制非拟合鼻子(鼻子轮廓为直线型 非曲线型)   则输入鼻子关键点索引 需要分成组 每组内前后两点相连
        示例1：
            line_point_index_list = [[0, 1, 2, 3, 6],
                                    [4, 5, 6, 7, 8],
                                    [9, 11, 13, 4],
                                    [10, 12, 14, 8]]
            该示例将绘制直线鼻梁 直线鼻下沿 直线鼻左侧 直线鼻右侧 
            
        示例2：
            line_point_index_list = [[2, 3], [11, 13, 4, 5], [12, 14, 8, 7]]
            试了几版 截取部分直线鼻梁与鼻侧 完全舍弃鼻下沿的情况较美观一些
            所以该索引只有三组：
                2、3点构成的一小截鼻梁 基本上是鼻梁中部到鼻尖
                11、13、4、5点构成的半包围左鼻侧 会舍弃鼻侧上半截 同时包含一点鼻下沿
                12、14、8、7点构成的半包围右鼻侧 同上
        示例3：
            line_point_index_list = [[2, 3], [], []]
            可以传入空值 则该处不进行绘制
        """

        line_point_index_list = [[2, 3], [], []]
        label = draw_nose_line(label, nose_point_list, line_point_index_list, thickness=1)

    # 绘制拟合鼻子轮廓 用传入的三组点(三组分别为左鼻侧、右鼻侧、鼻下沿)分别进行多项式拟合 然后通过插值(点距为3) 获得更多点坐标 并将这些点相连
    fit_point_index_list = [[11, 13, 4], [12, 14, 8], [4, 5, 6, 7, 8]]
    # fit_point_index_list = [[], [], [4, 5, 6, 7, 8]]
    label = draw_nose_fit(label, nose_point_list, fit_point_index_list, thickness=1)

    return label


def get_contour_pupil_label(label, contour_point_file_path, img_rows, img_cols, is_canny=True):
    """
    基于商汤虹膜关键点接口 绘制虹膜轮廓与瞳孔中心点
    (瞳孔不支持 带瞳孔不好看 因为商汤提供的不是瞳孔的轮廓坐标 而是瞳孔中心点坐标 所以后面重写代码的时候 只保留了可供重写的接口但舍弃了实现)
    基本流程与绘制鼻子的流程相似

    :param is_canny:
    :param label:
    :param contour_point_file_path:
    :param img_rows:
    :param img_cols:
    :return:
    """
    label_rows, label_cols = label.shape
    iris_label, _ = draw_contour_pupil(contour_point_file_path, img_rows, img_cols, label_rows, label_cols)

    # 寻找眼睛整体与虹膜整体的交集 这么操作是因为虹膜轮廓不一定完全位于眼睛内部 会有超出的部分
    intersection_label = np.zeros(shape=(label_rows, label_cols), dtype=np.uint8)
    (rows, cols) = np.where(np.logical_or(label == 4, label == 5))
    for row, col in zip(rows, cols):
        if iris_label[row, col] == 255:
            intersection_label[row, col] = 19
    if is_canny:
        label = cv2.Canny(label, 0, 0)

        contours, _ = cv2.findContours(intersection_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(label, contours, -1, 255, 1)

        return label
    else:
        contours, _ = cv2.findContours(intersection_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours


def nose_side_fit(point_list):
    """
    鼻侧x、y颠倒拟合2次函数 以3为点距进行插值 获得插值后的x、y值 颠倒后返回
    x，y进行颠倒是为了便于拟合

    :param point_list:
    :return:
    """
    point_list = np.array(point_list)
    point_x = point_list[:, 1]
    point_y = point_list[:, 0]

    fit_point_x, fit_point_y = fit_interpolation(point_x, point_y, 2, (point_x[0] + point_x[1]) // 2, point_x[-1], 3)

    fit_point = np.empty(shape=(len(fit_point_x), 2), dtype=np.int32)
    fit_point[:, 1] = fit_point_x
    fit_point[:, 0] = fit_point_y

    return fit_point


def nose_lower_fit(point_list):
    """
    鼻下沿拟合4次函数 以3为点距 以五点组成的四段中的边缘两段的中点分别作为插值起始点 获得x、y值
    没有以给定点作为插值绘制的起始点主要是为了美观度

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
    """
    传入x坐标集合 y坐标集合 指定的多项式次幂 求出多项式曲线函数
    获得指定的插值起始点与插值间隔 返回插值后得到的密集点集

    :param point_x:
    :param point_y:
    :param polynomial_degree: 多项式次幂
    :param inter_left:
    :param inter_right:
    :param inter_space: 插值间距
    :return:
    """
    function = np.polyfit(point_x, point_y, polynomial_degree)
    polynomial = np.poly1d(function)

    fit_point_x = np.arange(inter_left, inter_right, inter_space, dtype=np.int32)
    fit_point_y = polynomial(fit_point_x)

    return fit_point_x, fit_point_y


def get_nose_point(point_file_path, point_x_resize, point_y_resize, new_point1_index_list, new_point2_index_list,
                   nose_index_list):
    """
    获得轮廓点坐标文件中的坐标 因为原图尺寸与标签图尺寸不一定一致 所以对坐标会进行放缩(不放缩时会传入1)
    对于文件中的坐标是有选择的读取的 只读取所需坐标的坐标点 还可以通过原有点集插值计算新点

    :param point_file_path:
    :param point_x_resize:
    :param point_y_resize:
    :param new_point1_index_list:
    :param new_point2_index_list:
    :param nose_index_list:
    :return:
    """
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


def get_point(point_file_path, point_x_resize=1, point_y_resize=1):
    """
    读取坐标点 如有需要则进行放缩

    :param point_file_path:
    :param point_x_resize:
    :param point_y_resize:
    :return:
    """
    point_list = []
    with open(point_file_path, "r") as f:
        for index, line in enumerate(f.readlines()):
            line = line.strip('\n')
            if line == 'end':
                break
            point_x_f = max(float(line.split(',')[0]) * point_x_resize, 0)
            point_y_f = max(float(line.split(',')[1]) * point_y_resize, 0)
            point_list.append([point_x_f, point_y_f])
    if f:
        f.close()

    return point_list


def draw_line(label, point_list, color, thickness):
    """
    传入为点集 对于点集的每一前后两点间绘制直线

    :param label:
    :param point_list:
    :param color:
    :param thickness:
    :return:
    """
    for index in range(1, len(point_list)):
        point1 = (int(point_list[index - 1][0]), int(point_list[index - 1][1]))
        point2 = (int(point_list[index][0]), int(point_list[index][1]))

        cv2.line(label, point1, point2, color, thickness, cv2.LINE_AA)

    return label


def extract_index_point(point_list, point_index_list):
    """
    抽取对应索引的坐标值

    :param point_list:
    :param point_index_list:
    :return:
    """
    extract_point_list = []
    if len(point_list) < len(point_index_list):
        # print('[error] 抽取队列大于点集队列')
        return extract_point_list

    for point_index in point_index_list:
        extract_point_list.append(point_list[point_index])

    return extract_point_list


def draw_nose_fit(label, nose_point_list, fit_point_index_list, thickness):
    """
    绘制拟合鼻子轮廓 若某一分组为空则该部位不进行绘制

    :param label:
    :param nose_point_list:
    :param fit_point_index_list:
    :param thickness:
    :return:
    """
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
    """
    绘制直线鼻子轮廓 若某一分组为空则该部位不进行绘制

    :param label:
    :param nose_point_list:
    :param line_point_index_list:
    :param thickness:
    :return:
    """
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
    """
    读取38点虹膜坐标(每只眼睛各19点)与中心点坐标
    这里没有在读取坐标的时候对坐标进行放缩 因为这些点坐标值比较密集 担心直接对坐标值进行放缩 会带来一定的误差
    目前选择直接绘制原图尺寸的虹膜标签 然后对图片进行resize

    瞳孔这里没有做实现 可以在中心点区域画圆来实现

    :param point_file_path:
    :param img_rows:
    :param img_cols:
    :param label_rows:
    :param label_cols:
    :return:
    """
    contour_left, contour_right, center_left, center_right = get_eye_point(point_file_path)

    iris_label = np.zeros(shape=(img_rows, img_cols), dtype=np.uint8)
    center_label = np.zeros(shape=(img_rows, img_cols), dtype=np.uint8)  # 暂不做实现  返回为空

    # 对数据进行检查 若为空 或数量不对 则不绘制虹膜
    if contour_left is None or contour_right is None:
        return iris_label, center_label
    if contour_left.shape != (1, 19, 2) or contour_right.shape != (1, 19, 2):
        return iris_label, center_label

    cv2.fillPoly(iris_label, contour_left, 255)
    cv2.fillPoly(iris_label, contour_right, 255)
    iris_label = cv2.resize(iris_label, dsize=(label_rows, label_cols), interpolation=cv2.INTER_NEAREST)

    return iris_label, center_label


def get_eye_point(point_file_path):
    """
    获得眼部坐标

    :param point_file_path:
    :return:
    """
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


def get_iris_fit_ellipse(label, is_canny=True):
    """
    读取分割标签，获得canny后的边缘标签记作edge_label，分割标签读取左右眼区域，获得轮廓，进行椭圆拟合获得椭圆中心点坐标以及长短轴直径
    以椭圆中心点坐标作为假定虹膜中心点坐标，以短轴半径微调作为假定虹膜半径
    在空npArray上绘制该假定虹膜

    :param is_canny:
    :param label:
    :return:
    """
    eyes_label = np.zeros(shape=label.shape, dtype=np.uint8)
    (rows, cols) = np.where(np.logical_and(label == 4, label == 5))
    # for row, col in zip(rows, cols):
    #     eyes_label[row, col] = 255
    eyes_label[rows, cols] = 255

    contours, hierarchy = cv2.findContours(eyes_label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if is_canny:
        label = cv2.Canny(label, 0, 0)

    for contour in contours:
        if len(contour) < 5:
            continue
        # 如果关键点小于5 无法确定眼睛轮廓  则不绘制
        (x, y), (a, b), angle = cv2.fitEllipse(contour)  # (x, y)中心点 (a, b)长短轴直径 angle中心旋转角度
        rad = int(min(a, b) * 0.5)
        if rad <= 7:
            rad += 1
        else:
            rad = int(rad * 1.25)
        label = cv2.circle(label, center=(int(x), int(y)), radius=rad, color=255, thickness=1)

    return label


def get_centroid_bbox(label):
    """
    获得连通域属性

    :param label:
    :return:
    """
    labels, labels_num = ski_label(label, background=0, return_num=True)
    regions = regionprops(labels)  # 获取各个联通域的属性
    if labels_num == 0:
        return None, None
    if labels_num == 1:
        return regions[0].centroid, regions[0].bbox

    max_area_index = 0
    max_area = regions[0].area
    for index, region in enumerate(regions):
        if region.area > max_area:
            max_area_index = index
            max_area = region.area
    return regions[max_area_index].centroid, regions[max_area_index].bbox


def get_nose_by_seg_label(label, small_rate=1.0):
    """
    获得放缩的鼻子轮廓 这里是基于鼻子逐像素标注的标签来实现的
    将所有的鼻子像素点复制到一张空白图上 在空白图上进行放缩
    计算原图鼻子的质心和新图鼻子的质心 将原图鼻子标注值变为皮肤的标注值 将新鼻子放到对应质心位置处

    :param small_rate:
    :param label:
    :return:
    """
    nose_label = np.zeros(shape=label.shape, dtype=np.uint8)
    rows, cols = label.shape

    (tar_rows, tar_cols) = np.where(label == 6)
    label[tar_rows, tar_cols] = 1
    nose_label[tar_rows, tar_cols] = 6

    # 暂时没用bbox信息 感觉写起来麻烦
    ori_centroid, _ = get_centroid_bbox(nose_label)
    ori_centroid_row, ori_centroid_col = int(ori_centroid[0]), int(ori_centroid[1])
    temp = np.zeros(shape=label.shape, dtype=np.uint8)
    s_rows, s_cols = int(small_rate * rows), int(small_rate * cols)
    nose_label = cv2.resize(nose_label, dsize=(s_cols, s_rows), interpolation=cv2.INTER_NEAREST)

    new_centroid_row, new_centroid_col = int(small_rate * ori_centroid_row), int(small_rate * ori_centroid_col)

    temp[ori_centroid_row - new_centroid_row: ori_centroid_row - new_centroid_row + s_rows,
         ori_centroid_col - new_centroid_col: ori_centroid_col - new_centroid_col + s_cols] = nose_label[:, :]

    (temp_rows, temp_cols) = np.where(temp == 6)
    label[temp_rows, temp_cols] = 6

    return label


def check_coordinate(coordinate):
    if coordinate[1] <= coordinate[0] or coordinate[3] <= coordinate[2]:
        return False
    return True


def get_class_code(class_label):
    dict_class_code = {'skin': 1, 'l_brow': 2, 'r_brow': 3, 'l_eye': 4, 'r_eye': 5, 'nose': 6, 'u_lip': 7,
                       'mouth': 8, 'l_lip': 9, 'hair': 10, 'neck': 11, 'l_ear': 12, 'r_ear': 13, 'cloth': 14,
                       'ear_r': 15, 'hat': 16, 'eye_g': 17, 'neck_l': 18}
    return dict_class_code[class_label]


def code_label(label, class_code, is_nose=True):
    """
    转为数字标签
    鼻子的编码为6 若不要原生鼻子 则将其转为皮肤 皮肤编码为1

    :param is_nose:
    :param label:
    :param class_code:
    :return:
    """
    if not is_nose and class_code == 6:
        class_code = 1
    (rows, cols) = np.where(label == 255)
    label[rows, cols] = class_code
    return label


def overlay_label(priority_labels, priority_labels_class_code):
    """
    将所有的label依次进行覆盖

    :param priority_labels_class_code:
    :param priority_labels:
    :return:
    """
    con_label = np.array(priority_labels[0], dtype=np.uint8)
    for label, code in zip(priority_labels[1:], priority_labels_class_code[1:]):
        (rows, cols) = np.where(label == code)
        con_label[rows, cols] = code

    return con_label


def get_lower_nose_edge(semantic_label):
    """
    通过分割label获得鼻子下半部分

    :param semantic_label:
    :return:
    """
    nose_label = np.zeros(shape=semantic_label.shape, dtype=np.uint8)
    (rows, cols) = np.where(semantic_label == 6)
    nose_label[rows, cols] = 255
    semantic_label[rows, cols] = 1
    label_edge = cv2.Canny(semantic_label, 0, 0)

    _, bbox = get_centroid_bbox(nose_label)
    if bbox is None:
        return label_edge
    nose_edge = cv2.Canny(nose_label, 0, 0)
    temp_rows = bbox[0] + (bbox[2] - bbox[0]) // 3
    nose_edge[:temp_rows, :] = 0
    (rows, cols) = np.where(nose_edge == 255)
    label_edge[rows, cols] = 255

    return label_edge


def concat_label(labels, class_codes, priority=None, is_nose=True):
    """
    合并各类数字标签
    同时需要注意遮盖问题 目前的优先级是预估的 不一定是准确的
    labels 和 class_codes的索引需要对应

    :param is_nose:
    :param priority:
    :param class_codes:
    :param labels:
    :return:
    """
    if priority is None:
        priority = (1, 14, 11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18)
    priority_labels = []
    priority_labels_class_code = []

    # 调整传入label的顺序
    for pri_code in priority:
        if pri_code in class_codes:
            index_class_codes = class_codes.index(pri_code)
            label = labels[index_class_codes]
            label = code_label(label, pri_code, is_nose=is_nose)
            priority_labels.append(label)
            priority_labels_class_code.append(pri_code)

    con_label = overlay_label(priority_labels, priority_labels_class_code)

    return con_label
